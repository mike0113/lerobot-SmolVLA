#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import datetime
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Float32MultiArray

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import numpy as np
import cv2

# 強制離線模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


MODEL_DIR = "/home/min/Desktop/forRTK/model/100000/pretrained_model"
#MODEL_DIR = "/home/nvidia/smolvla/lerobot/min/model/0714_5hz_addWaypoints_merge_output-1/checkpoints/040000/pretrained_model"
#DATASET_ROOT = "/home/nvidia/smolvla/lerobot/min/dataset/data_v3(80set)_5hz_baseball_GoBack_addwaypoints_merge"
DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_merge"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

OUTPUT_DIR = Path("ros2_inference_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 1. 建立一個與發布者相容的 QoS 設定 (馨慧姐)
video_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)


class VLAInferenceNode(Node):
    def __init__(self):
        super().__init__('vla_inference_node')
        
        self.latest_task = "Fly to the target waypoint..."
        self.latest_state = None
        self.inference_count = 0
        
        self.get_logger().info("⏳ [測試模式] 正在載入 Dataset 以獲取 Ground Truth...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT, video_backend="pyav")
        
        self.received_frame_count = 0       
        self.last_processed_frame_count = -1  # 🌟 [新增] 紀錄上一次推論用掉的照片編號
        self.latest_image_msg = None        
        self.latest_frame_idx = 0  # 這是跟發送端完美同步的影格號碼         
        self.new_image_event = threading.Event() 

        # 實機專用的 GT 同步計數器 (以 5Hz 為基準)
        self.gt_sync_frame_idx = 0

        # 紀錄無人機的真實物理軌跡
        self.global_pred_state = np.array([0.0, 0.0])
        self.global_pred_path = [[0.0, 0.0]] 
        
        self.action_queue = []                   
        self.queue_lock = threading.Lock()       
        self.failsafe_action = [0.0, 0.0, 0.0]   
        
        # [MODIFIED] 新增推論計步器與觸發事件 [source: 1]
        self.steps_since_last_inference = 0
        self.need_inference_event = threading.Event()
        self.need_inference_event.set() # 初始設定為 Set，讓第一筆推論馬上啟動

        # ==========================================
        # 📁 實飛資料 Log 設定 (記錄輸入與輸出)
        # ==========================================
        self.session_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"flight_logs/run_waypoints_{self.session_time}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.img_log_dir = self.log_dir / "images"
        self.img_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "inference_log.jsonl"
        self.get_logger().info(f"📁 實飛 Log 將自動保存在: {self.log_dir}")


        # 建立送貨員 Timer (0.2 秒發送一次)
        self.timer_period = 0.2
        self.action_timer = self.create_timer(self.timer_period, self.action_timer_callback)

        # --- 模型載入 ---
        self.get_logger().info(f"⏳ 正在載入 SmolVLA 模型...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        
        self.model = SmolVLAPolicy.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_preprocessor.json")
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_postprocessor.json")
        self.get_logger().info("✅ 模型載入完畢！")

        self.task_sub = self.create_subscription(String, '/task_name', self.task_callback, 10)
        self.state_sub = self.create_subscription(String, '/sensor_data', self.state_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage, '/front_image/compressed', self.image_callback,video_qos)
        self.action_pub = self.create_publisher(String, '/action', 10)
        
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()
        self.get_logger().info("🎧 櫃台已開啟 (智慧同步模式)，等待影像訊號...")

    def task_callback(self, msg):
        # 🚨 解析發送端傳來的隱藏 Frame Index，確保時間軸 100% 同步！
        if msg.data.startswith("FRAME:"):
            parts = msg.data.split("|", 1)
            self.latest_frame_idx = int(parts[0].replace("FRAME:", ""))
            self.latest_task = parts[1]
        else:
            self.latest_task = msg.data

    def state_callback(self, msg):
        # 🚨 解析實機飛控傳來的逗號字串
        try:
            # 飛控傳來的 total_data 包含 10 個值：
            # [0: GPS_lat, 1: GPS_lon, 2: GPS_alt, 3: NED_x, 4: NED_y, 5: NED_z, 6: YAW, 7: tx, 8: ty, 9: tz]
            data_list = [float(x) for x in msg.data.split(',')]
            if len(data_list) >= 7:
                # 這裡我們擷取 [NED_x, NED_y, NED_z, YAW] 作為大腦的 state 輸入 (index 3, 4, 5, 6)
                self.latest_state = torch.tensor([data_list[3], data_list[4], data_list[5], data_list[6], data_list[7], data_list[8]], dtype=torch.float32)
                #self.latest_target = torch.tensor([data_list[7], data_list[8]], dtype=torch.float32)
        except Exception as e:
            self.get_logger().warn(f"解析 sensor_data 失敗: {e}")


    def image_callback(self, msg):
        self.latest_image_msg = msg
        self.received_frame_count += 1
        self.new_image_event.set()


    def action_timer_callback(self):
        if self.received_frame_count == 0:
            return

        current_action = self.failsafe_action 
        with self.queue_lock:
            if len(self.action_queue) > 0:
                current_action = self.action_queue.pop(0) 
                
                # 如果已經執行到最後一步（懸停），把它塞回去無限輪迴
                if len(self.action_queue) == 0:
                    self.action_queue.append(current_action)
            
            # Timer 每次發送，就記錄一次無人機真實移動的足跡
            self.global_pred_state[0] += current_action[0]
            self.global_pred_state[1] += current_action[1]
            self.global_pred_path.append(self.global_pred_state.copy())
            
            # 控制 GT 前進的節拍器 (完美的 5Hz)
            # 只要無人機有在動，GT 就往前推進；無人機懸停睡覺，GT 就跟著暫停！
            is_hovering = (abs(current_action[0]) < 1e-4) and (abs(current_action[1]) < 1e-4)
            if not is_hovering:
                self.gt_sync_frame_idx += 1

            # [MODIFIED] 當步數執行到第 7 步時，觸發下一次推論 [source: 1]
            self.steps_since_last_inference += 1
            self.get_logger().info(f"步數計數器: {self.steps_since_last_inference}")
            if self.steps_since_last_inference == 7:
                self.need_inference_event.set()
                self.get_logger().info("✅ 步數達到 7，觸發下一次推論！")

        # 🚨 [通訊升級] 將動作打包成實機飛控要求的字串格式: "x, y, z, heading"
        # 模型輸出是 [dx, dy, heading]，我們補上 dz = 0.0
        action_str = f"{current_action[0]:.4f},{current_action[1]:.4f},0.0000,{current_action[2]:.4f}"

        msg = String()
        msg.data = action_str
        self.action_pub.publish(msg)

    def inference_worker(self):
        while rclpy.ok():

            # 1. 等待「需要推論」的信號 (第7步觸發，或落後時手動觸發)
            self.need_inference_event.wait()
            #self.get_logger().info("11111")

            # 🌟 [漏洞修復區塊開始] 檢查影像是否真的有更新！
            self.get_logger().info(f"收到推論觸發信號，檢查是否有新影像可用 (已收到 {self.received_frame_count} 張照片，已處理 {self.last_processed_frame_count} + 1 張照片)")
            if self.received_frame_count <= self.last_processed_frame_count:
                # 把照片事件清空，準備等待全新的照片
                self.new_image_event.clear()

                # 等待新照片，如果 0.5 秒都沒進來，代表相機斷線或發送端暫停了
                if not self.new_image_event.wait(timeout=0.5):
                    self.get_logger().info("⏳ 觸發信號已響，但等待相機新畫面中... (避免重複推論舊圖)", throttle_duration_sec=2.0)
                    # ⚠️ 關鍵：直接 continue 回去，不要 clear() 觸發信號！
                    # 這樣下一圈它會秒過 need_inference_event.wait()，繼續在這裡安全地等新圖
                    continue
            # 🌟 [漏洞修復區塊結束]


            # 2. 直接使用已經進來的最新影像與狀態！
            # 必須確保「影像」與「狀態 (sensor_data)」兩者都已連線，才能開始
            if self.latest_image_msg is None or self.latest_state is None:
                if self.inference_count == 0:
                    self.get_logger().info("等待飛控 sensor_data 及影像傳入中...", throttle_duration_sec=2.0)       #每 2 秒鐘最多只能印出一次
                # 稍微等一下第一張影像
                self.new_image_event.wait(timeout=0.5)
                #self.get_logger().info("22222")
                continue

            self.get_logger().info("✅ 收到推論觸發信號，準備進行新一輪推論...")

            # 3. 🌟 更新處理紀錄，代表這張熱騰騰的新照片我們正式收下了！
            self.last_processed_frame_count = self.received_frame_count

            #self.get_logger().info("33333")
            # 4. 確定資料雙雙齊全，正式開始推論，把觸發信號清空！
            self.need_inference_event.clear()   # 收到信號後立刻重置

            with self.queue_lock:
                state_at_capture = self.global_pred_state.copy()
                snap_pred_path = list(self.global_pred_path)

            # 直接提取當下緩衝區裡最新的一張照片與狀態
            msg = self.latest_image_msg
            task = self.latest_task
            state = self.latest_state

            #self.get_logger().info("44444")
            # 🟢 決定畫圖用的正確 Frame Index
            if self.latest_task.startswith("FRAME:"):
                current_frame_idx = self.latest_frame_idx  # 如果接的是模擬器，聽模擬器的
            else:
                current_frame_idx = self.gt_sync_frame_idx # 如果接的是真實飛控，聽 5Hz 節拍器的

            start_time = time.perf_counter()
            
            try:
                # 1. 影像解碼邏輯 (支援 CompressedImage)
                np_arr = np.frombuffer(msg.data, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                
                raw_batch = {"observation.images.main": img_tensor.unsqueeze(0), "task": [task]}
                if state is not None: 
                    raw_batch["observation.state"] = state.unsqueeze(0)


                batch_for_policy = self.preprocessor(raw_batch)
                for k, v in batch_for_policy.items():
                    if isinstance(v, torch.Tensor):
                        batch_for_policy[k] = v.to(DEVICE, dtype=DTYPE if torch.is_floating_point(v) else None)

                #self.get_logger().info("55555")


                # 2. 模型推論
                with torch.inference_mode():
                    if DEVICE == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            action_pred = self.model.predict_action_chunk(batch_for_policy)
                    else:
                        action_pred = self.model.predict_action_chunk(batch_for_policy)

                action_pred_dict = self.postprocessor({"action": action_pred})
                pred_actions = np.squeeze(action_pred_dict["action"].detach().float().cpu().numpy(), axis=0) 
                
                infer_duration = time.perf_counter() - start_time
                
                # [MODIFIED] 算出推論時間並換算成消耗掉的步數 [source: 1]
                raw_skip_steps = int(infer_duration / self.timer_period)

                # 舊推論在第 7 步觸發，代表舊陣列最多只剩 16 - 7 = 9 步可以走。
                # 走完 9 步後機器會原地懸停，所以新推論最多也只該跳過 9 步。
                max_skip_steps = 16 - 7 
                skip_steps = min(raw_skip_steps, max_skip_steps)

                # 🌟 [核心修正] 首次推論保護機制
                # 因為第一次推論期間，無人機一直待在原地懸停，並沒有「預先執行」任何動作，
                # 所以新陣列必須保留完整的 16 步，一格都不能砍！
                if self.inference_count == 0:
                    skip_steps = 0
                    self.get_logger().info("🚀 首次推論完成，無人機即將起步，完整保留 16 步！")

                self.get_logger().info(f"推論耗時 {infer_duration:.3f}s，將砍掉前 {skip_steps} 步，從新推論的第 {skip_steps + 1} 步開始執行")

                #self.get_logger().info("66666")

                # ==========================================
                # 📝 記錄 Log 資料 (影像 + 輸入狀態 + 輸出預測)
                # ==========================================
                try:
                    # 儲存輸入照片
                    img_filename = f"infer_{self.inference_count:04d}_frame_{current_frame_idx:04d}.jpg"
                    img_save_path = self.img_log_dir / img_filename
                    cv2.imwrite(str(img_save_path), img_bgr)
                    
                    # 儲存推論的輸入與輸出
                    log_entry = {
                        "inference_count": self.inference_count,
                        "frame_idx": current_frame_idx,
                        "timestamp": time.time(),
                        "task": task,
                        "state_input": state.cpu().numpy().tolist() if state is not None else [],
                        #"target_input": target.cpu().numpy().tolist() if target is  not None else [],
                        "pred_actions_output": pred_actions.tolist(),
                        "infer_duration_sec": infer_duration,
                        "raw_skip_steps": raw_skip_steps, # 記錄實際耗時算出的步數
                        "applied_skip_steps": skip_steps, # 記錄最終砍掉的步數
                        "image_file": img_filename
                    }
                    # "a" 代表 append (附加寫入)，斷電也不怕前面的資料不見
                    with open(self.log_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    self.get_logger().warn(f"寫入 Log 失敗: {e}")

                #self.get_logger().info("77777")
                # 3. 準備放入佇列
                raw_new_queue = []
                for i in range(len(pred_actions)):
                    dx = float(pred_actions[i, 0])
                    dy = float(pred_actions[i, 1])
                    heading = float(pred_actions[i, 2])
                    raw_new_queue.append([dx, dy, heading])
                    
                #last_heading = float(pred_actions[-1, 2])
                raw_new_queue.append([0.0, 0.0, 0.0]) # 結尾懸停 (這會觸發發送端暫停)
                
                # 砍掉推論期間消耗的相對應步數 (最多砍 9 步)
                new_queue = raw_new_queue[skip_steps:]
                
                #self.get_logger().info("8.8.8.8.8")
                with self.queue_lock:
                    self.action_queue = new_queue.copy()
                    # 因為砍掉前 skip_steps 步，對這份新推論而言相當於已經跑了 skip_steps 步
                    self.steps_since_last_inference = skip_steps

                    # [核心修正] 如果砍掉的步數已經 >= 7 (例如 9)，代表已經達到了下一次推論的標準
                    # 我們直接手動喚醒 Event，讓 while 迴圈一進入下一圈就立刻開始抓新影像推論！
                    if self.steps_since_last_inference >= 7:
                        self.need_inference_event.set()

                #self.get_logger().info("88888")
                # ==========================================
                # 🎨 畫圖資料準備與背景 Thread
                # ==========================================
                gt_actions_list = []
                for i in range(16):
                    idx = current_frame_idx + i
                    if idx < len(self.dataset):
                        gt_actions_list.append(self.dataset[idx]["action"].numpy()[:3])
                    else:
                        break
                gt_actions = np.array(gt_actions_list)
                
                # 精確計算 GT 歷史：只有在「移動」的 Frame 才累加
                gt_hist_path = [[0.0, 0.0]]
                curr_gt = np.array([0.0, 0.0])
                for i in range(current_frame_idx):
                    if i < len(self.dataset):
                        curr_gt += self.dataset[i]["action"].numpy()[:2]
                        gt_hist_path.append(curr_gt.copy())
                gt_state_at_capture = curr_gt.copy()
                
                plot_thread = threading.Thread(
                    target=self.save_prediction_plot, 
                    args=(pred_actions, gt_actions, infer_duration, current_frame_idx, 
                            state_at_capture, snap_pred_path, gt_state_at_capture, gt_hist_path)
                )
                plot_thread.start()
                self.inference_count += 1

                #self.get_logger().info("99999")
                # [MODIFIED] 移除停頓期 (time.sleep)，讓機器無縫接軌繼續飛行 [source: 1]
                self.get_logger().info(f"[Frame {current_frame_idx:04d}] 推論耗時 {infer_duration:.3f}s，"
                                        f"砍去前 {skip_steps} 步，無縫銜接從新推論的第 {skip_steps + 1} 步開始執行")
                if skip_steps == max_skip_steps:
                    self.get_logger().warn("⚠️ 推論時間過長，已達到最大削減步數限制，將立刻啟動下一次推論！")
                
                #self.get_logger().info("10101010")


            except Exception as e:
                self.get_logger().error(f"推論發生錯誤: {e}")
                time.sleep(1.0)

    def save_prediction_plot(self, pred_actions, gt_actions, duration, frame_idx, state_at_capture, snap_pred_path, gt_state_at_capture, gt_hist_path):
        plt.figure(figsize=(8, 8))
        plt.title(f"Global Trajectory @ Frame {frame_idx}\n(Synced Simulator Mode | Latency: {duration:.3f}s)", fontsize=12, fontweight='bold')

        global_pred_arr = np.array(snap_pred_path)
        global_gt_arr = np.array(gt_hist_path)
        
        if len(global_gt_arr) > 0:
            plt.plot(global_gt_arr[:, 0], global_gt_arr[:, 1], 'g-', label='History (GT)', linewidth=2.5, alpha=0.5)
        if len(global_pred_arr) > 0:
            plt.plot(global_pred_arr[:, 0], global_pred_arr[:, 1], 'r-', label='History (Pred Real)', linewidth=2.5, alpha=0.5)

        plt.scatter(gt_state_at_capture[0], gt_state_at_capture[1], color='darkgreen', s=100, zorder=5, marker='s', label='GT State @ Capture')
        plt.scatter(state_at_capture[0], state_at_capture[1], color='orange', s=120, zorder=8, marker='D', label='Pred Start (Capture)')

        if len(gt_actions) > 0:
            future_gt_x = np.cumsum(gt_actions[:, 0]) + gt_state_at_capture[0]
            future_gt_y = np.cumsum(gt_actions[:, 1]) + gt_state_at_capture[1]
            future_gt_x = np.insert(future_gt_x, 0, gt_state_at_capture[0])
            future_gt_y = np.insert(future_gt_y, 0, gt_state_at_capture[1])
            plt.plot(future_gt_x, future_gt_y, 'g--o', label='Future 16 (GT)', linewidth=1.5, markersize=4, alpha=0.9)

        if len(pred_actions) > 0:
            future_pred_x = np.cumsum(pred_actions[:, 0]) + state_at_capture[0]
            future_pred_y = np.cumsum(pred_actions[:, 1]) + state_at_capture[1]
            future_pred_x = np.insert(future_pred_x, 0, state_at_capture[0])
            future_pred_y = np.insert(future_pred_y, 0, state_at_capture[1])
            plt.plot(future_pred_x, future_pred_y, 'r--X', label='Future 16 (Pred)', linewidth=2.5, markersize=6, alpha=0.9)

        plt.scatter(0, 0, color='blue', s=100, zorder=7, marker='P', label='Start (0,0)')

        plt.xlabel("Accumulated X (meters)", fontsize=10)
        plt.ylabel("Accumulated Y (meters)", fontsize=10)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') 

        save_path = OUTPUT_DIR / f"global_traj_{self.inference_count:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main(args=None):
    rclpy.init(args=args)
    node = VLAInferenceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
