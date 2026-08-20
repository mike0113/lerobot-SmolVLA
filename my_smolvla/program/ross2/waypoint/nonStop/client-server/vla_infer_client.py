#!/usr/bin/env python3
import time
import threading
import datetime
import json
import base64
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import numpy as np
import cv2
from pathlib import Path

class VLAClientNode(Node):
    def __init__(self):
        super().__init__('vla_client_node')
        
        # --- 狀態變數[cite: 6] ---
        self.latest_task = "Fly to the target waypoint..."
        self.latest_state = None
        self.latest_image_msg = None
        
        # --- 連續推論與防呆控制變數[cite: 6] ---
        self.timer_period = 0.2
        self.received_frame_count = 0       
        self.last_processed_frame_count = -1 
        self.steps_since_last_inference = 0
        self.new_image_event = threading.Event() 
        self.inference_ready_event = threading.Event()
        self.inference_ready_event.set()    # first inference can be sent immediately
        
        # --- Server 通訊與等待變數 ---
        self.response_received_event = threading.Event()
        self.pending_req_id = -1
        self.received_actions = None

        # --- 推論與動作控制[cite: 6] ---
        self.action_queue = []
        self.queue_lock = threading.Lock()
        self.failsafe_action = [0.0, 0.0, 0.0]

        # --- Log 分段控制變數[cite: 6] ---
        self.inference_count = 0
        self.start_new_session() 
        
        self.last_sensor_time = time.time()
        self.sensor_count = 0
        
        # --- ROS2 通訊設定[cite: 6] ---
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(String, '/task_name', lambda msg: setattr(self, 'latest_task', msg.data), 10)
        self.create_subscription(String, '/sensor_data', self.state_callback, 10)
        self.create_subscription(CompressedImage, '/front_image/compressed', self.image_callback, qos)
        self.action_pub = self.create_publisher(String, '/action', 10)
        
        # Client 專用：發送請求與接收結果
        self.req_pub = self.create_publisher(String, '/vla/inference_req', 10)
        self.create_subscription(String, '/vla/inference_res', self.res_callback, 10)
        
        # --- 計時器與執行緒[cite: 6] ---
        self.create_timer(self.timer_period, self.action_timer_callback) 
        threading.Thread(target=self.inference_worker, daemon=True).start()
        self.get_logger().info("Client 控制端已啟動！等待飛控訊號與 Server 待命...")

    def start_new_session(self):
        """建立新的航程紀錄資料夾與初始化變數[cite: 6]"""
        self.session_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"flight_logs/run_{self.session_time}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.img_log_dir = self.log_dir / "images"
        self.img_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "inference_log.jsonl"
        self.inference_count = 0
        self.steps_since_last_inference = 0
        self.last_processed_frame_count = -1
        self.inference_ready_event.set()
        self.get_logger().info(f"建立全新航程紀錄: {self.log_dir}")

    def state_callback(self, msg):
        try:
            if self.sensor_count == 0:
                self.get_logger().info(f"✅ 收到第一筆飛控資料: {msg.data}...")

            if any(char.isalpha() for char in msg.data):
                return 

            current_time = time.time()
            if self.last_sensor_time is not None:
                if current_time - self.last_sensor_time > 10.0:
                    self.get_logger().warn("⏳ 超過 10 秒未收到訊號，重置航程！")
                    self.start_new_session()
                    with self.queue_lock:
                        self.action_queue.clear()
                    self.sensor_count = 0
            
            self.last_sensor_time = current_time
            data = [float(x) for x in msg.data.split(',')]
            self.latest_state = np.array(data[3:9], dtype=np.float32)
            self.sensor_count += 1
            
        except Exception as e:
            self.get_logger().warn(f"解析 State 失敗: {e}")

    def image_callback(self, msg):
        self.latest_image_msg = msg
        self.received_frame_count += 1
        self.new_image_event.set()

    def res_callback(self, msg):
        """接收 Server 傳回的推論結果"""
        try:
            data = json.loads(msg.data)
            # 確保這是我們正在等待的那一筆推論
            if data.get("req_id") == self.pending_req_id:
                self.received_actions = np.array(data["actions"])
                self.response_received_event.set() # 喚醒等待中的 Worker
        except Exception as e:
            self.get_logger().error(f"解析回傳結果失敗: {e}")

    def inference_worker(self):
        while rclpy.ok():
            self.inference_ready_event.wait() 
            
            # 影像防呆檢查[cite: 6]
            if self.received_frame_count <= self.last_processed_frame_count:
                self.new_image_event.clear()
                if not self.new_image_event.wait(timeout=0.5):
                    continue

            if self.latest_image_msg is None or self.latest_state is None:
                self.new_image_event.wait(timeout=0.5)
                continue
            
            self.last_processed_frame_count = self.received_frame_count
            self.inference_ready_event.clear()
            
            # === 打包並發送請求給 Server ===
            req_start_time = time.perf_counter()
            self.pending_req_id = self.inference_count
            
            # 保存當下的影像、狀態與任務，用於後續紀錄
            current_img_msg = self.latest_image_msg
            current_state = self.latest_state.copy()
            current_task = self.latest_task

            req_dict = {
                "req_id": self.pending_req_id,
                "task": current_task,
                "state": current_state.tolist(),
                "image_b64": base64.b64encode(current_img_msg.data).decode('utf-8')
            }
            req_msg = String()
            req_msg.data = json.dumps(req_dict)
            self.req_pub.publish(req_msg)
            
            # 等待 Server 回傳結果
            self.response_received_event.clear()
            if not self.response_received_event.wait(timeout=5.0):
                self.get_logger().error("❌ 等待 Server 回傳超時！請檢查 Server 是否運作中。")
                continue

            # === 收到結果，進行時空同步處理 ===
            infer_duration = time.perf_counter() - req_start_time
            actions = self.received_actions
            
            # 時空同步與削減步數邏輯[cite: 6]
            raw_skip_steps = int(infer_duration / self.timer_period)
            max_skip_steps = 16 - 7 
            skip_steps = min(raw_skip_steps, max_skip_steps)

            # 首次推論保護[cite: 6]
            if self.inference_count == 0:
                skip_steps = 0
                self.get_logger().info("🚀 首次起步推論完成，完整保留 16 步！")
            else:
                self.get_logger().info(f"🎯 Server 推論耗時 {infer_duration:.3f}s，砍掉前 {skip_steps} 步以無縫接軌")

            new_queue = actions[skip_steps:].tolist()
            
            # 更新佇列與計步器[cite: 6]
            with self.queue_lock:
                self.action_queue = new_queue
                self.steps_since_last_inference = skip_steps
                
                if self.steps_since_last_inference >= 7:
                    self.inference_ready_event.set()
                    
            # 寫入 Log (使用剛剛送出去的那張確切照片)[cite: 6]
            self.log_inference(actions, current_img_msg, current_task, current_state)
            self.inference_count += 1

    def action_timer_callback(self):
        # 預設動作為 failsafe_action[cite: 6]
        current_action = self.failsafe_action

        with self.queue_lock:
            if len(self.action_queue) > 0:
                current_action = self.action_queue.pop(0)

            # 觸發下一次推論的計步器[cite: 6]
            if self.latest_state is not None and self.latest_image_msg is not None:
                self.steps_since_last_inference += 1
                if self.steps_since_last_inference == 7:
                    self.inference_ready_event.set()
                    self.get_logger().info("⚡ 步數達到 7，向 Server 發出推論請求！")

        msg = String()
        msg.data = f"{current_action[0]:.4f},{current_action[1]:.4f},0.0000,{current_action[2]:.4f}"
        self.action_pub.publish(msg)

    def log_inference(self, actions, img_msg, task, state):
        # 將壓縮影像轉回 cv2 格式儲存
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        img_filename = f"infer_{self.inference_count:04d}.jpg"
        cv2.imwrite(str(self.img_log_dir / img_filename), img)
        
        log_entry = {
            "timestamp": timestamp,
            "count": self.inference_count,
            "task": task,
            "state_input": state.tolist(),
            "actions": actions.tolist(),
            "image": img_filename
        }
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

def main():
    rclpy.init()
    node = VLAClientNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()