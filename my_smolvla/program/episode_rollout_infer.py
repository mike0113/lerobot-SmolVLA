#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2D Rollout Inference
import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # 強制使用無頭模式存圖，避免在沒有桌面環境的 Linux 終端機報錯
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 0. 環境設定：強制離線模式
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ==========================================
# 1. 參數設定區
# ==========================================
# 你的 SmolVLA 模型權重資料夾路徑
#MODEL_DIR = "/home/min/workspace/lerobot/my_smolvla/model/0608_5hz_output-1/checkpoints/040000/pretrained_model"
MODEL_DIR = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/model/0714_5hz_addWaypoints_merge_output/checkpoints/100000/pretrained_model"

# 你的本地端 Parquet/MP4 資料集路徑
#DATASET_ROOT = "/home/min/workspace/lerobot/my_smolvla/dataset/data_v3_5hz_baseball_GoBack_2Prompt_revisedNED"
DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_merge"


# 設定推論結果的圖片輸出資料夾
OUTPUT_DIR = Path("inference_rollout_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 設定要測試的 Episode 編號
TARGET_EPISODE = 1
CHUNK_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ==========================================
# 2. 視覺化函數：畫出全路線軌跡
# ==========================================
def plot_full_episode_trajectory(global_pred_traj, global_gt_traj, inference_nodes_pred, inference_nodes_gt, ep_idx):
    global_pred_traj = np.array(global_pred_traj)
    global_gt_traj = np.array(global_gt_traj)
    inference_nodes_pred = np.array(inference_nodes_pred)
    inference_nodes_gt = np.array(inference_nodes_gt)

    plt.figure(figsize=(10, 10))
    plt.title(f"Full Episode {ep_idx} Trajectory (Chunked Rollout)", fontsize=16, fontweight='bold')

    plt.plot(global_gt_traj[:, 0], global_gt_traj[:, 1], 'g-', label='Ground Truth Path', linewidth=2, alpha=0.7)
    plt.plot(global_pred_traj[:, 0], global_pred_traj[:, 1], 'r--', label='SmolVLA Predicted Path', linewidth=2, alpha=0.7)
    
    plt.scatter(0, 0, color='blue', s=150, zorder=5, marker='*', label='Start (0,0)')
    
    plt.scatter(inference_nodes_gt[:, 0], inference_nodes_gt[:, 1], color='darkgreen', s=50, zorder=4, label='Inference Nodes (GT)')
    plt.scatter(inference_nodes_pred[:, 0], inference_nodes_pred[:, 1], color='darkred', s=50, zorder=4, marker='X', label='Inference Nodes (Pred)')

    plt.xlabel("Accumulated dx (m)", fontsize=12)
    plt.ylabel("Accumulated dy (m)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') 
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"full_episode_{ep_idx}_trajectory.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 整趟完整軌跡圖已儲存為: {save_path}")
    plt.close()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    print(f"⏳ 1. 正在載入模型與 Pipeline... (DEVICE={DEVICE})")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    model = SmolVLAPolicy.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
    model.eval() ### NEW ADD ###
    

    # 🛠️ [新增] 載入前處理器與後處理器
    try:
        preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_preprocessor.json")
    except Exception as e:
        print(f"⚠️ 無法載入 Preprocessor！錯誤訊息: {e}")
        preprocessor = None

    try:
        postprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_postprocessor.json")
    except Exception as e:
        print(f"⚠️ 無法載入 Postprocessor！錯誤訊息: {e}")
        postprocessor = None

    print("\n⏳ 2. 正在載入 LeRobotDataset...")
    #dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT, video_backend="pyav")     # on Jetson Nano, use "pyav" for better performance
    dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT)       # on PC, use default backend (ffmpeg) for better compatibility

    start_frame = None
    end_frame = None
    for i in range(len(dataset)):
        ep = int(dataset[i]["episode_index"].item()) if "episode_index" in dataset[i] else 0
        if ep == TARGET_EPISODE and start_frame is None:
            start_frame = i
        elif ep > TARGET_EPISODE:
            end_frame = i
            break
    if end_frame is None: 
        end_frame = len(dataset)

    if start_frame is None:
        raise ValueError(f"找不到 Episode {TARGET_EPISODE} 的資料！")

    print(f"\n🚀 準備開始 [全路線接力推論]！")
    print(f"   - 目標 Episode: {TARGET_EPISODE}")
    print(f"   - Frame 範圍: {start_frame} 到 {end_frame - 1}")
    print(f"   - 總共步數: {end_frame - start_frame}")

    global_pred_traj = [[0.0, 0.0, 0.0]]  
    global_gt_traj = [[0.0, 0.0, 0.0]]
    current_pred_state = np.array([0.0, 0.0, 0.0])
    current_gt_state = np.array([0.0, 0.0, 0.0])
    inference_nodes_pred = [[0.0, 0.0, 0.0]]
    inference_nodes_gt = [[0.0, 0.0, 0.0]]

    inference_times_log = [] # 🛠️ [新增] 用來記錄每次推論耗時的陣列

    print("\n" + "="*125)
    print(f" {'Global Step':<11} | {'Type':<4} | {'dx (Raw)':>8} | {'dy (Raw)':>8} | {'X (Acc)':>9} | {'Y (Acc)':>9} | {'Heading':>10} | {'Error (X, Y, Head)':>22}")
    print("="*125)

    global_step_counter = 0

    for frame_idx in range(start_frame, end_frame, CHUNK_SIZE):
        
        current_item = dataset[frame_idx]
        task_id = int(current_item["task_index"].item()) if "task_index" in current_item else 0
        print(f"👉 Task_index: {current_item['task_index'].item()}")
        
        #  目標航點 task 都相同
        # if task_id == 0:
        #     print(f"\n👉 Task {task_id} Detected: 'Turn Right' Instruction")
        # else:
        #     print(f"\n👉 Task {task_id} Detected: 'Turn Left' Instruction")

        # 🛠️ [關鍵修改] 動態打包 raw_batch，不再寫死 Token
        raw_batch = {}
        for key, value in current_item.items():
            if isinstance(value, torch.Tensor):
                raw_batch[key] = value.unsqueeze(0)
            else:
                raw_batch[key] = [value]

        # 執行前處理
        if preprocessor is not None:
            batch_for_policy = preprocessor(raw_batch)
        else:
            batch_for_policy = raw_batch

        # 型別轉換與放上 GPU
        for k, v in batch_for_policy.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    batch_for_policy[k] = v.to(DEVICE, dtype=DTYPE)
                else:
                    batch_for_policy[k] = v.to(DEVICE)

        # --- 收集這 16 步內的真實打桿資料 (Ground Truth) ---
        gt_actions_list = []
        for i in range(CHUNK_SIZE):
            if frame_idx + i < end_frame:
                next_item = dataset[frame_idx + i]
                gt_actions_list.append(next_item["action"].numpy())
            else:
                break
        gt_action_array = np.array(gt_actions_list)
        valid_steps = len(gt_action_array)

        # 🛠️ [關鍵修改] 加上 GPU 同步計時器
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # --- 讓模型大腦預測這 16 步 ---
        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    action_pred = model.predict_action_chunk(batch_for_policy)
            else:
                action_pred = model.predict_action_chunk(batch_for_policy)


        # 🛠️ [關鍵修改] 結束計時
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 計算單次耗時並記錄
        infer_duration = end_time - start_time
        inference_times_log.append(infer_duration)


        # 🛠️ 執行後處理 (反正規化)
        if postprocessor is not None:
            action_pred_dict = postprocessor({"action": action_pred})
            action_pred = action_pred_dict["action"]

        pred_action_array = action_pred.detach().float().cpu().numpy()
        pred_action_array = np.squeeze(pred_action_array, axis=0)       
        if pred_action_array.ndim == 1: 
            pred_action_array = np.expand_dims(pred_action_array, axis=0)   

        # 🛠️ [修改印出格式] 加入時間與 FPS 顯示
        print(f"--- [Inference Node Triggered @ Frame {frame_idx:05d}] ⏱️ 耗時: {infer_duration:.4f}s ({1/infer_duration:.1f} FPS) ---")

        for i in range(valid_steps):
            global_step_counter += 1

            pred_action = pred_action_array[i, :3]
            gt_action = gt_action_array[i, :3]

            current_pred_state[0] += pred_action[0] 
            current_pred_state[1] += pred_action[1] 
            current_pred_state[2] = pred_action[2]  

            current_gt_state[0] += gt_action[0]
            current_gt_state[1] += gt_action[1]
            current_gt_state[2] = gt_action[2]

            err_x = abs(current_pred_state[0] - current_gt_state[0])
            err_y = abs(current_pred_state[1] - current_gt_state[1])
            err_h = abs(current_pred_state[2] - current_gt_state[2])

            global_pred_traj.append(current_pred_state.copy())
            global_gt_traj.append(current_gt_state.copy())

            print(f" {global_step_counter:<11} | {'Pred':<4} | "
                  f"{pred_action[0]:>8.4f} | {pred_action[1]:>8.4f} | "
                  f"{current_pred_state[0]:>9.4f} | {current_pred_state[1]:>9.4f} | {current_pred_state[2]:>10.4f} | "
                  f"({err_x:>6.4f}, {err_y:>6.4f}, {err_h:>6.4f})")
            
            print(f" {'':<11} | {'GT':<4} | "
                  f"{gt_action[0]:>8.4f} | {gt_action[1]:>8.4f} | "
                  f"{current_gt_state[0]:>9.4f} | {current_gt_state[1]:>9.4f} | {current_gt_state[2]:>10.4f} | "
                  f"{'':>22}")

        inference_nodes_pred.append(current_pred_state.copy())
        inference_nodes_gt.append(current_gt_state.copy())

    print("="*125)

    # 🛠️ [新增] 在結尾印出平均效能報告
    avg_infer_time = np.mean(inference_times_log)
    print(f"\n🚀 效能報告 (Performance Summary):")
    print(f"   - 單次推論平均耗時: {avg_infer_time:.4f} 秒")
    print(f"   - 模型等效推論幀率: {1 / avg_infer_time:.2f} Hz (FPS)")

    print("\n📊 正在生成整趟任務軌跡圖表...")
    plot_full_episode_trajectory(global_pred_traj, global_gt_traj, inference_nodes_pred, inference_nodes_gt, TARGET_EPISODE)

if __name__ == "__main__":
    main()
