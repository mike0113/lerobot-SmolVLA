#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 0. 環境設定
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ==========================================
# 1. 參數設定區
# ==========================================
#MODEL_DIR = "/home/min/workspace/lerobot/my_smolvla/model/0608_5hz_output-1/checkpoints/040000/pretrained_model" 
#MODEL_DIR = "/home/min/workspace/lerobot/my_smolvla/model/0713_5hz_addWaypoints_ADDBYME_NEW_output-1/checkpoints/040000/pretrained_model"
MODEL_DIR = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/model/0714_5hz_addWaypoints_merge_output/checkpoints/040000/pretrained_model"
#MODEL_DIR = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/model/0724_5hz_W_test_NEWoutput/checkpoints/020000/pretrained_model"

#DATASET_ROOT = "/home/min/workspace/lerobot/my_smolvla/dataset/data_v3_5hz_baseball_GoBack_2Prompt_revisedNED"
#DATASET_ROOT = "/home/min/workspace/lerobot/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_ADDBYME_NEW"
DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_merge"
#DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/260717IsaacSim_Data_v3"

OUTPUT_DIR = Path("inference_1d_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EPISODE = 10
CHUNK_SIZE = 16  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ==========================================
# 2. 畫圖函數
# ==========================================
def plot_1d_actions(gt_actions, pred_actions, ep_idx):
    """畫出 dx, dy, heading 的 1D 比較折線圖"""
    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)
    
    rmse = np.sqrt(np.mean((gt_actions - pred_actions) ** 2))
    mae = np.mean(np.abs(gt_actions - pred_actions))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Episode {ep_idx} 1D Action (16-Step Chunks) | RMSE={rmse:.6f}, MAE={mae:.6f}", fontsize=14, fontweight='bold')
    
    action_names = ['dx (action 0)', 'dy (action 1)', 'heading (action 2)']
    
    for i in range(3):
        axes[i].plot(gt_actions[:, i], label=f'GT action {i}', color='tab:blue', linewidth=1.5)
        axes[i].plot(pred_actions[:, i], label=f'Pred action {i}', color='tab:orange', linewidth=1.5, linestyle='--')
        axes[i].set_ylabel(action_names[i])
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
        for chunk_boundary in range(0, len(gt_actions), CHUNK_SIZE):
            axes[i].axvline(x=chunk_boundary, color='gray', linestyle=':', alpha=0.4)
        
    axes[-1].set_xlabel("Frame (Step)")
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / f"episode_{ep_idx}_1d_actions_16chunks.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 1D 動作比較圖已儲存為: {save_path}")
    plt.close()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    print(f"⏳ 1. 正在載入模型與 Pipeline... (DEVICE={DEVICE})")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # 載入模型
    model = SmolVLAPolicy.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
    
    # 🛠️ [關鍵修改 1] 載入前處理器 (Preprocessor)
    try:
        preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_preprocessor.json")
    except Exception as e:
        print(f"⚠️ 無法載入 Preprocessor！錯誤訊息: {e}")
        preprocessor = None

    # 載入後處理器 (Postprocessor)
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
    if end_frame is None: end_frame = len(dataset)

    print(f"\n🚀 準備開始 [16步整包預測比較] (Stride=16)！")
    print(f"   - 目標 Episode: {TARGET_EPISODE}")
    print(f"   - Frame 範圍: {start_frame} 到 {end_frame - 1}")

    gt_actions_log = []
    pred_actions_log = []
    inference_times_log = [] # 🛠️ [新增] 記錄每次推論耗時的陣列

    for frame_idx in range(start_frame, end_frame, CHUNK_SIZE):
        
        current_item = dataset[frame_idx]
        task_id = int(current_item["task_index"].item()) if "task_index" in current_item else 0
        
        # 🛠️ [關鍵修改 2] 完整保留 Dataset 的所有 Key，並模擬 Batch Size = 1 的格式
        raw_batch = {}
        for key, value in current_item.items():
            if isinstance(value, torch.Tensor):
                # 如果是影像或狀態等 Tensor，加上 batch 維度
                raw_batch[key] = value.unsqueeze(0)
            else:
                # 如果是字串 (例如 "task") 或數字，用 List 包起來
                raw_batch[key] = [value]

        # 執行前處理 (影像正規化、Token 轉換全部在這裡自動完成)
        if preprocessor is not None:
            batch_for_policy = preprocessor(raw_batch)
        else:
            batch_for_policy = raw_batch

        # 將處理好的 Tensor 搬到 GPU，並處理浮點數型別防呆
        for k, v in batch_for_policy.items():
            if isinstance(v, torch.Tensor):
                # 只有影像和狀態這種浮點數才轉 float16，整數(Token)維持不變
                if torch.is_floating_point(v):
                    batch_for_policy[k] = v.to(DEVICE, dtype=DTYPE)
                else:
                    batch_for_policy[k] = v.to(DEVICE)

        # 🛠️ [新增] 加上 GPU 同步計時器開始
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # --- 模型預測 ---
        with torch.inference_mode():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    action_pred = model.predict_action_chunk(batch_for_policy)
            else:
                action_pred = model.predict_action_chunk(batch_for_policy)

        # 🛠️ [新增] 加上 GPU 同步計時器結束
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 計算單次耗時並紀錄
        infer_duration = end_time - start_time
        inference_times_log.append(infer_duration)


        # 🛠️ [關鍵修改 3] 正確執行後處理 (反正規化)
        if postprocessor is not None:
            action_pred_dict = postprocessor({"action": action_pred})
            action_pred = action_pred_dict["action"]

        pred_array = action_pred.detach().float().cpu().numpy()
        pred_array = np.squeeze(pred_array, axis=0) 
        
        valid_steps = min(CHUNK_SIZE, end_frame - frame_idx)
        chunk_pred_action = pred_array[:valid_steps, :3]
        pred_actions_log.extend(chunk_pred_action)

        for i in range(valid_steps):
            future_item = dataset[frame_idx + i]
            gt_actions_log.append(future_item["action"].numpy()[:3])

        # 🛠️ [修改] 印出預測進度時加上耗時資訊
        print(f"已預測 Frame {frame_idx - start_frame:03d} ~ {frame_idx - start_frame + valid_steps - 1:03d} | ⏱️ 耗時: {infer_duration:.4f}s ({1/infer_duration:.1f} FPS)")

    print("="*60)
    
    # 🛠️ [新增] 結尾印出平均效能報告
    avg_infer_time = np.mean(inference_times_log)
    print(f"\n🚀 效能報告 (Performance Summary):")
    print(f"   - 單次推論平均耗時: {avg_infer_time:.4f} 秒")
    print(f"   - 模型等效推論幀率: {1 / avg_infer_time:.2f} Hz (FPS)")
    
    print("\n📊 正在生成舊模型 16步整包預測 比較圖表...")
    plot_1d_actions(gt_actions_log, pred_actions_log, TARGET_EPISODE)

if __name__ == "__main__":
    main()
