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
MODEL_DIR = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/model/0714_5hz_addWaypoints_merge_output/checkpoints/100000/pretrained_model"

#DATASET_ROOT = "/home/min/workspace/lerobot/my_smolvla/dataset/data_v3_5hz_baseball_GoBack_2Prompt_revisedNED"
DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_merge"


OUTPUT_DIR = Path("single_chunk_inference_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 🎯 指定你要無人機「睜開眼睛」的那一幀 (Frame Index)
TARGET_FRAME = 940
CHUNK_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ==========================================
# 2. 畫圖函數：單次 16 步的 2D 軌跡預測
# ==========================================
def plot_single_chunk_2d(pred_actions, gt_actions, frame_idx):
    """
    將 16 步的 dx, dy 累積成絕對座標，並畫出 2D 鳥瞰軌跡
    """
    # 累積 dx, dy 計算絕對座標 (從 0,0 開始)
    pred_x = np.cumsum(pred_actions[:, 0])
    pred_y = np.cumsum(pred_actions[:, 1])
    gt_x = np.cumsum(gt_actions[:, 0])
    gt_y = np.cumsum(gt_actions[:, 1])

    # 補上起點 (0, 0)
    pred_x = np.insert(pred_x, 0, 0.0)
    pred_y = np.insert(pred_y, 0, 0.0)
    gt_x = np.insert(gt_x, 0, 0.0)
    gt_y = np.insert(gt_y, 0, 0.0)

    plt.figure(figsize=(8, 8))
    plt.title(f"Single Inference @ Frame {frame_idx}\n(Predicting Future 16 Steps)", fontsize=14, fontweight='bold')

    # 畫出 Ground Truth 與預測路線
    plt.plot(gt_x, gt_y, 'g-o', label='Ground Truth (Future 16 steps)', linewidth=2, markersize=5, alpha=0.7)
    plt.plot(pred_x, pred_y, 'r--X', label='SmolVLA Predicted (16 steps)', linewidth=2, markersize=6, alpha=0.9)
    
    # 標示起點
    plt.scatter(0, 0, color='blue', s=200, zorder=5, marker='*', label='Current Drone Pos (0,0)')

    plt.xlabel("X (meters - relative to drone)", fontsize=12)
    plt.ylabel("Y (meters - relative to drone)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 確保 X Y 比例 1:1，這樣看轉彎才準
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"single_infer_frame_{frame_idx}_2d_traj.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 單次 16 步預測 2D 軌跡圖已儲存為: {save_path}")
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
    preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_postprocessor.json")

    print("\n⏳ 2. 正在載入 LeRobotDataset...")
    #dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT, video_backend="pyav")  # on Jetson, use "pyav" for better performance
    dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT)  # on PC, use default backend (ffmpeg) for better compatibility   

    
    if TARGET_FRAME >= len(dataset):
        raise ValueError(f"指定的 Frame {TARGET_FRAME} 超出資料集範圍 (總長度: {len(dataset)})")

    print(f"\n🚀 準備進行實機單次推論模擬！")
    print(f"   - 模擬當下時間點: Frame {TARGET_FRAME}")

    # ==========================================
    # 擷取單一 Frame 的資料 (模擬當下拍到的畫面)
    # ==========================================
    current_item = dataset[TARGET_FRAME]
    task_id = int(current_item["task_index"].item()) if "task_index" in current_item else 0
    print(f"👉 偵測到指令 Task ID: {task_id}")

    #ADD BY ME
    print("current_item:", current_item)

    # 模擬 Batch Size = 1 的實機輸入格式
    raw_batch = {}
    for key, value in current_item.items():
        if isinstance(value, torch.Tensor):
            raw_batch[key] = value.unsqueeze(0)
        else:
            raw_batch[key] = [value]


    # ==========================================
    # 🎯 [新增] 強制寫死你自己的文字指令 (Prompt)
    # ==========================================
    #my_custom_prompt = "Fly to the target waypoint, going around any obstacles on the right."  # 👈 在這裡換成你想下的指令
    #raw_batch["language_instruction"] = [my_custom_prompt]
    #print(f"\n💬 收到長官文字指令: '{my_custom_prompt}'")
    # ==========================================

    #ADD BY ME
    print("raw_batch:", raw_batch)

    # 前處理與送上 GPU
    batch_for_policy = preprocessor(raw_batch)
    for k, v in batch_for_policy.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                batch_for_policy[k] = v.to(DEVICE, dtype=DTYPE)
            else:
                batch_for_policy[k] = v.to(DEVICE)

    # 取得未來 16 步的 Ground Truth 用來比較
    gt_actions_list = []
    for i in range(CHUNK_SIZE):
        if TARGET_FRAME + i < len(dataset):
            gt_actions_list.append(dataset[TARGET_FRAME + i]["action"].numpy()[:3])
    gt_actions = np.array(gt_actions_list)

    # ==========================================
    # 模型進行「唯一一次」推論
    # ==========================================
    if DEVICE == "cuda": torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                action_pred = model.predict_action_chunk(batch_for_policy)
        else:
            action_pred = model.predict_action_chunk(batch_for_policy)

    if DEVICE == "cuda": torch.cuda.synchronize()
    infer_duration = time.perf_counter() - start_time

    # 後處理 (反正規化)
    action_pred_dict = postprocessor({"action": action_pred})
    pred_actions = action_pred_dict["action"].detach().float().cpu().numpy()
    pred_actions = np.squeeze(pred_actions, axis=0) # Shape: (16, Action_Dim)
    pred_actions = pred_actions[:len(gt_actions), :3] # 取前三軸 (dx, dy, heading)

    # ==========================================
    # 輸出結果
    # ==========================================
    print("\n" + "="*70)
    print(f" ⏱️ 單次推論耗時: {infer_duration:.4f}s ({1/infer_duration:.1f} FPS)")
    print("="*70)
    print(f" {'Step':<5} | {'Pred dx, dy (m)':<25} | {'GT dx, dy (m)'}")
    print("-" * 70)
    
    for i in range(len(pred_actions)):
        p_dx, p_dy = pred_actions[i, 0], pred_actions[i, 1]
        g_dx, g_dy = gt_actions[i, 0], gt_actions[i, 1]
        print(f" {i:<5} |  {p_dx:>8.4f}, {p_dy:>8.4f}          |  {g_dx:>8.4f}, {g_dy:>8.4f}")

    print("\n📊 正在生成未來 16 步的 2D 軌跡圖表...")
    plot_single_chunk_2d(pred_actions, gt_actions, TARGET_FRAME)

if __name__ == "__main__":
    main()
