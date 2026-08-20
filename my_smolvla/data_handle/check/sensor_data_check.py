import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

def save_ned_trajectory(data_txt_path, output_img_path, step=1):
    if not os.path.exists(data_txt_path):
        print("找不到 data.txt")
        return

    # 由於檔案格式是逗號分隔，且總共有 36 個欄位 (25感測器 + 8坐標 + 3字串/狀態)
    col_names = [f"col_{i}" for i in range(36)]
    
    try:
        # 讀取 CSV 格式的 txt 檔
        df = pd.read_csv(data_txt_path, names=col_names)
        
        # [核心修改] 根據 step 參數進行下採樣 (Downsampling)
        # step=1 代表全取，step=6 代表每 6 個取 1 個 (模擬 5fps)
        df_sampled = df.iloc[::step].copy()
        
        # 提取 NED 資料
        ned_x = df_sampled['col_22']
        ned_y = df_sampled['col_23']
        ned_z = df_sampled['col_24']

        # 為了讓顏色能正確對應到原始的時間軸，我們直接使用原本的 index
        time_seq = df_sampled.index 
        
        # 建立畫布
        fig = plt.figure(figsize=(16, 7))
        
        # ==========================================
        # 1. 繪製 2D 軌跡 (North vs East)
        # ==========================================
        ax1 = fig.add_subplot(121)
        
        # 只畫點，不畫線
        sc1 = ax1.scatter(ned_y, ned_x, c=time_seq, cmap='jet', s=15, zorder=5)
        
        # 將「圖上的每一個點」都標示出原始行數 Index
        for i in range(0, len(df_sampled), 5 if step == 6 else 30):
            orig_idx = df_sampled.index[i]
            # 加上微小偏移量與縮小字體
            ax1.text(ned_y.iloc[i], ned_x.iloc[i], f' {orig_idx}', fontsize=6, color='black', zorder=10)
        
        ax1.set_xlabel('East (Y) [m]')
        ax1.set_ylabel('North (X) [m]')
        ax1.set_title(f'2D NED Trajectory (Sample Step={step})')
        ax1.grid(True)
        ax1.axis('equal') 

        # 加入顏色條
        cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Original Frame Index (Time)')
        
        # ==========================================
        # 2. 繪製 3D 軌跡
        # ==========================================
        ax2 = fig.add_subplot(122, projection='3d')
        
        # 只畫點，不畫線
        sc2 = ax2.scatter(ned_y, ned_x, ned_z, c=time_seq, cmap='jet', s=15, zorder=5)
        
        # 3D 同樣標上每一個點的原始 Index
        for i in range(0, len(df_sampled), 5 if step == 6 else 30):
            orig_idx = df_sampled.index[i]
            ax2.text(ned_y.iloc[i], ned_x.iloc[i], ned_z.iloc[i], f' {orig_idx}', fontsize=6, color='black', zorder=10)
        
        ax2.set_xlabel('East (Y) [m]')
        ax2.set_ylabel('North (X) [m]')
        ax2.set_zlabel('Altitude [m]')
        ax2.set_title(f'3D Flight Trajectory (Sample Step={step})')
        
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
        print(f"已生成軌跡圖: {output_img_path} (圖上共標示 {len(df_sampled)} 個點)")

        plt.close(fig)  

    except Exception as e:
        print(f"解析資料發生錯誤: {e}")

# ==========================================
# 處理命令列參數
# ==========================================
parser = argparse.ArgumentParser(description="事後批次處理：將 sensor_data.txt 中的 NED 軌跡繪製成 2D/3D 圖形")

parser.add_argument(
    '-d', '--dir', 
    type=str,              
    default="../data_openfile/round00003",             
    help="設定要處理的資料夾路徑 ../data_openfile/round00003 (預設值)"
)

args = parser.parse_args()

data_file = os.path.join(args.dir, "data.txt")
base_name = os.path.basename(args.dir)

# 執行第 1 次：所有點都畫，且所有點都標示 (step=1)
out_img_1 = os.path.join(base_name + "_ned_all_points.png")
save_ned_trajectory(data_file, output_img_path=out_img_1, step=1)

# 執行第 2 次：每 6 個點取 1 個點畫，並標示這些點 (step=6，模擬 5fps)
out_img_2 = os.path.join(base_name + "_ned_every_6th_point.png")
save_ned_trajectory(data_file, output_img_path=out_img_2, step=6)