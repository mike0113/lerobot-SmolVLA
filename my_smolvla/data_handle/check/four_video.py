import cv2
import glob
import os
import numpy as np
import argparse


def create_separate_videos(round_folder, output_prefix="output_video", fps=30):
    # 取得 video0 的所有影像路徑並排序，以此作為基準時間軸
    search_path = os.path.join(round_folder, "*_0.jpg")
    frame0_paths = sorted(glob.glob(search_path))
    
    if not frame0_paths:
        print(f"在 {round_folder} 找不到影像資料。")
        return

    # 讀取第一張圖來取得原始解析度
    sample_img = cv2.imread(frame0_paths[0])
    if sample_img is None:
        print("無法讀取第一張影像。")
        return
    
    h, w, _ = sample_img.shape
    
    # 設定影片輸出編碼
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 建立 4 個獨立的 VideoWriter
    writers = []
    for i in range(4):
        out_filename = f"{output_prefix}_cam{i}.mp4"
        writers.append(cv2.VideoWriter(out_filename, fourcc, fps, (w, h)))
        
    print(f"開始轉換影片，每個鏡頭總幀數預估：{len(frame0_paths)}")
    
    # 產生全黑影像，用於遺失幀的填補，確保時間軸不亂掉
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    for f0_path in frame0_paths:
        # 推導四個鏡頭的檔案路徑
        paths = [
            f0_path,
            f0_path.replace('_0.jpg', '_1.jpg'),
            f0_path.replace('_0.jpg', '_2.jpg'),
            f0_path.replace('_0.jpg', '_3.jpg')
        ]
        
        # 依序讀取並寫入四支影片
        for i in range(4):
            img = cv2.imread(paths[i])
            if img is None:
                img = black_frame
            writers[i].write(img)

    # 迴圈結束，釋放所有資源
    for writer in writers:
        writer.release()
        
    print(f"四路獨立影片轉換完成！(輸出檔名為 {output_prefix}_cam0.mp4 ~ cam3.mp4)")



# ==========================================
# 處理命令列參數
# ==========================================
parser = argparse.ArgumentParser(description="影像檢查程式 (支援多鏡頭同步影像拼接成影片)")

# 新增 -d 參數
parser.add_argument(
    '-d', '--dir', 
    type=str, 
    default="../data_openfile/round00000", 
    help="設定要處理的資料夾路徑 ../data_openfile/round00000 (預設值)"
)

# 新增的 -f 參數
parser.add_argument(
    '-f', '--fps', 
    type=int,              # 型態用 int
    default=30,             # 預設值，可依需求改為 None 或其他數字
    help="設定要處理的幀數或框架 (預設值: 30)"
)


args = parser.parse_args()

dir_path = args.dir
fps = args.fps


# 執行轉換 (請替換為你的實際 round 資料夾名稱與期望的輸出前綴)
create_separate_videos(dir_path, f"output-{os.path.basename(dir_path)}", fps=fps)
print(f"{os.path.basename(dir_path)} 獨立影片轉換完成。")
