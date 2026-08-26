import cv2
import glob
import os
import numpy as np
import argparse

def create_sync_video(round_folder, output_filename="sync_video.mp4", fps=30):
    # 取得 video0 的所有影像路徑並排序
    search_path = os.path.join(round_folder, "*_0.jpg")
    frame0_paths = sorted(glob.glob(search_path))
    
    if not frame0_paths:
        print(f"在 {round_folder} 找不到影像資料。")
        return

    # 讀取第一張圖來取得解析度
    sample_img = cv2.imread(frame0_paths[0])
    if sample_img is None:
        print("無法讀取影像。")
        return
    
    h, w, _ = sample_img.shape
    
    # 設定影片輸出 (2x2 網格，所以長寬都要乘以 2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w * 2, h * 2))
    
    print(f"開始轉換影片，總幀數預估：{len(frame0_paths)}")
    
    for f0_path in frame0_paths:
        # 推導其他三個鏡頭的檔案路徑
        f1_path = f0_path.replace('_0.jpg', '_1.jpg')
        f2_path = f0_path.replace('_0.jpg', '_2.jpg')
        f3_path = f0_path.replace('_0.jpg', '_3.jpg')
        
        # 讀取四張圖片
        img0 = cv2.imread(f0_path)
        img1 = cv2.imread(f1_path)
        img2 = cv2.imread(f2_path)
        img3 = cv2.imread(f3_path)
        
        # 若有掉幀或遺失，用全黑影像代替
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        img0 = img0 if img0 is not None else black_frame
        img1 = img1 if img1 is not None else black_frame
        img2 = img2 if img2 is not None else black_frame
        img3 = img3 if img3 is not None else black_frame
        
        # 拼接影像 (Top: 0, 3 | Bottom: 1, 2)
        top_row = np.hstack((img0, img3))
        bottom_row = np.hstack((img1, img2))
        grid = np.vstack((top_row, bottom_row))
        
        out.write(grid)

    out.release()
    print(f"影片轉換完成：{output_filename}")



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

# 執行轉換 (請替換為你的實際 round 資料夾名稱)
create_sync_video(dir_path, f"output_{fps}fps-{os.path.basename(dir_path)}.mp4", fps=fps)
print(f"{os.path.basename(dir_path)} 影片轉換完成。")