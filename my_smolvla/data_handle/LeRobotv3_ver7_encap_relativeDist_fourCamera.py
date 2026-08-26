# Data Encapsulation Program for LeRobot Dataset (v3) for Relative Distance and Heading Error with Four Cameras
import os
import cv2
import torch
import numpy as np
import math
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 基本設定
raw_fps = 30
dataset_fps = 5
frame_step = int(raw_fps / dataset_fps)  # 算出間隔 = 6
#frame_step = 1  # 先直接以 5fps 來算

video_width = 640
video_height = 480
video_channel = 3

repo_id = "Drone_4Cam_Relative_v1"
root = Path("LeRobot_Dataset_4Cam")     # 資料打包完，存的地方
source_data_path = Path("data_openfile")  # raw data 的來源資料夾

# 定義這四顆鏡頭的後綴對應關係 (需與你在收集程式裡的 CAM_INDEX_MAP 吻合)
CAM_SUFFIX = {
    "front": "_0.jpg",
    "left": "_1.jpg",
    "right": "_2.jpg",
    "bottom": "_3.jpg"
}

# 2. 宣告 LeRobot 資料集
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    root=root,
    robot_type="custom_drone",
    fps=dataset_fps, # 這裡告訴 LeRobot 最終生出的資料集是 5fps
    features={
        "observation.state": {"dtype": "float32", "shape": (3,), "names": ["dist_fwd", "dist_right", "heading_err"]}, 
        "observation.images.front": {"dtype": "video", "shape": (video_channel, video_height, video_width), "names": ["channel", "height", "width"]},
        "observation.images.left": {"dtype": "video", "shape": (video_channel, video_height, video_width), "names": ["channel", "height", "width"]},
        "observation.images.right": {"dtype": "video", "shape": (video_channel, video_height, video_width), "names": ["channel", "height", "width"]},
        "observation.images.bottom": {"dtype": "video", "shape": (video_channel, video_height, video_width), "names": ["channel", "height", "width"]},
        "action": {"dtype": "float32", "shape": (3,), "names": ["action_fwd", "action_right", "action_heading"]}, 
        "next.done": {"dtype": "bool", "shape": (1,), "names": "mission_done"},
    },
    use_videos=True,
)

#task_description = "Fly to the target waypoint, going around any obstacles on the left or right."
task_description = "Fly to the target waypoint, going around any obstacles, and follow the path with the red backpack if visible."

# 3. 讀取所有回合 (round00000, round00001...)
round_folders = sorted([x for x in source_data_path.iterdir() if x.is_dir() and x.name.startswith("round")])

for round_item in round_folders:
    txt_path = round_item / 'data.txt'
    if not txt_path.exists():
        print(f"⚠️ 找不到 {txt_path}，跳過此回合。")
        continue

    with open(txt_path, 'r', encoding='utf8') as f:
        text_data = f.readlines()
        
    num_frames = len(text_data)
    if num_frames == 0:
        continue
        
    print(f"🚀 處理 {round_item.name}... (總幀數: {num_frames})")


    # 假設最後一幀的座標是這趟航程的「終點/Target」
    last_data = text_data[-1].split(',')
    target_x = float(last_data[22]) # 假設 data[22] 是 NED_x
    target_y = float(last_data[23]) # 假設 data[23] 是 NED_y


    # ADDBYME
    print(f"最後一幀的資料: {last_data}")

    #target_waypoint = [float(last_line[16]), float(last_line[17])]      # 須確認要從哪裡抓 target_waypoint

# 4. 逐幀轉換
    for frame_idx in range(num_frames):
        if frame_idx % frame_step != 0:
            continue

        # --- A. 讀取四顆鏡頭的影像 ---
        img_dict = {}
        missing_image = False
        for cam_name, suffix in CAM_SUFFIX.items():
            img_path = str(round_item / f"frame{frame_idx:08d}{suffix}")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  ❌ 找不到圖片: {img_path}，提早結束本回合。")
                missing_image = True
                break
            
            # 將 BGR 轉 RGB 並轉成 PyTorch 要求的 (C, H, W) 格式
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = img_rgb.transpose(2, 0, 1)
            img_dict[cam_name] = img_tensor
            
        if missing_image:
            break # 少任何一張圖，直接斷開這回合

        # --- B. 讀取與轉換 State (絕對轉相對) ---
        data = text_data[frame_idx].split(',')
        ned_x = float(data[22])
        ned_y = float(data[23])
        ned_z = float(data[24])
        heading = float(data[12])

        # 計算 dx, dy (與目標的絕對位移差)
        dx_ned = target_x - ned_x
        dy_ned = target_y - ned_y
        print(f" Frame {frame_idx}: dx_ned={dx_ned:.2f} = target_x({target_x:.2f}) - ned_x({ned_x:.2f}), dy_ned={dy_ned:.2f} = target_y({target_y:.2f}) - ned_y({ned_y:.2f})")
        
        # 🟢 將絕對位移投影到機體的「前方」與「右方」
        rad = math.radians(heading)
        dist_fwd = dx_ned * math.cos(rad) + dy_ned * math.sin(rad)
        dist_right = -dx_ned * math.sin(rad) + dy_ned * math.cos(rad)
        
        # 🟢 計算目標在相對於車頭的「夾角誤差」 (-180 到 180)
        target_angle_deg = math.degrees(math.atan2(dy_ned, dx_ned))
        heading_err = target_angle_deg - heading
        heading_err = (heading_err + 180) % 360 - 180
        
        #state_data = [dist_fwd, dist_right, heading_err, ned_z]
        state_data = [dist_fwd, dist_right, heading_err]

        # --- C. 計算 Action (預測下一步) ---
        next_idx = frame_idx + frame_step
        if next_idx < num_frames:
            next_data = text_data[next_idx].split(',')
            next_ned_x = float(next_data[22])
            next_ned_y = float(next_data[23])
            next_heading = float(next_data[12])
            
            # 這一步的真實 NED 位移量
            dx_ned = next_ned_x - ned_x
            dy_ned = next_ned_y - ned_y
            
            # 🟢 將下一步的 NED 位移，旋轉成相對於「當下車頭」的機體座標位移
            action_fwd = dx_ned * math.cos(rad) + dy_ned * math.sin(rad)
            action_right = -dx_ned * math.sin(rad) + dy_ned * math.cos(rad)
            
            # 旋轉量
            action_heading = next_heading - heading
            action_heading = (action_heading + 180) % 360 - 180
            
            frame_done = False
        else:
            action_fwd, action_right, action_heading = 0.0, 0.0, 0.0
            frame_done = True

        # --- D. 寫入 Dataset ---
        dataset.add_frame({
            "observation.state": torch.tensor(state_data, dtype=torch.float32),
            "observation.images.front": img_dict["front"],
            "observation.images.left": img_dict["left"],
            "observation.images.right": img_dict["right"],
            "observation.images.bottom": img_dict["bottom"],
            "action": torch.tensor([action_fwd, action_right, action_heading], dtype=torch.float32),
            "task": task_description,
            "next.done": torch.tensor([frame_done], dtype=torch.bool),
        })

    dataset.save_episode()
    print(f"  ✅ {round_item.name} 儲存完畢。")

dataset.finalize()
print(f"🎉 全部轉換完成！目前資料集總 Episode 數: {dataset.num_episodes}")