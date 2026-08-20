#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import torch
import numpy as np
import cv2

# 強制離線模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

#DATASET_ROOT = "/home/nvidia/smolvla/lerobot/min/dataset/data_v3(80set)_5hz_baseball_GoBack_addwaypoints_merge"
DATASET_ROOT = "/home/min/workspace/lerobot-SmolVLA/my_smolvla/dataset/data_v3(80set)_5hz_baseball_GoBack_addWaypoints_merge"


class DatasetPublisherNode(Node):
    def __init__(self):
        super().__init__('dataset_publisher_node')
        
        self.image_pub = self.create_publisher(CompressedImage, '/front_image/compressed', 10)
        self.task_pub = self.create_publisher(String, '/task_name', 10)
        
        # 🚨 [通訊升級] sensor_data 改為 String，完全模擬真實飛控的輸出格式
        self.state_pub = self.create_publisher(String, '/sensor_data', 10)

        #self.target_pub = self.create_publisher(String, '/target', 10)
        
        # 🚨 [通訊升級] action_sub 改為 String，接收大腦傳來的 "dx,dy,dz,yaw" 字串
        self.action_sub = self.create_subscription(String, '/action', self.action_callback, 10)
        
        self.get_logger().info("⏳ 正在載入 LeRobotDataset 作為模擬訊號源...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.dataset = LeRobotDataset("local/baseball", root=DATASET_ROOT, video_backend="pyav")
        
        self.frame_idx = 0
        self.total_frames = len(self.dataset)
        self.finished = False
        self.last_hover_log_time = 0.0
        
        # [MODIFIED] 移除原本只發送一次的 publish_current_frame()[cite: 2]
        # 新增狀態標記與喚醒計時器
        self.brain_connected = False
        self.wakeup_timer = self.create_timer(1.0, self.wakeup_timer_callback)

        self.get_logger().info("智慧模擬器已啟動！(已切換為實機 String 通訊格式)")

    # [MODIFIED] 新增喚醒計時器的 callback，每秒發送一次直到大腦回應[cite: 2]
    def wakeup_timer_callback(self):
        if not self.brain_connected:
            self.get_logger().info("🔔 大腦尚未回應，持續發送初始畫面喚醒中...")
            self.publish_current_frame()
        else:
            # 已經連上，把這個定時器關掉，避免浪費資源
            self.wakeup_timer.cancel()


    def action_callback(self, msg):
        # [MODIFIED] 只要收到第一筆 Action，就標記大腦已連線[cite: 2]
        if not self.brain_connected:
            self.brain_connected = True
            self.get_logger().info("成功接收到大腦的 Action！喚醒模式結束，開始同步播放資料集。")

        
        if self.finished:
            return
            
        try:
            # 🚨 解析大腦傳來的字串: "x, y, z, heading"
            action_list = [float(x) for x in msg.data.split(',')]
            
            # 判斷大腦是否處於「懸停/睡覺」狀態 (dx, dy 都趨近於 0)
            # 🐛 [Bug 修復]: action_list[1] 確保不會報錯
            is_hovering = (abs(action_list[0]) < 1e-4) and (abs(action_list[1]) < 1e-4)
            
            if is_hovering:
                # 🛡️ 懸停狀態優化：安靜模式
                # 終端機每秒只印一次，避免洗頻，但「絕對不能 return」，必須繼續往下發送畫面！
                current_time = time.time()
                if current_time - self.last_hover_log_time > 1.0:
                    self.get_logger().info(f"⏸️ 無人機正在懸停 (Frame {self.frame_idx} 凍結中)...")
                    self.last_hover_log_time = current_time
            else:
                # 只有無人機真的在移動，影片才會前進一格！
                self.frame_idx += 1
                
            if self.frame_idx >= self.total_frames:
                self.get_logger().info("🏁 資料集播放完畢！")
                self.finished = True
                return
                
            # 🚨 不論是前進還是暫停，都【必須】持續發送當下應有的畫面與數值給大腦
            # 這樣大腦在睡醒時，才會有最新的實況可以觸發下一輪推論
            self.publish_current_frame()
            
        except Exception as e:
            self.get_logger().warn(f"解析 Action 字串失敗: {e}")

    def publish_current_frame(self):
        item = self.dataset[self.frame_idx]
        
        # 1. 發布壓縮影像
        img_tensor = item["observation.images.main"]
        img_np = (img_tensor.numpy() * 255.0).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        success, encoded_image = cv2.imencode('.jpg', img_bgr)
        if success:
            img_msg = CompressedImage()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.format = "jpeg"
            img_msg.data = encoded_image.tobytes()
            self.image_pub.publish(img_msg)
        
        # 2. 把真實的 frame_idx 藏在字串裡傳給大腦，確保畫圖 100% 精準！
        # 🟢 [任務升級] 自動判斷去程與回程，每 300 Frame 切換一次指令
        #cycle_state = (self.frame_idx // 300) % 2
        #if cycle_state == 0:
        #    task_instruction = "Fly to the target waypoint, going around any obstacles on the right."
        #else:
        #    task_instruction = "Fly to the target waypoint, going around any obstacles on the left."
        task_instruction = "Fly to the target waypoint, going around any obstacles on the left and right."
        
        task_msg = String()
        #task_msg.data = f"FRAME:{self.frame_idx}|{task_instruction}"
        task_msg.data = f"{task_instruction}"
        self.task_pub.publish(task_msg)

        # 3. 發布 State (偽裝成真實飛控的 10 個數值逗號字串)
        if "observation.state" in item :
            state_list = item["observation.state"].tolist()
            #target_list = item["observation.target"].tolist()
            
            # 為了配合真實飛控的格式: [GPS_lat, GPS_lon, GPS_alt, NED_x, NED_y, NED_z, Heading, tx, ty, tz]
            # 我們將 Dataset 的數值塞入對應的 NED_x, NED_y, NED_z 與 Yaw 的位置
            ned_x = state_list[0] if len(state_list) > 0 else 0.0
            ned_y = state_list[1] if len(state_list) > 1 else 0.0
            ned_z = state_list[2] if len(state_list) > 2 else 0.0
            # 通常 LeRobot state 第 13 個數值 (index 12) 是 yaw
            heading = state_list[3] if len(state_list) > 3 else 0.0

            #target_x = target_list[0] if len(target_list) > 0 else 0.0
            #target_y = target_list[1] if len(target_list) > 1 else 0.0
            target_x = state_list[4] if len(state_list) > 4 else 0.0
            target_y = state_list[5] if len(state_list) > 5 else 0.0
            
            # 組裝成 10 個元素的逗號字串 (前後補 0.0 模擬 GPS 和 target waypoint)
            state_str = f"0.0,0.0,0.0,{ned_x},{ned_y},{ned_z},{heading},{target_x},{target_y},0.0"
            #state_str = f"0.0,0.0,0.0,{ned_x},{ned_y},{ned_z},{heading},30.0,-23.0,0.0"
            

            state_msg = String()
            state_msg.data = state_str
            self.state_pub.publish(state_msg)

        # 4. 發布 Target
        #if "observation.target" in item:
        #    target_list = item["observation.target"].tolist()

        #    target_x = target_list[0] if len(target_list) > 0 else 0.0
        #    target_y = target_list[1] if len(target_list) > 1 else 0.0

        #    #target_str = f"{target_x},{target_y}"
        #    target_str = f"100000, 100000"

        #    target_msg = String()
        #    target_msg.data = target_str
        #    self.target_pub.publish(target_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DatasetPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
