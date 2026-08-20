#!/usr/bin/env python3
import os
import json
import base64
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import numpy as np
import cv2

# 強制離線設定
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_DIR = "/home/nvidia/smolvla/lerobot/program/min/model/0714_5hz_addWaypoints_merge_output-1/checkpoints/040000/pretrained_model" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

class VLAServerNode(Node):
    def __init__(self):
        super().__init__('vla_server_node')
        
        # --- 模型載入[cite: 6] ---
        self.get_logger().info("⏳ 伺服器啟動中：正在載入 SmolVLA 模型 (這需要一點時間)...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        
        self.model = SmolVLAPolicy.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
        # 🌟 加上這行，強制鎖死 Dropout 與 BatchNorm，保證推論的確定性 (Determinism)
        self.model.eval()
        
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_preprocessor.json")
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_DIR, config_filename="policy_postprocessor.json")
        
        # --- ROS2 通訊 ---
        # 接收 Client 的推論請求
        self.req_sub = self.create_subscription(String, '/vla/inference_req', self.inference_callback, 10)
        # 回傳推論結果給 Client
        self.res_pub = self.create_publisher(String, '/vla/inference_res', 10)
        
        self.get_logger().info("✅ 伺服器載入完畢！等待 Client 傳送推論請求...")

    def inference_callback(self, msg):
        try:
            # 1. 解析 Client 傳來的 JSON 請求
            req_data = json.loads(msg.data)
            req_id = req_data["req_id"]
            task = req_data["task"]
            state = torch.tensor(req_data["state"], dtype=torch.float32)
            
            # 將 Base64 影像轉回原始 byte 陣列，並解碼[cite: 6]
            img_bytes = base64.b64decode(req_data["image_b64"])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
            
            self.get_logger().info(f"📥 收到請求 #{req_id}，開始推論...")

            # 2. 準備模型輸入[cite: 6]
            batch = {
                "observation.images.main": img_tensor.unsqueeze(0).to(DEVICE, dtype=DTYPE),
                "observation.state": state.unsqueeze(0).to(DEVICE, dtype=DTYPE),
                "task": [task]
            }
            batch = self.preprocessor(batch)
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v):
                        batch[k] = v.to(DEVICE, dtype=DTYPE)
                    else:
                        batch[k] = v.to(DEVICE)

            # 3. 執行推論[cite: 6]
            with torch.inference_mode():
                if DEVICE == "cuda":
                    with torch.autocast(device_type="cuda", dtype=DTYPE):
                        pred = self.model.predict_action_chunk(batch)
                else:
                    pred = self.model.predict_action_chunk(batch) 
            
            pred = self.postprocessor({"action": pred})["action"]
            actions = np.squeeze(pred.detach().cpu().numpy(), axis=0).tolist() 
            
            # 4. 將結果打包回傳給 Client
            res_dict = {
                "req_id": req_id,
                "actions": actions
            }
            res_msg = String()
            res_msg.data = json.dumps(res_dict)
            self.res_pub.publish(res_msg)
            
            self.get_logger().info(f"📤 請求 #{req_id} 推論完成並回傳！")

        except Exception as e:
            self.get_logger().error(f"推論過程發生錯誤: {e}")

def main():
    rclpy.init()
    node = VLAServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()