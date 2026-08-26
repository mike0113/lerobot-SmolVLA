# revise to ctrl+c to stop the program safely
# 30 fps and 480*640 pixol
#                   0               1               2               3              4                5               6               7               8           9           10          11                                  12
# sensor_data: [xacc(x 加速度), yacc(y 加速度), zacc(z 加速度), xgyro(x 角速度), ygyro(y 角速度), zgyro(z 角速度), xmag(x 磁場強度), ymag(y 磁場強度), zmag(z 磁場強度), lat(緯度), lon(經度), relative_alt(相對高度)(相對起飛點), heading(絕對航向角)(0~360), 
#               roll(翻滾角), pitch(俯仰角), yaw(偏航角), ekf_flags, velocity_variance (速度誤差估計), pos_horiz_variance (水平位置誤差估計), pos_vert_variance (垂直位置誤差估計), compass_variance (羅盤誤差估計), terrain_alt_variance (地形高度誤差估計), ned_x (向北位移), ned_y (向東位移), ned_z (向下位移)]
#                   13          14              15          16          17                                  18                                  19                              20                              21                                  22              23              24
from collections import deque       #new
from pymavlink import mavutil
from v4l2py import Device
from pathlib import Path
import multiprocessing              #new
import numpy as np
import threading
import json
import time
import math
import cv2
import os
import argparse

# ==========================================
# 處理命令列參數
# ==========================================
parser = argparse.ArgumentParser(description="資料收集程式 (支援多鏡頭與 MAVLink 感測器)")

# 新增 --frames 參數，預設為 1830 (以 30 fps 計算，約等於一分鐘)
parser.add_argument(
    '-f', '--frames', 
    type=int, 
    default=1830, 
    help="設定要收集的總幀數。輸入 0 代表無限制一直收集，直到按下 Ctrl+C，例如 30 fps下一分鐘為 1800 + 30 (緩衝) (預設值)"
)

args = parser.parse_args()

# 1. 數據 (假設長度為 300 幀，即 10 秒)
num_frames = args.frames

# 如果輸入 0 或負數，就把幀數設為「無限大」
if num_frames <= 0:
    num_frames = math.inf


sensor_count = 25
video_width = 640
video_height = 480
video_channel = 3
sensor_data = list(range(0, sensor_count))  # 25個感測器
old_sensor_data = list(range(0, sensor_count))  # 前一個frame的資料，為了配合action的計算
send_sensor_data = list(range(0, sensor_count))  # 為了與輸出的action frame一致
action_data = list(range(0, 4))  # x, y, z, heading
real_timestamp = 0.0
send_real_timestamp = 0.0
MAX_TIMEGAP = 0.0
# 定義要讀取的相機裝置列表
CAMERA_DEVICES = ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"]
QUEUE_MAX_SIZE = 10
# 儲存每個相機對應的 Queue
# 結構為：{"/dev/video0": Queue, ...}
camera_deques = {
    "/dev/video0": deque(maxlen=10),
    "/dev/video1": deque(maxlen=10),
    "/dev/video2": deque(maxlen=10),
    "/dev/video3": deque(maxlen=10)
}
stop_event = threading.Event()
video_process_deques = {
    "/dev/video0": multiprocessing.Queue(maxsize=100),
    "/dev/video1": multiprocessing.Queue(maxsize=100),
    "/dev/video2": multiprocessing.Queue(maxsize=100),
    "/dev/video3": multiprocessing.Queue(maxsize=100)
}
def camera_capture_worker(device_path, data_deque):
    """
    背景執行緒工作函式：持續讀取相機影像與取像時間
    """
    print(f"[啟動] 執行緒開始讀取: {device_path}")
    try:
        with Device(device_path) as cam:
            for frame in cam:               # 這是一個無窮迭代器 (Infinite Iterator)，會一直抓新影像
                if stop_event.is_set():     # 如果是 set , 代表要停止，則跳出迴圈 
                    break
                data_packet = {
                    "timestamp": frame.timestamp, # 核心層高精度時間
                    "image": frame.data,
                }
                data_deque.append(data_packet)
    except Exception as e:
        print(f"[錯誤] 裝置 {device_path} 發生異常: {e}")
    finally:
        print(f"[停止] 執行緒已關閉: {device_path}")
def get_synchronized_frames(camera_deques):
    """
    以 video0 最新影格為基準，同步取出 video1~3 時間最接近的影像
    """
    # 取得基準點：清空 video0 的舊資料，只拿最後（最新）一張
    v0_deque = camera_deques["/dev/video0"]
    if not v0_deque:
        return None
        
    # 基準點：直接拿 video0 目前最後面（最新）的那一張，但不從 deque 移出
    base_packet = v0_deque[-1] 
    target_time = base_packet["timestamp"]
    
    synchronized_results = {
        "/dev/video0": base_packet
    }
    
    # 2. 搜尋其他相機
    other_cameras = ["/dev/video1", "/dev/video2", "/dev/video3"]
    
    for cam in other_cameras:
        d = None
        while d == None:
            d = camera_deques[cam]
            time.sleep(0.001)
            
        # 將目前的 deque 轉成快照（避免搜尋時被背景執行緒修改長度）
        snapshot = list(d)
        if len(d) == 0:
            print(d)
        # 尋找時間差絕對值最小的影格
        if len(snapshot) == 0:
            return None
        best_packet = min(snapshot, key=lambda x: abs(x["timestamp"] - target_time))
        synchronized_results[cam] = best_packet
        # print(len(best_packet['image']))
        
    return synchronized_results
def camera_process_worker(DEV_PATH):
    while True:
        raw_data = video_process_deques[DEV_PATH].get()
        if raw_data is None:
            print("子進程：收到結束訊號，準備安全退出。", flush=True)
            break
        raw_bytes, save_path = raw_data
        raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        uyvy_frame = raw_array.reshape((1080, 1920, 2))
        bgr_frame = cv2.cvtColor(uyvy_frame, cv2.COLOR_YUV2BGR_UYVY)
        bgr_frame_resized = cv2.resize(bgr_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path, bgr_frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # 可改為 95 看看效果，但可能會當機 (JPEG 圖片的壓縮品質 (Quality))
def set_msg_interval(msg_id, interval_us):
    """
    設定指定消息的發送間隔
    interval_us: 微秒 (例如 33333us = 30Hz)
    """
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,              # 確認傳輸
        msg_id,         # 參數 1: 消息 ID
        interval_us,    # 參數 2: 間隔時間 (us)
        0, 0, 0, 0, 0   # 參數 3-7 (未設定)
    )

def collect_data():
    global sensor_data, old_sensor_data, send_sensor_data, action_data, real_timestamp, send_real_timestamp, CAMERA_DEVICES, QUEUE_MAX_SIZE, camera_deques
    # 蒐集到其中一組資料就先暫停撈該資料，直到都蒐集並存取整組資料才收下一次的
    # flag_imu = False
    # flag_gps = False
    # flag_ned = False
    # flag_atd = False
    # # flag_ekf = False
    # flag_image = False
    while True:

        # ========== 關鍵新增 ==========
        # 利用 timeout 醒來的瞬間，檢查主程式是不是發出了 Ctrl+C 的停止訊號
        if stop_event.is_set():
            print("[停止] MAVLink 資料收集執行緒安全關閉。")
            break

        # 每次最多睡 0.1 秒，不佔 CPU，又不會永遠卡死
        # msg = master.recv_match(type=['RAW_IMU', 'GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED', 'ATTITUDE', 'EKF_STATUS_REPORT'], blocking=False)
        msg = master.recv_match(type=['HIGHRES_IMU', 'GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED', 'ATTITUDE'], blocking=True, timeout=0.1)

        # TEST
        # if msg != None:
        #     print(f"msg: {msg}")
        
        if msg == None:
            continue

        msg_type = msg.get_type()

        if msg_type == 'HIGHRES_IMU':
            sensor_data[0] = msg.xacc
            sensor_data[1] = msg.yacc
            sensor_data[2] = msg.zacc
            sensor_data[3] = msg.xgyro
            sensor_data[4] = msg.ygyro
            sensor_data[5] = msg.zgyro
            sensor_data[6] = msg.xmag
            sensor_data[7] = msg.ymag
            sensor_data[8] = msg.zmag
            # print(msg)
            flag_imu = True
        elif msg_type == 'GLOBAL_POSITION_INT':
            # print(msg.lat)
            sensor_data[9] = msg.lat / 1e7
            sensor_data[10] = msg.lon / 1e7
            sensor_data[11] = msg.relative_alt * 0.001
            sensor_data[12] = msg.hdg * 0.01
            #print(f"Location: {msg.lat/1e7}, {msg.lon/1e7}, {msg.relative_alt*0.01}, Heading: {msg.hdg*0.01}")
            flag_gps = True
        elif msg_type == 'ATTITUDE':
            yaw_rad = msg.yaw
            # 2. 轉換為角度 (-180 到 180)
            yaw_deg = math.degrees(yaw_rad)
            # 3. 轉換為標準導航航向 (0 到 360)
            heading = yaw_deg if yaw_deg >= 0 else yaw_deg + 360
            # sensor_data[12] = heading     # GPS有提供heading，這邊先註解掉
            sensor_data[13] = msg.roll
            sensor_data[14] = msg.pitch
            sensor_data[15] = msg.yaw
            flag_atd = True
        # elif msg_type == 'EKF_STATUS_REPORT' and not flag_ekf:
        #     sensor_data[16] = msg.flags
        #     sensor_data[17] = msg.velocity_variance
        #     sensor_data[18] = msg.pos_horiz_variance
        #     sensor_data[19] = msg.pos_vert_variance
        #     sensor_data[20] = msg.compass_variance
        #     sensor_data[21] = msg.terrain_alt_variance
        #     flag_ekf = True
        elif msg_type == 'LOCAL_POSITION_NED': # z 是向下的，所以取負號變成高度
            altitude = -msg.z
            # sensor_data[9] = msg.x
            # sensor_data[10] = msg.y
            # sensor_data[11] = altitude
            sensor_data[22] = msg.x
            sensor_data[23] = msg.y
            sensor_data[24] = altitude
            flag_ned = True
        # 暫時總計25個數值
        # if flag_imu and  flag_ned and flag_atd and flag_ekf and flag_gps:
        # if flag_imu and flag_atd and flag_ekf:
        # if flag_imu and flag_atd:
        # #if True:
        #     # 要算出action的移動量，所以x,y,z採用ned去回推出數值，但實際sensor data採用gps
        #     # action_data[0] = sensor_data[22] - old_sensor_data[22]
        #     # action_data[1] = sensor_data[23] - old_sensor_data[23]
        #     # action_data[2] = sensor_data[24] - old_sensor_data[24]
        #     # action_data[3] = sensor_data[12] - old_sensor_data[12]
        #     # for i in range(len(send_sensor_data)):
        #     #     send_sensor_data[i] = old_sensor_data[i]
        #     #     old_sensor_data[i] = sensor_data[i]
        #     send_real_timestamp = real_timestamp
        #     real_timestamp = time.perf_counter()
        #     flag_imu = False
        #     flag_ned = False
        #     flag_atd = False
        #     #flag_ekf = False
        #     flag_gps = False
        #     #print('check')
    time.sleep(0.1) 

#master = mavutil.mavlink_connection('/dev/ttyTHS3', baud=921600)
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=921600)

# 強制等待 PX4 的 Heartbeat 以取得 System ID 與 Component ID
#print("Waiting for PX4 heartbeat...")
#master.wait_heartbeat()
#print(f"Heartbeat received! Target System: {master.target_system}, Component: {master.target_component}")

time.sleep(1)
# 1. IMU 設定為 30Hz (約 33.3ms)
freq = 25000

# set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU, freq)
set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, freq)


# # TEST
# # 等待飛控回傳 ACK (確認指令是否成功)
# ack_msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
# print ("收到 ACK 訊息:", ack_msg)
# if ack_msg:
#     if ack_msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
#         if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED: # 數值為 0
#             print("指令成功！飛控同意發送資料。")
#         else:
#             print(f"指令被拒絕！錯誤碼: {ack_msg.result}") 
#             # 如果是 RAW_IMU，PX4 通常會回傳 3 (MAV_RESULT_UNSUPPORTED) 或 4 (FAILED)


# 2. Location
set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, freq)
set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, freq)

# 3. EKF 狀態設定
#set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, freq)

# 4. ATTITUDE 狀態設定
set_msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, freq)
t = threading.Thread(target=collect_data, daemon=True)
t.start()
# 2. 初始化 LeRobot 資料集 (v3.0 格式)
root = Path("data_openfile_480p")
root.mkdir(exist_ok=True)
threads = []
    
# 啟動 4 個背景執行緒蒐集相機影像
for dev_path in CAMERA_DEVICES:     # ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3"]
    q = camera_deques[dev_path]
    cam_t = threading.Thread(
        target=camera_capture_worker, 
        args=(dev_path, q), 
        daemon=True # 設定為守護執行緒，主程式結束時會自動關閉
    )
    threads.append(cam_t)
    cam_t.start()
time.sleep(3)
# fake image
# fake_frame = cv2.imread('./E60J6FDVEAEeSTQ.jpg')
# frame_resized = cv2.resize(fake_frame, (640, 480), interpolation=cv2.INTER_AREA)
start_timestamp = 0
frame_done = False
round_limit = 1
rounds = 0
CAM_INDEX_MAP = {
    "/dev/video0": 0,
    "/dev/video1": 1,
    "/dev/video2": 2,
    "/dev/video3": 3
}
cam_processes = []
for cam_id in CAM_INDEX_MAP:
    p = multiprocessing.Process(target=camera_process_worker, args=(cam_id,))
    p.daemon = True
    p.start()
    cam_processes.append(p)

try:
    while rounds < round_limit:
        frame_idx = 0
        path_idx = rounds
        new_folder = 'round' + f"{path_idx:05d}"
        while (root / new_folder).exists():
            path_idx += 1
            new_folder = 'round' + f"{path_idx:05d}"
        round_path = root / new_folder
        round_path.mkdir()
        f = open(str(round_path) + '/data.txt', 'w', encoding='utf8')
        frame_timestamp = time.time()
        while frame_idx < num_frames:
            if time.time() - frame_timestamp < 0.033:
                continue
            frame_timestamp = time.time()
            if frame_idx + 1 == num_frames:
                frame_done = True
            else:
                frame_done = False
            check_tact_time = time.time()
            # 4. 儲存與統計
            fens_location = [24.775844, 121.042378,24.775983, 121.042523,24.775877, 121.042684,24.775740, 121.042546]
            # total_data = sensor_data + fens_location + action_data + ["perform sensor recording"] + [frame_done] + [0]
            # --- 組合文字資料 ---
            total_data = sensor_data + fens_location + ["perform sensor recording"] + [frame_done] + [0]
            #print(total_data)
            # f.write(','.join([str(item) for item in total_data]) + '\n')
            # print('tact1:', (time.time() - check_tact_time) * 1000)
            check_tact_time = time.time()
            
            # 1. 先取得同步影像
            sync_data = get_synchronized_frames(camera_deques)
            print('tact1:', (time.time() - check_tact_time) * 1000)
            check_tact_time = time.time()
            # if sync_data is not None:
                # 檢查其他相機與 v0 的時間差
                # for cam in ["/dev/video1", "/dev/video2", "/dev/video3"]:
                #     if sync_data[cam]:
                #         diff_ms = (sync_data[cam]['timestamp'] - sync_data['/dev/video0']['timestamp']) * 1000
                #         # print(f" -> {cam} 時間差: {diff_ms:+.2f} ms")
                #         if MAX_TIMEGAP < diff_ms:
                #             MAX_TIMEGAP = diff_ms
                #     else:
                #         print(f" -> {cam} 無可用影像")

            # 2. 確定有拿到完整的 4 張圖片後，才同時進行文字與圖片的處理
            if sync_data is not None and len(sync_data) == 4:
                # --- 寫入文字 (讓 Python 自動管理 Buffer，維持最高效能) ---
                f.write(','.join([str(item) for item in total_data]) + '\n')

                # --- 寫入圖片 ---
                for dev_path, packet in sync_data.items():
                    sync_data_index = CAM_INDEX_MAP[dev_path]
                    # print(sync_data_index, type(packet["image"]))
                    raw_bytes = packet["image"]
                    save_path = str(round_path) + '/frame' + f"{frame_idx:08d}_{sync_data_index}.jpg"
                    video_process_deques[dev_path].put((raw_bytes, save_path))
                
                frame_idx += 1
                print('save frame success:', frame_idx)
            print('tact3:', (time.time() - check_tact_time) * 1000)
        f.close()
        # print('MAX_TIMEGAP:', MAX_TIMEGAP)
        # with open("MAX_TIMEGAP.txt", "a") as f:
        #    f.write(f"{MAX_TIMEGAP:.3f}\n")
        rounds += 1

except KeyboardInterrupt:
    print("\n[中斷] 收到 Ctrl+C，正在安全存檔並關閉程式...")
    stop_event.set()  # 設定停止事件，通知所有執行緒停止

finally:
    if 'f' in locals() and not f.closed:
        f.close()
        print("文字資料檔已安全關閉，無資料遺失！")

    for dev_path, q in video_process_deques.items():
        q.put(None) # 塞入 None 作為毒藥丸
    for p in cam_processes:
            p.join()
