# usage: python add_gps_exif.py -d 你的資料夾名稱

import os
import piexif
from fractions import Fraction
import glob
import argparse

# ==========================================
# 輔助函式：轉換經緯度為 EXIF 格式
# ==========================================
def to_deg(value, loc):
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value - deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return (deg, min, sec, loc_value)

def change_to_rational(number):
    f = Fraction(str(number))
    return (f.numerator, f.denominator)

# ==========================================
# 核心處理函式
# ==========================================
def process_folder(folder_path):
    data_file = os.path.join(folder_path, 'data.txt')
    if not os.path.exists(data_file):
        print(f"  [略過] 找不到 {data_file}")
        return

    # 讀取 data.txt 的所有紀錄
    with open(data_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

    print(f"\n➤ 開始處理資料夾: {folder_path} (共 {len(lines)} 筆紀錄)")
    
    success_count = 0
    missing_img_count = 0

    # 每一行對應一個 frame_idx
    for frame_idx, line in enumerate(lines):
        try:
            row = line.strip().split(',')
            
            # 從你的陣列定義中擷取 GPS 資訊:
            # sensor_data[9] = lat, sensor_data[10] = lon, sensor_data[11] = alt
            lat = float(row[9])
            lon = float(row[10])
            alt = float(row[11])

            print(f"  處理 frame {frame_idx}: lat={lat}, lon={lon}, alt={alt}")

            # 轉換為 EXIF 格式
            lat_deg = to_deg(lat, ["S", "N"])
            lng_deg = to_deg(lon, ["W", "E"])

            exif_ifd = {
                piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
                piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1, # 0=海平面上, 1=海平面下
                piexif.GPSIFD.GPSAltitude: change_to_rational(round(abs(alt), 2)),
                # Python3 的 piexif 要求 Ref 為 ascii bytes
                piexif.GPSIFD.GPSLatitudeRef: lat_deg[3].encode('ascii') if lat_deg[3] else b'N',
                piexif.GPSIFD.GPSLatitude: [change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2])],
                piexif.GPSIFD.GPSLongitudeRef: lng_deg[3].encode('ascii') if lng_deg[3] else b'E',
                piexif.GPSIFD.GPSLongitude: [change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2])],
            }
            exif_dict = {"GPS": exif_ifd}
            exif_bytes = piexif.dump(exif_dict)

            # 將 EXIF 寫入 4 顆相機對應的影像
            for cam_id in range(4):
                img_name = f"frame{frame_idx:08d}_{cam_id}.jpg"
                img_path = os.path.join(folder_path, img_name)
                
                if os.path.exists(img_path):
                    piexif.insert(exif_bytes, img_path)
                    success_count += 1
                else:
                    missing_img_count += 1

        except Exception as e:
            print(f"  [錯誤] 處理 frame {frame_idx} 時發生異常: {e}")
            
    print(f"  ✓ 完成！成功寫入 {success_count} 張圖片的 EXIF。")
    if missing_img_count > 0:
        print(f"  ! 找不到 {missing_img_count} 張圖片 (可能中斷未存完)")

# ==========================================
# 主程式進入點
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="事後批次處理：將 data.txt 中的 MAVLink 座標寫入 JPG 的 EXIF")
    parser.add_argument(
        '-d', '--dir', 
        type=str, 
        default='data_openfile_1080p', 
        help="資料集的最外層主資料夾路徑 (預設: data_openfile_1080p)"
    )
    
    args = parser.parse_args()
    root_dir = args.dir

    if not os.path.exists(root_dir):
        print(f"錯誤：找不到資料夾 '{root_dir}'")
        exit(1)

    # 尋找所有的 roundXXXXX 資料夾
    round_folders = glob.glob(os.path.join(root_dir, 'round*'))
    
    if not round_folders:
        print(f"在 '{root_dir}' 中找不到任何以 'round' 開頭的資料夾。")
    else:
        # 排序確保依照順序處理
        for folder in sorted(round_folders):
            if os.path.isdir(folder):
                process_folder(folder)