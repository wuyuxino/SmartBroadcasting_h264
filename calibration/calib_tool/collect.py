import cv2
import os
import time
from config import settings

# ===================== 基础配置 =====================
CAMERA_INDEX = 0
SAVE_FOLDER = "captured_images"

# 创建保存目录
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 打开相机
cap = cv2.VideoCapture(CAMERA_INDEX)

# 尝试设置4K分辨率，如果不支持会自动降级
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.IMAGE_HEIGHT)

# 获取实际分辨率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"相机分辨率: {width}x{height}")

print("简单拍照程序")
print("按【空格键】拍照")
print("按【q键】退出")
print("=" * 40)

photo_count = 0

try:
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机画面")
            break
        
        # 显示当前画面（缩小到适合屏幕）
        display_frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("拍照程序 (按空格拍照，q退出)", display_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # 空格键拍照
            photo_count += 1
            filename = f"{SAVE_FOLDER}/photo_{photo_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存第 {photo_count} 张照片: {filename}")
            time.sleep(0.5)  # 拍照后暂停0.5秒，避免连续拍照
            
        elif key == ord('q'):  # q键退出
            print("退出程序")
            break

except KeyboardInterrupt:
    print("\n程序被中断")

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n总共拍摄了 {photo_count} 张照片")
    print(f"照片保存在: {SAVE_FOLDER}")