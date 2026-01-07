import cv2

def get_supported_resolutions(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开相机！")
        return []
    
    resolutions = []
    # 常见分辨率档位（覆盖大部分相机）
    test_res = [
        (3840, 2160), (2592, 1944), (2048, 1536), (1920, 1080),
        (1280, 720), (640, 480)
    ]
    
    for w, h in test_res:
        # 先设置宽度，再设置高度（部分相机要求顺序）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # 获取实际设置的分辨率
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = (actual_w, actual_h)
        if res not in resolutions:
            resolutions.append(res)
    
    cap.release()
    return resolutions

# 执行检测
camera_index = 0  # 你的相机索引
supported_res = get_supported_resolutions(camera_index)
print(f"相机{camera_index}支持的分辨率：{supported_res}")