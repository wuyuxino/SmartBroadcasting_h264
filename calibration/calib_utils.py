"""
相机标定相关 - 标定参数加载、畸变矫正
"""
import cv2
import numpy as np

def load_calib_params(calib_file="camera_calib_params.npz"):
    """加载标定参数（兼容无标定文件）"""
    try:
        calib_data = np.load(calib_file)
        K = calib_data["K"]
        D = calib_data["D"]
        return K, D
    except FileNotFoundError:
        print("⚠️ 未找到标定文件，使用原始帧检测")
        return None, None

def init_undistort_map(K, D, img_size):
    """初始化矫正映射表（兼容无标定）"""
    if K is None or D is None:
        return None, None, None, None
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, img_size, 1, img_size)
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, img_size, cv2.CV_32FC1)
    return mapx, mapy, new_K, roi