import numpy as np
import cv2
from typing import Optional

# ===================== 核心配置（从标定结果复制） =====================
# 1. 标定参数
CALIB_PARAMS_PATH = r"D:\work\code\SmartBroadcasting\calibration\calib_tool\camera_calib_params.npz"
# 2. 拟合得到的系数（从上面的推导复制）
FIT_K = 107.43    # 比例系数
FIT_B = -0.3717   # 截距
# 3. 足球真实直径（m）
FOOTBALL_REAL_DIAMETER = 0.15

# ===================== 加载标定参数 =====================
def load_calib_params():
    """加载相机标定参数（用于畸变校正）"""
    calib_data = np.load(CALIB_PARAMS_PATH)
    return {
        "K": calib_data["K"],          # 原始内参
        "new_K": calib_data["new_K"],  # 最优内参
        "D": calib_data["D"],          # 畸变系数
        "roi": calib_data["roi"],      # ROI裁剪区域
        "image_size": calib_data["image_size"]  # 图像尺寸 (1280,720)
    }

# ===================== 像素直径→距离映射函数 =====================
def football_pixel2distance(d_pixel: float, use_simple: bool = False) -> float:
    """
    足球像素直径转实际距离
    :param d_pixel: 足球的像素直径（float）
    :param use_simple: 是否使用简化公式（D=100.16/d_pixel）
    :return: 相机到足球的实际距离（m）
    """
    if d_pixel <= 0:
        raise ValueError("像素直径必须大于0")
    
    if use_simple:
        # 简化公式（无截距）
        distance = 100.16 / d_pixel
    else:
        # 完整拟合公式
        distance = FIT_K / d_pixel + FIT_B
    
    # 距离不能为负（兜底）
    return max(distance, 0.1)  # 最小距离限制为0.1m

# ===================== 可选：畸变校正后计算像素直径 =====================
def correct_distortion(img: np.ndarray, calib_params: dict) -> np.ndarray:
    """
    对图像进行畸变校正（广角相机必须做，否则像素尺寸有误差）
    :param img: 原始图像（BGR格式）
    :param calib_params: 标定参数（load_calib_params返回的字典）
    :return: 校正后的图像
    """
    h, w = img.shape[:2]
    # 畸变校正
    dst = cv2.undistort(
        img,
        calib_params["K"],
        calib_params["D"],
        None,
        calib_params["new_K"]
    )
    # 裁剪ROI（可选，标定结果推荐ROI：(0,0,1280,700)）
    x, y, w_roi, h_roi = calib_params["roi"]
    dst = dst[y:y+h_roi, x:x+w_roi]
    return dst



# ===================== 核心拟合参数（从推导结果复制） =====================
# 线性拟合系数：f = a*D + b
FIT_A = 1.856   # 斜率
FIT_B = -4.084  # 截距

# 变焦倍数/距离的合理边界（避免异常值）
MIN_ZOOM = 1.0   # 最小变焦倍数（实测最小为1）
MAX_ZOOM = 10.0  # 最大变焦倍数（实测最大为9.1，留余量）
MIN_DIST = 2.5   # 最小距离（小于该值时公式失效）
MAX_DIST = 8.0   # 最大距离（大于该值时公式失效）

# ===================== 距离→变焦倍数（核心函数） =====================
def distance_to_zoom(distance: float, clamp: bool = True) -> Optional[float]:
    """
    实测距离 → 云台变焦倍数（基于线性拟合公式）
    :param distance: 相机到足球的距离（m）
    :param clamp: 是否限制变焦倍数在[MIN_ZOOM, MAX_ZOOM]范围内
    :return: 推荐变焦倍数（异常返回None）
    """
    # 输入校验
    if not isinstance(distance, (int, float)):
        print("❌ 距离必须是数字！")
        return None
    if distance < 0:
        print("❌ 距离不能为负数！")
        return None
    
    # 公式计算变焦倍数
    zoom = FIT_A * distance + FIT_B
    
    # 边界限制（可选）
    if clamp:
        if distance < MIN_DIST or distance > MAX_DIST:
            print(f"⚠️  距离{distance}m超出有效范围[{MIN_DIST},{MAX_DIST}]m，结果仅供参考")
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    
    # 兜底：避免计算出无意义的变焦
    if zoom < 0:
        zoom = MIN_ZOOM
    
    return round(zoom, 2)  # 保留2位小数，适配云台控制精度

# ===================== 变焦倍数→距离（反向映射，可选） =====================
def zoom_to_distance(zoom: float) -> Optional[float]:
    """
    云台变焦倍数 → 估算距离（反向推导）
    :param zoom: 云台变焦倍数
    :return: 估算距离（m），异常返回None
    """
    if not isinstance(zoom, (int, float)):
        print("❌ 变焦倍数必须是数字！")
        return None
    if zoom < MIN_ZOOM or zoom > MAX_ZOOM:
        print(f"❌ 变焦倍数{zoom}超出有效范围[{MIN_ZOOM},{MAX_ZOOM}]")
        return None
    
    # 反向公式：D = (f - b)/a
    distance = (zoom - FIT_B) / FIT_A
    return round(distance, 2)


# ===================== 测试示例 =====================
if __name__ == "__main__":
    # 1. 加载标定参数
    calib_params = load_calib_params()
    print("✅ 标定参数加载成功")
    
    # 2. 测试像素直径→距离
    distance = football_pixel2distance(12)
    print(f"足球距离：{distance:.2f}m")  # 输出：5.00m

    # 3. 测试距离→变焦倍数
    zoom = distance_to_zoom(distance)
    print(f"推荐变焦倍数：{zoom}倍")  # 输出：5.2倍