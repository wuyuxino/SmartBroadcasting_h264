"""
配置项管理 - 所有全局配置、常量
"""

# 摄像头相关配置
IMAGE_WIDTH = 3840                # 摄像头初始化图像宽度 3840/2560/1920/1280/1280/640
IMAGE_HEIGHT = 2160               # 摄像头初始化图像高度 2160/1440/1080/720/720/480

# 检测模型预热加载尺寸 必须为32倍数
MODEL_WARMUP_SIZE = (3840, 2176)  # (width, height)

CAMERA_INDEX = 0                  # 默认摄像头索引  # Windows系统下，0为默认摄像头，1为外接摄像头，依此类推
CAMERA_INDEX_2 = 2                # 第二路摄像头索引（双路模式）
# 是否强制将摄像头输出设置为 MJPG（有助于减轻CPU解码开销，若相机支持）
FORCE_CAPTURE_MJPG = True

# 检测与缓存配置
FRAME_CACHE_LEN = 10              # 缓存最近10帧用于预测模型
CONF_THRESHOLD = 0.6             # 置信度阈值

# 预测接口配置
PREDICT_API_URL = "http://localhost:8000/predict"
REQUEST_TIMEOUT = (0.01, 0.02)    # 预测请求超时时间（秒）此处可以设置双值 [连接耗时, 读取耗时]

# 线程管理
THREAD_JOIN_TIMEOUT = 2.0           # 线程退出等待超时时间

# 裁剪偏移量 给检测算法输入裁剪后的宽高
CALIB_OFFSET_X = 0
CALIB_OFFSET_Y = 0
CALIB_OFFSET_H = 0
CALIB_OFFSET_W = 0

# 云台控制相关配置
ANGLE_THRESHOLD = 5.0  # 云台角度变化阈值，单位：度

# 根据帧数调整预测逻辑
USE_PREDICTION_AFTER_FRAMES = 720000  # 达到该帧数后启用预测逻辑 10帧为调用预测逻辑，否则为检测结果

# 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣
# 显示缩放：窗体显示时按比例缩放（例如 0.6 表示显示为原图 60% 大小）
DISPLAY_SCALE = 0.5
# 合成显示左右交换（True 表示把第二路放左边、第一路放右边）
COMBINE_SWAP = True

PTZ_PAN_INVERT = False
PTZ_PAN_OFFSET = 0.0
# 可选：按摄像头索引配置单独的翻转/偏移（默认空）
PTZ_PAN_INVERT_MAP = {}
PTZ_PAN_OFFSET_MAP = {}

# Tilt 偏移支持（按摄像头或全局）
PTZ_TILT_OFFSET = 0.0
PTZ_TILT_OFFSET_MAP = {}

# 自动标定映射（按摄像头）：默认关闭（空字典）。
# 要启用自动标定，请填写 PTZ_PAN_AUTO_CALIB_MAP/PTZ_TILT_AUTO_CALIB_MAP
PTZ_PAN_AUTO_CALIB_MAP = {}
PTZ_TILT_AUTO_CALIB_MAP = {}

# 在 config.py 中新增配置
class CameraGlobalConfig:
    # 摄像头FOV（拆分水平/垂直）
    SINGLE_FOV_H = 90  # 单摄像头水平FOV(°)
    SINGLE_FOV_V = 60  # 单摄像头垂直FOV(°)

    # 目标高度（替代硬编码的0.1m）
    TARGET_HEIGHT = 0.1  # 目标离地高度(m)

    # 云台物理范围（可选）
    PTZ_TILT_MIN = -30   # 云台垂直最小角度(°)
    PTZ_TILT_MAX = 90    # 云台垂直最大角度(°)

    # 其他原有配置（左/右摄像头角度、位置，云台位置等）
    LEFT_CAM_ANGLE = -30.0
    LEFT_CAM_POSITION = (-0.05, 0.0, 1.75)  # 摄像头全局坐标(x,y,z)，z为安装高度(m)
    RIGHT_CAM_ANGLE = 30.0  # 80°-26°=54°，重合26°
    RIGHT_CAM_POSITION = (0.05, 0.0, 1.75)
    PTZ_POSITION = (0.0, 0.0, 1.95)  # 云台全局坐标(x,y,z)
    