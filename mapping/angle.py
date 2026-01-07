import numpy as np
import cv2
from sklearn.metrics import r2_score  # 评估拟合精度（可选，需安装：pip install scikit-learn）

# ===================== 1. 核心参数（你的标定数据） =====================
# 相机标定参数（从你的输出中复制）
CALIB_PARAMS_PATH = r"D:\\work\\code\\SmartBroadcasting\\calibration\\calib_tool\\camera_calib_params.npz"
# 标定点数据：[(u, v), (pan, tilt)]
CALIB_POINTS = [
    ((367, 268), (-23.32, -16.88)),
    ((876, 254), (32.92, -11.47)),
    ((168, 485), (-43.12, -30.00)),
    ((1156, 404), (55.12, -24.15)),
    ((670, 319), (11.62, -24.52)),
    ((697, 369), (15.22, -30.00)),
]
# 云台角度限位（根据你的设备调整）
PAN_RANGE = (-168, 168)    # Pan最大最小角度
TILT_RANGE = (-30, 90)    # Tilt最大最小角度

# ===================== 2. 加载相机标定参数（畸变矫正用） =====================
def load_camera_calib_params():
    """加载相机标定的内参、畸变系数、ROI等参数"""
    try:
        calib_data = np.load(CALIB_PARAMS_PATH)
        params = {
            "K": calib_data["K"],          # 原始内参
            "new_K": calib_data["new_K"],  # 最优内参（畸变矫正用）
            "D": calib_data["D"],          # 畸变系数
            "roi": calib_data["roi"],      # ROI裁剪参数
            "cx": calib_data["new_K"][0, 2],  # 图像中心x
            "cy": calib_data["new_K"][1, 2],  # 图像中心y
        }
        print("✅ 相机标定参数加载成功：")
        print(f"   图像中心 (cx, cy) = ({params['cx']:.1f}, {params['cy']:.1f})")
        print(f"   ROI裁剪区域 = {params['roi']}")
        return params
    except Exception as e:
        print(f"❌ 加载相机参数失败：{e}")
        return None

# ===================== 3. 线性回归拟合UV→云台角度的系数 =====================
def fit_uv2pt_coeffs(calib_points):
    """
    从标定点拟合线性系数：
    Pan = K_pan * u + pan0
    Tilt = K_tilt * v + tilt0
    返回：K_pan, pan0, K_tilt, tilt0
    """
    # 拆分标定点数据
    u_list = [p[0][0] for p in calib_points]
    v_list = [p[0][1] for p in calib_points]
    pan_list = [p[1][0] for p in calib_points]
    tilt_list = [p[1][1] for p in calib_points]

    # 线性回归拟合Pan（u → pan）
    K_pan, pan0 = np.polyfit(u_list, pan_list, deg=1)  # deg=1表示一次线性
    # 线性回归拟合Tilt（v → tilt）
    K_tilt, tilt0 = np.polyfit(v_list, tilt_list, deg=1)

    # 计算拟合精度（R²越接近1越好）
    pan_pred = K_pan * np.array(u_list) + pan0
    tilt_pred = K_tilt * np.array(v_list) + tilt0
    pan_r2 = r2_score(pan_list, pan_pred)
    tilt_r2 = r2_score(tilt_list, tilt_pred)

    print("\n✅ 线性拟合结果：")
    print(f"   Pan公式：Pan = {K_pan:.4f} × u + ({pan0:.4f}) （R² = {pan_r2:.4f}）")
    print(f"   Tilt公式：Tilt = {K_tilt:.4f} × v + ({tilt0:.4f}) （R² = {tilt_r2:.4f}）")

    return K_pan, pan0, K_tilt, tilt0

# ===================== 4. UV→云台角度计算函数（核心） =====================
def uv2pt(u, v, K_pan, pan0, K_tilt, tilt0):
    """
    从像素坐标计算云台角度
    :param u/v: 畸变矫正后的像素坐标
    :return: (pan, tilt) 云台角度（已做限位）
    """
    # 计算原始角度
    pan = K_pan * u + pan0
    tilt = K_tilt * v + tilt0

    # 云台角度限位（防止超出机械范围）
    pan = np.clip(pan, PAN_RANGE[0], PAN_RANGE[1])
    tilt = np.clip(tilt, TILT_RANGE[0], TILT_RANGE[1])

    return round(pan, 2), round(tilt, 2)

# ===================== 5. 畸变矫正函数（实际使用时需先矫正图像） =====================
def undistort_image(img, calib_params):
    """
    对原始图像做畸变矫正 + ROI裁剪
    :param img: 原始BGR图像（1280×720）
    :param calib_params: 相机标定参数（load_camera_calib_params返回的字典）
    :return: 矫正后的图像
    """
    # 畸变矫正
    undist_img = cv2.undistort(
        img,
        calib_params["K"],
        calib_params["D"],
        None,
        calib_params["new_K"]
    )
    # ROI裁剪（去除黑边）
    x, y, w, h = calib_params["roi"]
    undist_img_crop = undist_img[y:y+h, x:x+w]
    return undist_img_crop

# ===================== 6. 验证拟合效果（对标定点计算误差） =====================
def verify_fit_result(calib_points, K_pan, pan0, K_tilt, tilt0):
    """验证标定点的拟合误差"""
    print("\n📊 标定点拟合误差验证：")
    print("-" * 60)
    print(f"{'序号':<4} {'UV坐标':<12} {'实际角度(P,T)':<20} {'计算角度(P,T)':<20} {'偏差(P,T)':<15}")
    print("-" * 60)

    total_pan_error = 0.0
    total_tilt_error = 0.0
    for i, (uv, pt) in enumerate(calib_points):
        u, v = uv
        pan_true, tilt_true = pt
        pan_calc, tilt_calc = uv2pt(u, v, K_pan, pan0, K_tilt, tilt0)

        # 计算误差
        pan_error = abs(pan_calc - pan_true)
        tilt_error = abs(tilt_calc - tilt_true)
        total_pan_error += pan_error
        total_tilt_error += tilt_error

        print(f"{i+1:<4} ({u:<4},{v:<4})    ({pan_true:<6.2f},{tilt_true:<6.2f})    ({pan_calc:<6.2f},{tilt_calc:<6.2f})    ({pan_error:<5.2f},{tilt_error:<5.2f})")

    # 平均误差
    avg_pan_error = total_pan_error / len(calib_points)
    avg_tilt_error = total_tilt_error / len(calib_points)
    print("-" * 60)
    print(f"平均偏差：Pan = {avg_pan_error:.2f}°，Tilt = {avg_tilt_error:.2f}°")
    print("-" * 60)

# ===================== 7. 主函数（示例调用） =====================
if __name__ == "__main__":
    # 步骤1：加载相机标定参数
    calib_params = load_camera_calib_params()
    if not calib_params:
        exit()

    # 步骤2：拟合UV→云台角度的系数
    K_pan, pan0, K_tilt, tilt0 = fit_uv2pt_coeffs(CALIB_POINTS)

    # 步骤3：验证拟合效果
    verify_fit_result(CALIB_POINTS, K_pan, pan0, K_tilt, tilt0)

    # 步骤4：示例：输入任意UV坐标计算云台角度
    print("\n🔍 示例计算：")
    # 示例1：输入标定点1的UV，验证计算结果
    u_test1, v_test1 = 367, 268
    pan1, tilt1 = uv2pt(u_test1, v_test1, K_pan, pan0, K_tilt, tilt0)
    print(f"   UV({u_test1}, {v_test1}) → 云台角度(Pan={pan1}°, Tilt={tilt1}°)")

    # 示例2：输入图像中心的UV，计算对应角度
    u_center = calib_params["cx"]
    v_center = calib_params["cy"]
    pan_center, tilt_center = uv2pt(u_center, v_center, K_pan, pan0, K_tilt, tilt0)
    print(f"   图像中心UV({u_center:.1f}, {v_center:.1f}) → 云台角度(Pan={pan_center}°, Tilt={tilt_center}°)")

    # 示例3：输入自定义UV（比如新的检测点）
    u_custom, v_custom = 800, 300
    pan_custom, tilt_custom = uv2pt(u_custom, v_custom, K_pan, pan0, K_tilt, tilt0)
    print(f"   自定义UV({u_custom}, {v_custom}) → 云台角度(Pan={pan_custom}°, Tilt={tilt_custom}°)")

    # （可选）畸变矫正示例：对原始图像矫正后取UV
    # 假设你有一张原始全景图
    # raw_img = cv2.imread("your_raw_image.png")
    # if raw_img is not None:
    #     undist_img = undistort_image(raw_img, calib_params)
    #     cv2.imwrite("undistorted_image.png", undist_img)
    #     print("\n✅ 畸变矫正完成，保存为 undistorted_image.png")