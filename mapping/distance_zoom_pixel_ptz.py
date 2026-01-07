# 固定值参数
## 云台机械参数
    # pan_range=(-170, 170)    # 水平旋转范围（根据云台型号手册确定）
    # tilt_range=(-30, 90)     # 垂直旋转范围（机械限位参数）
    # zoom_range=(1, 30)       # 变焦范围（镜头光学参数）
    # ball_diameter=220.0      # 足球标准直径（国际足联规定值）
## 传感器硬件参数 （已记录现有摄像头真实参数）
    # sensor_width=6.22        # 传感器宽度（需查相机手册，如索尼IMX415为6.4mm）
    # img_width=3840           # 图像分辨率（如4K摄像头固定参数）
## 相机内参 （当前固定位置的值）
    # 内参矩阵（需通过相机标定获得，固定后不变）
    # camera_matrix = np.array([
    #    [1475.44869, 0.0, 1881.75239],
    #    [0.0, 1477.58982, 1098.91895],
    #    [0.0, 0.0, 1.0]
    # ]) 
    # 畸变系数（标定后固定）
    # dist_coeffs = np.zeros([
    #    [-0.04357757],
    #    [-0.01357247],
    #    [-0.00237252],
    #    [-0.00432121],
    #    [0.00496599]
    # ]) 
## 计算中间参数
    # self.f_pano = camera_matrix[0,0]  # 焦距（由内参矩阵固定）
    # self.pixel_size = sensor_width/img_width  # 像素物理尺寸（固定换算）
    
    # 📐 f_pano（焦距，像素单位）: 1475.44869 像素
    # 📏 pixel_size（像素物理尺寸，mm/像素）: 0.00162000 mm/像素
    # 🔍 pixel_size（像素物理尺寸，μm/像素）: 1.620 μm/像素


# 计算值参数
## 角度映射参数
    # self.K_pan, self.K_tilt  # 通过标定数据线性回归计算得到
    # self.pan0, self.tilt0    # 初始角度（标定数据第一个点）
## 变焦拟合参数
    # self.zoom_a, self.zoom_b, self.zoom_c  # 二次多项式拟合系数
## 实时计算参数
    # pan, tilt = self.calc_pan_tilt((u,v))  # 根据像素坐标动态计算角度
    # distance = self.calc_distance(d_pixel)  # 根据像素直径计算距离
    # zoom = self.calc_zoom(distance)         # 根据距离计算变焦倍数

import cv2
import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
import scipy
import os

class PTZController:
    def __init__(self, camera_matrix, dist_coeffs, 
                 pan_range=(-170, 170), tilt_range=(-30, 90), zoom_range=(1, 30),
                 ball_diameter=220.0, sensor_width=6.4, img_width=3840):
        """
        云台控制器初始化
        :param camera_matrix: 相机内参矩阵（3x3）
        :param dist_coeffs: 相机畸变系数
        :param pan_range: 云台水平旋转范围 (min, max) °
        :param tilt_range: 云台垂直旋转范围 (min, max) °
        :param zoom_range: 云台变焦范围 (min, max) 倍
        :param ball_diameter: 足球实际直径（mm），标准220mm
        :param sensor_width: 相机传感器宽度（mm），需查相机手册（如6.4mm）
        :param img_width: 相机图像宽度（像素），如3840/1920/1280
        """
        # 相机基础参数
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.f_pano = camera_matrix[0, 0]  # 全景相机焦距（像素）mtx[0,0]
        self.ball_d = ball_diameter        # 足球直径（mm）
        self.sensor_w = sensor_width       # 传感器宽度（mm）
        self.img_w = img_width             # 图像宽度（像素）
        self.pixel_size = self.sensor_w / self.img_w  # 像素物理尺寸（mm/pixel）
        
        # 云台范围限制
        self.pan_min, self.pan_max = pan_range
        self.tilt_min, self.tilt_max = tilt_range
        self.zoom_min, self.zoom_max = zoom_range
        
        # 角度映射参数（标定后赋值）
        self.K_pan = None    # 水平像素-角度系数（°/pixel）
        self.K_tilt = None   # 垂直像素-角度系数（°/pixel）
        self.pan0 = None     # 初始水平角度（°）
        self.tilt0 = None    # 初始垂直角度（°）
        self.cx = None       # 图像中心x（像素）
        self.cy = None       # 图像中心y（像素）
        
        # 变焦映射参数（标定后赋值）
        self.zoom_a = None   # 变焦拟合系数a
        self.zoom_b = None   # 变焦拟合系数b
        self.zoom_c = None   # 变焦拟合系数c

    # ===================== 1. 基础标定方法（复用+适配） =====================
    def calibrate_angle(self, world_points, pixel_points, pan_tilt_angles):
        """
        标定角度映射系数（Pan0/K_pan, Tilt0/K_tilt）
        :param world_points: 世界坐标 (Nx3)
        :param pixel_points: 像素坐标 (Nx2)
        :param pan_tilt_angles: 云台角度 (Nx2) [pan(°), tilt(°)]
        :return: 角度标定结果
        """
        if len(world_points) != len(pixel_points) or len(world_points) != len(pan_tilt_angles):
            raise ValueError("世界坐标/像素坐标/云台角度数量必须一致！")
        if len(world_points) < 3:
            raise ValueError("至少需要3个标定点！")
        
        # 提取图像中心
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        # 初始角度（第一个标定点）
        self.pan0 = pan_tilt_angles[0, 0]
        self.tilt0 = pan_tilt_angles[0, 1]
        
        # 构建线性方程组求解K_pan/K_tilt
        A_pan, b_pan = [], []
        A_tilt, b_tilt = [], []
        for i in range(1, len(world_points)):
            u, v = pixel_points[i]
            pan, tilt = pan_tilt_angles[i]
            
            # 像素偏移
            du = u - self.cx
            dv = v - self.cy
            
            # 角度偏移
            delta_pan = pan - self.pan0
            delta_tilt = tilt - self.tilt0
            
            A_pan.append([du])
            b_pan.append(delta_pan)
            A_tilt.append([dv])
            b_tilt.append(delta_tilt)
        
        # 最小二乘求解
        A_pan = np.array(A_pan)
        b_pan = np.array(b_pan)
        A_tilt = np.array(A_tilt)
        b_tilt = np.array(b_tilt)
        
        try:
            self.K_pan = linalg.lstsq(A_pan, b_pan)[0][0]
            self.K_tilt = linalg.lstsq(A_tilt, b_tilt)[0][0]
        except:
            self.K_pan = linalg.lstsq(A_pan, b_pan, rcond=None)[0][0]
            self.K_tilt = linalg.lstsq(A_tilt, b_tilt, rcond=None)[0][0]
        
        # 计算标定误差
        errors_pan, errors_tilt = [], []
        for i in range(len(world_points)):
            u, v = pixel_points[i]
            true_pan, true_tilt = pan_tilt_angles[i]
            pred_pan, pred_tilt = self.calc_pan_tilt((u, v))
            errors_pan.append(abs(pred_pan - true_pan))
            errors_tilt.append(abs(pred_tilt - true_tilt))
        
        return {
            'K_pan': self.K_pan,
            'K_tilt': self.K_tilt,
            'pan0': self.pan0,
            'tilt0': self.tilt0,
            'max_error_pan': max(errors_pan),
            'avg_error_pan': np.mean(errors_pan),
            'max_error_tilt': max(errors_tilt),
            'avg_error_tilt': np.mean(errors_tilt)
        }

    def calibrate_zoom(self, distance_list, zoom_list):
        """
        标定变焦映射系数（a/b/c）
        :param distance_list: 实测距离（m）
        :param zoom_list: 对应变焦倍数
        :return: 变焦标定结果
        """
        D = np.array(distance_list, dtype=np.float64)
        Zoom = np.array(zoom_list, dtype=np.float64)
        
        if len(D) != len(Zoom) or len(D) < 3:
            raise ValueError("距离/变焦列表长度≥3且必须一致！")
        if np.min(D) <= 0:
            raise ValueError("距离必须为正数！")
        
        # 二次多项式拟合
        def zoom_func(D, a, b, c):
            return a * D**2 + b * D + c
        
        popt, _ = curve_fit(zoom_func, D, Zoom)
        self.zoom_a, self.zoom_b, self.zoom_c = popt
        
        # 计算拟合误差
        pred_zoom = zoom_func(D, self.zoom_a, self.zoom_b, self.zoom_c)
        max_error = np.abs(pred_zoom - Zoom).max()
        avg_error = np.mean(np.abs(pred_zoom - Zoom))
        
        return {
            'zoom_formula': f"Zoom = {self.zoom_a:.6f}*D² + {self.zoom_b:.3f}*D + {self.zoom_c:.1f}",
            'a': self.zoom_a,
            'b': self.zoom_b,
            'c': self.zoom_c,
            'max_error': max_error,
            'avg_error': avg_error
        }

    # ===================== 2. 核心计算方法（算法核心） =====================
    def calc_pan_tilt(self, pixel_coord):
        """
        像素坐标→云台角度（带范围限制）
        :param pixel_coord: 足球像素坐标 (u, v)
        :return: (pan, tilt) 角度（°）
        """
        if self.K_pan is None or self.K_tilt is None:
            raise RuntimeError("请先执行calibrate_angle()标定角度系数！")
        
        u, v = pixel_coord
        # 基础公式：Pan = Pan0 + K_pan*(u - cx)；Tilt = Tilt0 + K_tilt*(v - cy)
        pan = self.pan0 + self.K_pan * (u - self.cx)
        tilt = self.tilt0 + self.K_tilt * (v - self.cy)
        
        # 范围限制（避免云台卡死）
        pan = np.clip(pan, self.pan_min, self.pan_max)
        tilt = np.clip(tilt, self.tilt_min, self.tilt_max)
        
        return pan, tilt

    def calc_distance(self, d_pixel):
        """
        足球像素直径→距离D（成像原理推导）
        公式：D (m) = (Ball_D(mm) * f_pano(pixel)) / (d_pixel(pixel) * pixel_size(mm/pixel)) / 1000
        :param d_pixel: 足球像素直径（像素）
        :return: 距离D（m）
        """
        if d_pixel <= 0:
            raise ValueError("足球像素直径必须>0！")
        
        # 核心计算（单位换算：mm→m）
        D_mm = (self.ball_d * self.f_pano) / (d_pixel * self.pixel_size)
        D_m = D_mm / 1000  # 转米
        
        # 距离下限（避免异常值）
        return max(D_m, 0.1)

    def calc_zoom(self, distance):
        """
        距离→变焦倍数（带范围限制）
        :param distance: 距离D（m）
        :return: 变焦倍数
        """
        if self.zoom_a is None or self.zoom_b is None or self.zoom_c is None:
            raise RuntimeError("请先执行calibrate_zoom()标定变焦系数！")
        if distance <= 0:
            raise ValueError("距离必须为正数！")
        
        # 基础公式：Zoom = a*D² + b*D + c
        zoom = self.zoom_a * (distance**2) + self.zoom_b * distance + self.zoom_c
        
        # 范围限制
        zoom = np.clip(zoom, self.zoom_min, self.zoom_max)
        
        return zoom

    # ===================== 3. 云台控制指令（ONVIF模拟+实际适配） =====================
    def get_ptz_commands(self, pixel_coord, d_pixel):
        """
        完整云台控制流程：像素坐标→角度→距离→变焦→指令
        :param pixel_coord: 足球像素坐标 (u, v)
        :param d_pixel: 足球像素直径（像素）
        :return: 控制指令字典（可直接对接ONVIF库）
        """
        # 步骤1：计算云台旋转角度
        pan, tilt = self.calc_pan_tilt(pixel_coord)
        
        # 步骤2：计算距离
        distance = self.calc_distance(d_pixel)
        
        # 步骤3：计算变焦倍数
        zoom = self.calc_zoom(distance)
        
        # 封装ONVIF指令（模拟格式，实际需对接ONVIF库如onvif-zeep）
        commands = {
            'Pan': round(pan, 1),          # 水平角度（保留1位小数）
            'Tilt': round(tilt, 1),        # 垂直角度
            'Zoom': round(zoom, 1),        # 变焦倍数
            'Distance': round(distance, 1),# 计算的距离
            'Pixel_Coord': pixel_coord,    # 输入像素坐标
            'Pixel_Diameter': d_pixel,     # 输入像素直径
            'Status': 'Ready'              # 状态
        }
        
        return commands

    def send_onvif_command(self, commands):
        """
        模拟发送ONVIF指令（实际项目替换为真实ONVIF调用）
        :param commands: get_ptz_commands()返回的指令字典
        :return: 发送结果
        """
        # 真实场景需替换为：
        # 1. 导入onvif库（pip install onvif-zeep）
        # 2. 连接云台设备（IP/用户名/密码）
        # 3. 发送PTZ控制指令
        # 以下为模拟逻辑
        try:
            print(f"\n📡 发送ONVIF控制指令：")
            print(f"  水平旋转（Pan）: {commands['Pan']}°")
            print(f"  垂直旋转（Tilt）: {commands['Tilt']}°")
            print(f"  变焦倍数（Zoom）: {commands['Zoom']}x")
            print(f"  目标距离（D）: {commands['Distance']}m")
            return {
                'success': True,
                'message': '指令发送成功',
                'commands': commands
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'指令发送失败：{str(e)}',
                'commands': commands
            }

# ===================== 测试与示例 =====================
if __name__ == "__main__":
    # 1. 加载相机标定参数（替换为实际路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_file = os.path.join(script_dir, "camera_calib_params.npz")
    if not os.path.exists(calib_file):
        # 模拟相机参数（无文件时使用）
        camera_matrix = np.array([[2000, 0, 1920],  # f_pano=2000像素，图像中心(1920,1080)
                                  [0, 2000, 1080],
                                  [0, 0, 1]])
        dist_coeffs = np.zeros((5, 1))
    else:
        calib_data = np.load(calib_file)
        camera_matrix = calib_data['K']
        dist_coeffs = calib_data['D']

    # 2. 初始化云台控制器（适配你的云台参数）
    ptz = PTZController(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pan_range=(-170, 170),    # 水平旋转范围（根据云台型号调整）
        tilt_range=(-30, 90),     # 垂直旋转范围
        zoom_range=(1, 30),       # 变焦范围
        ball_diameter=220.0,      # 足球直径220mm
        sensor_width=6.4,         # 传感器宽度6.4mm（查相机手册）
        img_width=3840            # 图像宽度3840像素（4K）
    )

    # 3. 标定角度映射系数（替换为实际标定数据）
    world_points = np.array([[1000, 5000, 0], [0, 0, 0], [-1000, 5000, 0]])
    pixel_points = np.array([[2120, 1080], [1920, 1080], [1720, 1080]])
    pan_tilt_angles = np.array([[10, 0], [0, 0], [-10, 0]])  # 初始角度Pan0=0°, Tilt0=0°
    angle_result = ptz.calibrate_angle(world_points, pixel_points, pan_tilt_angles)
    print("="*60)
    print("📌 角度标定结果")
    print("="*60)
    print(f"Pan0: {angle_result['pan0']}°, K_pan: {angle_result['K_pan']:.6f} °/pixel")
    print(f"Tilt0: {angle_result['tilt0']}°, K_tilt: {angle_result['K_tilt']:.6f} °/pixel")
    print(f"角度标定最大误差: Pan={angle_result['max_error_pan']:.2f}°, Tilt={angle_result['max_error_tilt']:.2f}°")

    # 4. 标定变焦映射系数（替换为实际实测数据）
    distance_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    zoom_list = [5, 8, 12, 16, 19, 22, 24, 26, 27, 29]
    zoom_result = ptz.calibrate_zoom(distance_list, zoom_list)
    print("\n" + "="*60)
    print("📌 变焦标定结果")
    print("="*60)
    print(f"拟合公式: {zoom_result['zoom_formula']}")
    print(f"变焦拟合最大误差: {zoom_result['max_error']:.1f} 倍")

    # 5. 模拟实时检测数据（替换为实际检测结果）
    detected_pixel = (2200, 1150)  # 足球像素坐标(u,v)
    detected_d_pixel = 20          # 足球像素直径（像素）

    # 6. 计算云台控制指令并发送
    ptz_commands = ptz.get_ptz_commands(detected_pixel, detected_d_pixel)
    send_result = ptz.send_onvif_command(ptz_commands)

    # 7. 输出最终结果
    print("\n" + "="*60)
    print("📌 最终控制结果")
    print("="*60)
    print(f"指令发送状态: {'✅ 成功' if send_result['success'] else '❌ 失败'}")
    if send_result['success']:
        print(f"核心控制参数:")
        print(f"  目标像素: {detected_pixel} → 云台角度: Pan={ptz_commands['Pan']}°, Tilt={ptz_commands['Tilt']}°")
        print(f"  像素直径: {detected_d_pixel}px → 距离: {ptz_commands['Distance']}m → 变焦: {ptz_commands['Zoom']}x")
    else:
        print(f"失败原因: {send_result['message']}")