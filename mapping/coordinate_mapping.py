# coordinate_mapping.py
import numpy as np
import cv2
import math
from config import settings
from config.settings import CameraGlobalConfig

class CameraCoordinateMapper:
    """摄像头坐标映射器"""
    
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.setup_camera_params()
        # 尝试应用运行时自动标定（如果在 settings 中配置了校准点）
        try:
            self._apply_auto_calib()
        except Exception:
            pass
        
    def setup_camera_params(self):
        """根据摄像头索引设置参数"""
        if self.camera_index == settings.CAMERA_INDEX:
            # 左摄像头
            self.cam_angle = CameraGlobalConfig.LEFT_CAM_ANGLE
            self.cam_position = CameraGlobalConfig.LEFT_CAM_POSITION
            self.fov = CameraGlobalConfig.SINGLE_FOV_H
            self.image_width = settings.IMAGE_WIDTH  # 需要定义
            self.image_height = settings.IMAGE_HEIGHT  # 需要定义
        elif self.camera_index == settings.CAMERA_INDEX_2:
            # 右摄像头
            self.cam_angle = CameraGlobalConfig.RIGHT_CAM_ANGLE
            self.cam_position = CameraGlobalConfig.RIGHT_CAM_POSITION
            self.fov = CameraGlobalConfig.SINGLE_FOV_H
            self.image_width = settings.IMAGE_WIDTH
            self.image_height = settings.IMAGE_HEIGHT
        else:
            raise ValueError(f"Unsupported camera index: {self.camera_index}")
        
        # 计算焦距（像素单位）
        self.focal_length_px = (self.image_width / 2) / math.tan(math.radians(self.fov / 2))
    
    def pixel_to_camera_coords(self, pixel_x, pixel_y):
        """将像素坐标转换为摄像头坐标系（单位：米）"""
        # 像素坐标系：原点在图像中心
        x_cam = (pixel_x - self.image_width / 2) / self.focal_length_px
        y_cam = (pixel_y - self.image_height / 2) / self.focal_length_px
        
        # 假设目标在水平面上（z=0），或者根据实际高度调整
        # 这里简化处理，假设目标在固定高度
        target_height = 0.01  # 目标高度，单位：米（例如足球）
        
        return x_cam, y_cam, target_height
    
    def camera_to_global_coords(self, x_cam, y_cam, z_cam):
        """将摄像头坐标系转换为全局坐标系"""
        # 摄像头坐标系旋转
        theta = math.radians(self.cam_angle)
        
        # 旋转矩阵（绕Y轴旋转，假设摄像头在水平面）
        R = np.array([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        
        # 摄像头坐标系中的点（确保为一维数组）
        point_cam = np.array([x_cam, y_cam, z_cam], dtype=float)

        # 确保摄像头位置为三维向量（如果配置为2维则补齐z=0）
        cam_pos = np.array(self.cam_position, dtype=float)
        if cam_pos.size == 2:
            cam_pos = np.array([cam_pos[0], cam_pos[1], 0.0], dtype=float)

        # 转换到全局坐标系
        point_global = np.dot(R, point_cam) + cam_pos

        return point_global
    
    def global_to_ptz_angles(self, x_global, y_global, z_global):
        """将全局坐标转换为云台角度"""
        # 计算相对于云台的位置
        dx = x_global - CameraGlobalConfig.PTZ_POSITION[0]
        dy = y_global - CameraGlobalConfig.PTZ_POSITION[1]
        dz = z_global  # 假设云台高度为0
        
        # 计算水平角（Pan）
        pan = math.degrees(math.atan2(dx, dz))
        
        # 计算垂直角（Tilt）
        distance_horizontal = math.sqrt(dx**2 + dz**2)
        tilt = math.degrees(math.atan2(dy, distance_horizontal))
        
        return pan, tilt
    
    def pixel_to_ptz_angles(self, pixel_x, pixel_y):
        """完整转换：像素坐标 -> 云台角度"""
        # 步骤1：像素 -> 摄像头坐标
        x_cam, y_cam, z_cam = self.pixel_to_camera_coords(pixel_x, pixel_y)
        
        # 步骤2：摄像头坐标 -> 全局坐标
        x_global, y_global, z_global = self.camera_to_global_coords(x_cam, y_cam, z_cam)
        
        # 步骤3：全局坐标 -> 云台角度
        pan, tilt = self.global_to_ptz_angles(x_global, y_global, z_global)

        # 可选：将几何映射的 pan/tilt 转换为目标云台坐标系（符号/偏移/范围）
        pan = self.convert_pan_for_ptz(pan)
        tilt = self.convert_tilt_for_ptz(tilt)

        return pan, tilt

    def convert_pan_for_ptz(self, pan, invert=None, offset=None):
        """
        将几何映射得到的 Pan 角转换为 PTZ 所用的 Pan。

        参数:
          - pan: 输入角度（度）
          - invert: 是否反转符号（None 表示读取 settings.PTZ_PAN_INVERT，默认为 False）
          - offset: 角度偏移（度）。None 表示读取 settings.PTZ_PAN_OFFSET，默认为 0.

        返回值：规范化到 (-180, 180] 的角度（度）
        """
        # 读取默认配置（若存在）
        if invert is None:
            try:
                invert = bool(getattr(settings, 'PTZ_PAN_INVERT', False))
            except Exception:
                invert = False
        if offset is None:
            try:
                offset = float(getattr(settings, 'PTZ_PAN_OFFSET', 0.0))
            except Exception:
                offset = 0.0

        # 优先使用实例级自动计算的偏移（如果存在）
        inst_offset = getattr(self, '_ptz_pan_offset', None)
        inst_invert = getattr(self, '_ptz_pan_invert', None)
        if inst_invert is not None:
            invert = inst_invert
        if inst_offset is not None:
            offset = inst_offset

        p = -pan if invert else pan
        p = p + offset

        # 规范化到 (-180, 180]
        p = ((p + 180.0) % 360.0) - 180.0
        return p

    def convert_tilt_for_ptz(self, tilt, offset=None):
        """
        将几何映射得到的 Tilt 转换为 PTZ 所用的 Tilt（仅支持偏移）。

        参数:
          - tilt: 输入角度（度）
          - offset: 角度偏移（度）。None 表示读取 settings.PTZ_TILT_OFFSET 或按摄像头映射 PTZ_TILT_OFFSET_MAP

        返回值：tilt + offset
        """
        try:
            if offset is None:
                offset_map = getattr(settings, 'PTZ_TILT_OFFSET_MAP', None)
                if isinstance(offset_map, dict) and self.camera_index in offset_map:
                    offset = float(offset_map[self.camera_index])
                else:
                    offset = float(getattr(settings, 'PTZ_TILT_OFFSET', 0.0))
        except Exception:
            offset = 0.0

        return tilt + offset

    def _apply_auto_calib(self):
        """应用运行时自动标定：从 settings.PTZ_PAN_AUTO_CALIB_MAP / PTZ_TILT_AUTO_CALIB_MAP 读取期望映射，计算并保存实例偏移。

        PTZ_PAN_AUTO_CALIB_MAP 格式示例：{0: {'test_pixel': (x,y), 'expected_pan': deg}}
        PTZ_TILT_AUTO_CALIB_MAP 格式示例：{0: {'test_pixel': (x,y), 'expected_tilt': deg}}
        """
        # Pan 自动校准
        try:
            pan_map = getattr(settings, 'PTZ_PAN_AUTO_CALIB_MAP', None)
            if isinstance(pan_map, dict) and self.camera_index in pan_map:
                item = pan_map[self.camera_index]
                tx, ty = item['test_pixel']
                expected = float(item['expected_pan'])
                # 计算当前几何映射得到的原始 pan（不应用实例偏移）
                x_cam, y_cam, z_cam = self.pixel_to_camera_coords(tx, ty)
                xg, yg, zg = self.camera_to_global_coords(x_cam, y_cam, z_cam)
                raw_pan, raw_tilt = self.global_to_ptz_angles(xg, yg, zg)
                # 需要的偏移，使 raw_pan + offset -> expected（不考虑 invert）
                self._ptz_pan_offset = expected - raw_pan
                # 如果需要，也可设置实例级 invert
                if 'invert' in item:
                    self._ptz_pan_invert = bool(item['invert'])
        except Exception:
            pass

    def calibrate_pan_with_sample(self, test_pixel, expected_pan, invert=None):
        """使用单个采样点即时标定 pan 偏移：

        - test_pixel: (x, y) 像素坐标
        - expected_pan: 期望的 PTZ Pan（度）
        - invert: 可选，是否反转符号（覆盖实例值）

        方法会计算并设置实例级偏移 `self._ptz_pan_offset`，并返回计算后的 (raw_pan, adjusted_pan)
        """
        tx, ty = test_pixel
        x_cam, y_cam, z_cam = self.pixel_to_camera_coords(tx, ty)
        xg, yg, zg = self.camera_to_global_coords(x_cam, y_cam, z_cam)
        raw_pan, raw_tilt = self.global_to_ptz_angles(xg, yg, zg)
        # 先确定最终的 invert 值（参数覆盖实例，再到 settings）
        if invert is not None:
            effective_invert = bool(invert)
            self._ptz_pan_invert = effective_invert
        else:
            inst_invert = getattr(self, '_ptz_pan_invert', None)
            if inst_invert is not None:
                effective_invert = bool(inst_invert)
            else:
                try:
                    effective_invert = bool(getattr(settings, 'PTZ_PAN_INVERT', False))
                except Exception:
                    effective_invert = False

        # 计算并保存实例偏移：使 ( -raw if invert else raw ) + offset == expected
        if effective_invert:
            self._ptz_pan_offset = expected_pan + raw_pan
        else:
            self._ptz_pan_offset = expected_pan - raw_pan

        adjusted = (-raw_pan if effective_invert else raw_pan) + self._ptz_pan_offset
        # 规范化
        adjusted = ((adjusted + 180.0) % 360.0) - 180.0
        return raw_pan, adjusted

    def calibrate_tilt_with_sample(self, test_pixel, expected_tilt):
        """使用单个采样点即时标定 tilt 偏移，设置实例级 `_ptz_tilt_offset` 并返回 (raw_tilt, adjusted_tilt)。"""
        tx, ty = test_pixel
        x_cam, y_cam, z_cam = self.pixel_to_camera_coords(tx, ty)
        xg, yg, zg = self.camera_to_global_coords(x_cam, y_cam, z_cam)
        raw_pan, raw_tilt = self.global_to_ptz_angles(xg, yg, zg)
        self._ptz_tilt_offset = expected_tilt - raw_tilt
        adjusted = raw_tilt + self._ptz_tilt_offset
        return raw_tilt, adjusted

        # Tilt 自动校准
        try:
            tilt_map = getattr(settings, 'PTZ_TILT_AUTO_CALIB_MAP', None)
            if isinstance(tilt_map, dict) and self.camera_index in tilt_map:
                item = tilt_map[self.camera_index]
                tx, ty = item['test_pixel']
                expected = float(item['expected_tilt'])
                x_cam, y_cam, z_cam = self.pixel_to_camera_coords(tx, ty)
                xg, yg, zg = self.camera_to_global_coords(x_cam, y_cam, z_cam)
                raw_pan, raw_tilt = self.global_to_ptz_angles(xg, yg, zg)
                self._ptz_tilt_offset = expected - raw_tilt
        except Exception:
            pass