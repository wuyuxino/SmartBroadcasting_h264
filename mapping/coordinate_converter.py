# coordinate_converter.py
import numpy as np

class ResolutionConverter:
    """处理不同分辨率之间的坐标转换"""
    
    def __init__(self, camera_index):
        self.camera_index = camera_index
        
        # 检测分辨率（YOLO使用的分辨率）
        self.detect_width = 1280
        self.detect_height = 736
        
        # 原始图像分辨率（摄像头输出）
        self.original_width = 3840
        self.original_height = 2160
        
        # ROI参数（从标定结果获取）
        self.roi_x = 0  # ROI起始X
        self.roi_y = 0  # ROI起始Y
        self.roi_width = 3840  # ROI宽度
        self.roi_height = 2160  # ROI高度
        
        # 计算缩放比例
        self.scale_x_roi = self.roi_width / self.detect_width
        self.scale_y_roi = self.roi_height / self.detect_height
        
        print(f"📊 摄像头{camera_index}坐标转换器初始化:")
        print(f"  检测分辨率: {self.detect_width}×{self.detect_height}")
        print(f"  ROI区域: ({self.roi_x},{self.roi_y},{self.roi_width},{self.roi_height})")
        print(f"  原始分辨率: {self.original_width}×{self.original_height}")
        print(f"  缩放比例: X={self.scale_x_roi:.3f}, Y={self.scale_y_roi:.3f}")
    
    def detect_to_original(self, detect_x, detect_y):
        """
        将检测分辨率下的坐标转换到原始分辨率
        
        参数:
        - detect_x: 检测图像中的X坐标 (0~1279)
        - detect_y: 检测图像中的Y坐标 (0~735)
        
        返回:
        - (original_x, original_y): 原始图像中的坐标
        """
        # 步骤1: 检测坐标 -> ROI坐标
        roi_x = detect_x * self.scale_x_roi
        roi_y = detect_y * self.scale_y_roi
        
        # 步骤2: ROI坐标 -> 原始图像坐标
        original_x = roi_x + self.roi_x
        original_y = roi_y + self.roi_y
        
        # 确保坐标在合理范围内
        original_x = np.clip(original_x, 0, self.original_width - 1)
        original_y = np.clip(original_y, 0, self.original_height - 1)
        
        return int(original_x), int(original_y)
    
    def original_to_detect(self, original_x, original_y):
        """
        将原始分辨率下的坐标转换到检测分辨率（反向转换）
        """
        # 原始坐标 -> ROI坐标
        roi_x = original_x - self.roi_x
        roi_y = original_y - self.roi_y
        
        # ROI坐标 -> 检测坐标
        detect_x = roi_x / self.scale_x_roi
        detect_y = roi_y / self.scale_y_roi
        
        return int(detect_x), int(detect_y)
    
    def get_conversion_factors(self):
        """获取转换因子（用于直接计算）"""
        return {
            'scale_x': self.scale_x_roi,
            'scale_y': self.scale_y_roi,
            'offset_x': self.roi_x,
            'offset_y': self.roi_y
        }