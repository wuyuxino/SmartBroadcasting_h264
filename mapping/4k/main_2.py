# main_program_integration.py
"""
修改你的主程序，集成双摄像头管理
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import cv2
from collections import deque
import copy
import queue
import threading
import time
import numpy as np

from config import settings
from threads import thread_manage as threads
from predict import predict_utils 
from predict import ptz_control
from dual_camera_manager import DualCameraManager
from typing import Optional, Tuple, Dict, List

# 禁用OpenCV内部线程
try:
    cv2.setNumThreads(0)
except Exception:
    pass

class DualCameraTrackingSystem:
    """
    双摄像头跟踪系统主程序
    """
    
    def __init__(self, left_camera_index=2, right_camera_index=0):
        # 初始化双摄像头管理器
        self.camera_manager = DualCameraManager(
            left_camera_index=left_camera_index,
            right_camera_index=right_camera_index
        )
        
        # 云台控制参数
        self.last_ptz_angles = {"pan": None, "tilt": None}
        self.ANGLE_THRESHOLD = settings.ANGLE_THRESHOLD
        
        # 跟踪状态
        self.tracking_mode = "idle"  # idle, acquiring, tracking, lost
        self.last_ball_detection = None
        self.ball_trajectory = []
        
        # 帧统计
        self.frame_counter = 0
        self.processed_frames = 0
        
        print("双摄像头跟踪系统初始化完成")
    
    def process_ball_detection(self, camera_id: int, x: float, y: float, 
                               confidence: float, frame_id: int) -> Optional[dict]:
        """
        处理球体检测结果，返回云台控制指令
        """
        if confidence < settings.CONF_THRESHOLD:
            return None
        
        # 1. 将像素坐标转换为云台角度
        pan, tilt = self.camera_manager.pixel_to_pantilt(x, y, camera_id)
        
        if pan is None or tilt is None:
            print(f"无法转换坐标: ({x}, {y})")
            return None
        
        # 2. 检查角度是否在云台范围内
        if not self.is_pantilt_valid(pan, tilt):
            print(f"角度超出范围: Pan={pan:.1f}°, Tilt={tilt:.1f}°")
            return None
        
        # 3. 更新跟踪状态
        self.camera_manager.update_tracking_state(True)
        
        # 4. 记录检测结果
        self.last_ball_detection = {
            'camera_id': camera_id,
            'x': x,
            'y': y,
            'pan': pan,
            'tilt': tilt,
            'confidence': confidence,
            'frame_id': frame_id,
            'timestamp': time.time()
        }
        
        # 5. 添加到轨迹历史
        self.ball_trajectory.append(self.last_ball_detection)
        if len(self.ball_trajectory) > 100:  # 保留最近100个点
            self.ball_trajectory.pop(0)
        
        # 6. 检查是否需要切换活动相机
        camera_for_angle, in_overlap = self.camera_manager.get_camera_for_angle(pan)
        if camera_for_angle != -1:
            self.camera_manager.switch_active_camera(camera_for_angle)
        
        # 7. 返回控制指令
        return {
            'pan': pan,
            'tilt': tilt,
            'camera_id': camera_id,
            'in_overlap': in_overlap,
            'confidence': confidence
        }
    
    def is_pantilt_valid(self, pan: float, tilt: float) -> bool:
        """检查云台角度是否有效"""
        return (ptz_control.ANGLE_RANGE_H[0] <= pan <= ptz_control.ANGLE_RANGE_H[1] and
                ptz_control.ANGLE_RANGE_V[0] <= tilt <= ptz_control.ANGLE_RANGE_V[1])
    
    def send_ptz_command(self, pan: float, tilt: float, debug: bool = True):
        """
        发送云台控制命令，带角度变化阈值检查
        """
        # 获取上次角度
        last_pan = self.last_ptz_angles["pan"]
        last_tilt = self.last_ptz_angles["tilt"]
        
        # 判断是否需要发送命令
        need_send = False
        if last_pan is None or last_tilt is None:
            need_send = True
        else:
            pan_diff = abs(pan - last_pan)
            tilt_diff = abs(tilt - last_tilt)
            
            if debug:
                print(f"角度变化: Pan={pan_diff:.2f}°, Tilt={tilt_diff:.2f}° (阈值={self.ANGLE_THRESHOLD}°)")
            
            # 任意轴超过阈值
            need_send = pan_diff > self.ANGLE_THRESHOLD or tilt_diff > self.ANGLE_THRESHOLD
        
        # 执行发送逻辑
        if need_send:
            success = ptz_control.control_ptz_absolute(pan, tilt, debug=debug)
            if success:
                # 更新上次角度
                self.last_ptz_angles["pan"] = pan
                self.last_ptz_angles["tilt"] = tilt
                if debug:
                    print(f"✅ 发送云台命令: Pan={pan:.2f}°, Tilt={tilt:.2f}°")
            return success
        else:
            if debug:
                print(f"❌ 角度变化未超过阈值({self.ANGLE_THRESHOLD}°)，不发送命令")
            return False
    
    def handle_lost_target(self):
        """处理目标丢失的情况"""
        current_time = time.time()
        
        # 检查是否真的丢失目标
        if (self.last_ball_detection and 
            current_time - self.last_ball_detection['timestamp'] > 3.0):
            
            self.tracking_mode = "lost"
            print("目标丢失，切换到搜索模式")
            
            # 执行搜索模式
            self.execute_search_pattern()
            
            return True
        
        return False
    
    def execute_search_pattern(self):
        """执行搜索模式"""
        print("执行搜索模式...")
        
        # 简单的搜索模式：水平扫描
        search_angles = [
            (0, 0),     # 中心
            (30, 0),    # 右30°
            (-30, 0),   # 左30°
            (60, 0),    # 右60°
            (-60, 0),   # 左60°
            (0, 15),    # 上15°
            (0, -15),   # 下15°
        ]
        
        for pan, tilt in search_angles:
            # 检查角度是否在摄像头覆盖范围内
            camera_id, _ = self.camera_manager.get_camera_for_angle(pan)
            if camera_id != -1:
                self.send_ptz_command(pan, tilt, debug=False)
                time.sleep(1.0)  # 在每个位置停留1秒
    
    def visualize_dual_camera(self, left_frame, right_frame, ball_info=None):
        """可视化双摄像头画面"""
        if left_frame is None or right_frame is None:
            return None
        
        # 调整大小以适应显示
        display_height = 540
        left_display = cv2.resize(left_frame, (960, display_height))
        right_display = cv2.resize(right_frame, (960, display_height))
        
        # 在左画面添加标记
        if ball_info and ball_info['camera_id'] == 0:
            x, y = int(ball_info['x'] * 960 / left_frame.shape[1]), int(ball_info['y'] * 540 / left_frame.shape[0])
            cv2.circle(left_display, (x, y), 20, (0, 255, 0), 3)
            cv2.putText(left_display, f"Ball ({ball_info['confidence']:.2f})", 
                       (x+30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 在右画面添加标记
        if ball_info and ball_info['camera_id'] == 1:
            x, y = int(ball_info['x'] * 960 / right_frame.shape[1]), int(ball_info['y'] * 540 / right_frame.shape[0])
            cv2.circle(right_display, (x, y), 20, (0, 255, 0), 3)
            cv2.putText(right_display, f"Ball ({ball_info['confidence']:.2f})", 
                       (x+30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加状态信息
        status_text = f"Mode: {self.tracking_mode} | Camera: {self.camera_manager.active_camera}"
        cv2.putText(left_display, status_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 添加角度信息
        if self.last_ptz_angles["pan"] is not None:
            angle_text = f"Pan: {self.last_ptz_angles['pan']:.1f}°, Tilt: {self.last_ptz_angles['tilt']:.1f}°"
            cv2.putText(right_display, angle_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 并排显示
        combined = np.hstack([left_display, right_display])
        
        return combined
    
    def run(self):
        """运行双摄像头跟踪系统"""
        print("=" * 80)
        print("双摄像头跟踪系统启动")
        print("=" * 80)
        
        # 记录开始时间
        self.start_time = time.time()

        # 初始化你的现有线程
        threads.init_global_variables()
        
        # 启动抽帧线程（左摄像头）
        t_capture = threading.Thread(
            target=threads.camera_capture_thread, 
            args=(self.camera_manager.left_camera_index,), 
            daemon=True
        )
        t_detection = threading.Thread(target=threads.yolo_detection_thread, daemon=True)
        
        t_capture.start()
        t_detection.start()
        
        # 判断是否开启预测
        if settings.USE_PREDICTION_AFTER_FRAMES == 10:
            t_predict = threading.Thread(target=threads.predict_thread, daemon=True)
            t_predict.start()
        
        frame_id = 0
        last_frame_time = time.time()
        
        try:
            while threads.is_running:
                frame_id += 1
                current_time = time.time()
                
                # 1. 捕获双摄像头帧
                left_frame, right_frame = self.camera_manager.capture_frames()
                
                # 2. 从检测队列获取结果（使用你现有的检测逻辑）
                try:
                    # 这里使用你现有的检测队列
                    # 注意：这里假设检测的是左摄像头
                    detection_result = threads.result_queue.get(timeout=0.1)
                    frame_id, frame_original, frame_calib, results, first_target, all_targets = detection_result
                    
                    # 处理检测结果
                    if first_target:
                        # 提取球体坐标
                        x = first_target['center_x']
                        y = first_target['center_y']
                        confidence = first_target.get('confidence', 0.7)
                        
                        # 处理检测结果
                        control_info = self.process_ball_detection(
                            camera_id=0,  # 左摄像头
                            x=x,
                            y=y,
                            confidence=confidence,
                            frame_id=frame_id
                        )
                        
                        # 发送云台控制命令
                        if control_info:
                            self.send_ptz_command(
                                pan=control_info['pan'],
                                tilt=control_info['tilt'],
                                debug=(frame_id % 10 == 0)  # 每10帧打印一次调试信息
                            )
                    
                    # 处理目标丢失
                    else:
                        if frame_id % 30 == 0:  # 每30帧检查一次
                            self.handle_lost_target()
                
                except queue.Empty:
                    pass
                
                # 3. 可视化显示
                if left_frame is not None and right_frame is not None:
                    ball_info = self.last_ball_detection
                    display_frame = self.visualize_dual_camera(left_frame, right_frame, ball_info)
                    
                    if display_frame is not None:
                        cv2.imshow("Dual Camera Tracking", display_frame)
                
                # 4. 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户主动退出")
                    break
                elif key == ord('s'):
                    # 切换摄像头
                    new_camera = 1 if self.camera_manager.active_camera == 0 else 0
                    self.camera_manager.switch_active_camera(new_camera)
                    print(f"手动切换摄像头: {new_camera}")
                elif key == ord('c'):
                    # 显示摄像头覆盖信息
                    print(f"\n摄像头覆盖信息:")
                    print(f"  左摄像头: [{self.camera_manager.left_min:.1f}°, {self.camera_manager.left_max:.1f}°]")
                    print(f"  右摄像头: [{self.camera_manager.right_min:.1f}°, {self.camera_manager.right_max:.1f}°]")
                    print(f"  重叠区域: [{self.camera_manager.right_min:.1f}°, {self.camera_manager.left_max:.1f}°]")
                
                # 5. 性能统计
                if frame_id % 100 == 0:
                    fps = 100 / (current_time - last_frame_time)
                    last_frame_time = current_time
                    print(f"帧率: {fps:.1f} FPS | 处理帧数: {frame_id}")
        
        except KeyboardInterrupt:
            print("\n收到中断信号")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        threads.is_running = False
        time.sleep(1)
        
        # 关闭摄像头
        self.camera_manager.close()
        
        # 关闭云台串口
        ptz_control.close_ptz_serial()
        
        cv2.destroyAllWindows()
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"  总帧数: {self.frame_counter}")
        if hasattr(self, 'start_time'):
            print(f"  跟踪时间: {time.time() - self.start_time:.1f}s")
        else:
            print("  跟踪时间: 未记录")
        print(f"  轨迹点数: {len(self.ball_trajectory)}")

# 使用示例
if __name__ == "__main__":
    # 创建双摄像头跟踪系统
    tracking_system = DualCameraTrackingSystem(left_camera_index=2, right_camera_index=0 )
    
    # 运行
    tracking_system.run()