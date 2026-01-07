"""
主程序入口
"""
import os
# Limit native thread usage to avoid oversubscription when running multiple processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import cv2
import numpy as np
from collections import deque
import copy
import queue
import threading

from config import settings
from threads import thread_manage as threads
from predict import predict_utils 
from predict import ptz_control
from mapping.coordinate_converter import ResolutionConverter
    

# Disable OpenCV internal threading to avoid contention with native thread pools
try:
    cv2.setNumThreads(0)
except Exception:
    pass

def update_frame_cache(frame_id, first_target):
    """更新帧缓存逻辑"""
    # 兼容多路：使用指定摄像头的缓存（由调用者确保camera_index在上下文中）
    camera_index = threading.current_thread().name.split("-")[-1]
    # 当作为普通调用时，camera_index可能不是数字，尝试转换；失败则默认0
    try:
        camera_index = int(camera_index)
    except Exception:
        camera_index = settings.CAMERA_INDEX

    with threads.cache_locks[camera_index]:
        if first_target is not None:
            cache_item = {
                "frame_id": frame_id,
                "target_info": first_target,
                "is_real_frame": True
            }
            threads.target_frames_caches[camera_index].append(cache_item)
            print(f"✅ 第{frame_id}帧：缓存更新 | 缓存帧数={len(threads.target_frames_caches[camera_index])}/{settings.FRAME_CACHE_LEN}")
            print(f"当前缓存最新帧：帧ID={frame_id} | 中心点=({first_target['center_x']},{first_target['center_y']})")
        else:
            print(f"\r🔄🔄🔄 第{frame_id}帧：未检测到目标| 缓存帧数={len(threads.target_frames_caches[camera_index])}/{settings.FRAME_CACHE_LEN}")


def visualize_results(frame_result, first_target):
    """可视化结果显示"""
    # 标注缓存信息
    # 尝试从当前线程名解析摄像头索引，否则使用默认
    camera_index = threading.current_thread().name.split("-")[-1]
    try:
        camera_index = int(camera_index)
    except Exception:
        camera_index = settings.CAMERA_INDEX
    cv2.putText(frame_result, f"Cache: {len(threads.target_frames_caches[camera_index])}/{settings.FRAME_CACHE_LEN}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    
    # 标注第一个目标
    if first_target:
        x1, y1, x2, y2 = first_target["x1"], first_target["y1"], first_target["x2"], first_target["y2"]
        cv2.rectangle(frame_result, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.circle(frame_result, (first_target["center_x"], first_target["center_y"]), 
                   5, (0,255,0), -1)
    
    # 复用帧时标注复用的目标
    elif len(threads.target_frames_caches[camera_index]) > 0:
        latest_target = threads.target_frames_caches[camera_index][-1]["target_info"]
        x1, y1, x2, y2 = latest_target["x1"], latest_target["y1"], latest_target["x2"], latest_target["y2"]
        cv2.rectangle(frame_result, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(frame_result, "REUSE", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def camera_calib_yolov8(camera_index):
    """主程序入口"""    
    print("="*80)
    print("摄像头启动！调试模式：")
    print(f"- 缓存帧数上限：{settings.FRAME_CACHE_LEN}")
    print(f"- 置信度阈值：{settings.CONF_THRESHOLD}")
    print("="*80)

    # 初始化摄像头专属全局变量
    threads.init_global_variables(camera_index)

    # 初始化坐标转换器
    # coord_converter = ResolutionConverter(camera_index)

    # 初始化坐标映射器
    coordinate_mapper = None
    try:
        print(f"➡️ 准备导入 mapping.coordinate_mapping（camera_index={camera_index}）")
        from mapping.coordinate_mapping import CameraCoordinateMapper
        print("➡️ 导入 mapping.coordinate_mapping 成功，准备实例化 CameraCoordinateMapper")
        coordinate_mapper = CameraCoordinateMapper(camera_index)
        print(f"✅ 坐标映射器初始化成功：摄像头索引={camera_index}")
    except Exception as e:
        print(f"⚠️ 坐标映射器初始化失败：{e}")
        coordinate_mapper = None

    last_ptz_angles = {"pan": None, "tilt": None}
    ANGLE_THRESHOLD = settings.ANGLE_THRESHOLD  # 可配置的
    
    # 启动抽帧/检测/预测线程
    t_capture = threading.Thread(target=threads.camera_capture_thread, args=(camera_index,), name=f"capture-{camera_index}", daemon=True)
    t_detection = threading.Thread(target=threads.yolo_detection_thread, args=(camera_index,), name=f"detect-{camera_index}", daemon=True)
    t_predict = threading.Thread(target=threads.predict_thread, args=(camera_index,), name=f"predict-{camera_index}", daemon=True)
    t_capture.start()
    t_detection.start()

    # 判断是否开启预测
    if settings.USE_PREDICTION_AFTER_FRAMES == 10:
        t_predict.start()

    # 仅在单路模式下创建单路显示窗；双路时使用主线程合成窗显示，避免启动多个黑窗
    cam2_cfg = getattr(settings, 'CAMERA_INDEX_2', None)
    single_display = cam2_cfg is None
    window_name = f"Detection Result (Calibrated)-{camera_index}"
    if single_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass

    frame_id = 0
    count_num = -9 # 前9帧为检测模型检测结果 非预测结果 不计入数据中

    # ========== 核心修改1：替换原有标记，新增双首次标记（<10帧/≥10帧） ==========
    # 标记1：<10帧时，仅首次取最新缓存结果执行云台控制（永久仅一次）
    is_first_cache_control = True
    # 标记2：≥10帧时，仅首次接口结果执行云台控制（永久仅一次）
    is_first_api_control = True
    # 状态锁（保证多线程读写标记安全）
    control_lock = threading.Lock()

    # <10帧专用：记录上一次执行控制的最新帧ID（用于对比数据是否变化）
    last_cache_frame_id = None
    # ≥10帧专用：记录上一次执行控制的缓存帧ID列表（用于对比数据是否变化）
    last_api_cache_frame_ids = None

    while threads.is_running:
        frame_id += 1
        
        try:
            # 从指定摄像头的结果队列取数据
            frame_id, frame_original, frame_calib, results, first_target, all_targets = threads.result_queues[camera_index].get(timeout=1.0)
        except queue.Empty:
            continue

        # 更新缓存
        update_frame_cache(frame_id, first_target)

        # ========== 核心修改2：先处理<10帧场景（仅首次取最新结果执行控制） ==========
        with threads.cache_locks[camera_index]:  # 加锁读取缓存（threads里的cache_lock）
            cache_len = len(threads.target_frames_caches[camera_index]) if threads.target_frames_caches[camera_index] else 0
            # 取缓存最新结果（仅<10帧时用）
            latest_cache_item = None
            if threads.target_frames_caches[camera_index] and cache_len > 0:
                latest_cache_item = threads.target_frames_caches[camera_index][-1]  # deque最后一个是最新

        # <10帧：首次/数据变化执行云台控制，数据不变跳过
        if cache_len < settings.USE_PREDICTION_AFTER_FRAMES and latest_cache_item is not None:
            # 提取当前最新帧ID（用于对比）
            current_cache_frame_id = latest_cache_item['frame_id']
            
            with control_lock:
                # 场景1：首次执行<10帧的云台控制
                if is_first_cache_control:
                    count_num += 1
                    # 提取最新缓存结果的核心信息
                    target_info = latest_cache_item['target_info']
                    x1, y1 = target_info['x1'], target_info['y1']
                    w, h = target_info['width'], target_info['height']

                    # 打印日志
                    print(f"\n🔴 【首次】缓存<10帧（当前{cache_len}帧）- 最新结果：")
                    print(f"📌 主线程（第{frame_id}帧）：缓存最新帧ID={current_cache_frame_id}")
                    print(f"🔮🔮🔮第{count_num}个控制结果：坐标=({x1}, {y1})，宽高=({w}, {h})")
                    print(f"✂️ 裁剪偏移：x={settings.CALIB_OFFSET_X}, y={settings.CALIB_OFFSET_Y}")
                    print(f"🌿 偏移后坐标=({x1+settings.CALIB_OFFSET_X}, {y1+settings.CALIB_OFFSET_Y})")

                    # ========== 执行云台控制（替换为你的实际代码） ========== 这里需要重新计算
                    # 获取目标的像素坐标（图像中心）
                    center_x = target_info['center_x']
                    center_y = target_info['center_y']
                    # 使用坐标映射器计算云台角度
                    print(f"\n🎯 摄像头{camera_index}：检测到目标")
                    print(f"像素坐标：({center_x}, {center_y})")
                    
                    # 左摄像头拟合方程：pan = 0.026689 × u' + -73.58
                    # 右摄像头拟合方程：pan = 0.023512 × u' + -6.37

                    Pan = None
                    if camera_index == 2:
                        Pan = 0.026689 * center_x - 73.58
                    else:
                        Pan = 0.023512 * center_x - 6.37
                    Tilt = -30
                    print(f"计算云台角度：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")

                    # 获取上次角度
                    last_pan = last_ptz_angles["pan"]
                    last_tilt = last_ptz_angles["tilt"]

                    # 判断是否需要发送命令
                    need_send = False
                    if last_pan is None or last_tilt is None:
                        need_send = True
                    else:
                        pan_diff = abs(Pan - last_pan)
                        tilt_diff = abs(Tilt - last_tilt)
                        print(f"角度变化：Pan={pan_diff:.2f}°, Tilt={tilt_diff:.2f}° (阈值={ANGLE_THRESHOLD}°)")
                        # 可选择：任意轴超过阈值 或 两轴综合超过阈值
                        # 方式1：任意轴超过阈值
                        need_send = pan_diff > ANGLE_THRESHOLD or tilt_diff > ANGLE_THRESHOLD
                        # 方式2：两轴综合超过阈值（可选）
                        # import math
                        # total_diff = math.hypot(pan_diff, tilt_diff)
                        # need_send = total_diff > ANGLE_THRESHOLD
                    
                    # 执行发送逻辑
                    if need_send:
                        ptz_control.control_ptz_absolute(Pan, Tilt)
                        # 更新上次角度
                        last_ptz_angles["pan"] = Pan
                        last_ptz_angles["tilt"] = Tilt
                        print(f"✅ 发送云台命令：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                    else:
                        print(f"❌ 角度变化未超过阈值({ANGLE_THRESHOLD}°)，不发送命令")

                    # 计算足球距离与建议变焦等级
                    # football_d =  ptz_control.football_pixel2distance(w)
                    # print(f"足球宽度={w}像素，距离约为{football_d:.2f}米")
                    # football_zoom = ptz_control.distance2zoom(football_d)
                    # print(f"建议变焦等级={football_zoom}X")
                    # ptz_control.control_ptz_zoom(football_zoom)

                    # 标记为非首次，记录本次帧ID（用于后续对比）
                    is_first_cache_control = False
                    last_cache_frame_id = current_cache_frame_id
                    print(f"✅ <10帧首次云台控制执行完成，记录最新帧ID：{current_cache_frame_id}")
                
                # 场景2：非首次 → 对比最新帧ID是否变化（数据是否更新）
                else:
                    if current_cache_frame_id != last_cache_frame_id:
                        count_num += 1
                        # 提取最新缓存结果的核心信息
                        target_info = latest_cache_item['target_info']
                        x1, y1 = target_info['x1'], target_info['y1']
                        w, h = target_info['width'], target_info['height']

                        # 打印日志
                        print(f"\n🟠 【更新】缓存<10帧（当前{cache_len}帧）- 数据变化（旧ID：{last_cache_frame_id} → 新ID：{current_cache_frame_id}）：")
                        print(f"📌 主线程（第{frame_id}帧）：缓存最新帧ID={current_cache_frame_id}")
                        print(f"🔮🔮🔮第{count_num}个控制结果：坐标=({x1}, {y1})，宽高=({w}, {h})")
                        print(f"✂️ 裁剪偏移：x={settings.CALIB_OFFSET_X}, y={settings.CALIB_OFFSET_Y}")
                        print(f"🌿 偏移后坐标=({x1+settings.CALIB_OFFSET_X}, {y1+settings.CALIB_OFFSET_Y})")

                        # ========== 执行云台控制（替换为你的实际代码） ========== 这里需要重新计算
                        # 获取目标的像素坐标（图像中心）
                        center_x = target_info['center_x']
                        center_y = target_info['center_y']
                        # 使用坐标映射器计算云台角度
                        print(f"\n🎯 摄像头{camera_index}：检测到目标")
                        print(f"像素坐标：({center_x}, {center_y})")

                        # 左摄像头拟合方程：pan = 0.026689 × u' + -73.58
                        # 右摄像头拟合方程：pan = 0.023512 × u' + -6.37

                        Pan = None
                        if camera_index == 2:
                            Pan = 0.026689 * center_x - 73.58
                        else:
                            Pan = 0.023512 * center_x - 6.37
                        Tilt = -30
                        print(f"计算云台角度：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")

                        # 获取上次角度
                        last_pan = last_ptz_angles["pan"]
                        last_tilt = last_ptz_angles["tilt"]

                        # 判断是否需要发送命令
                        need_send = False
                        if last_pan is None or last_tilt is None:
                            need_send = True
                        else:
                            pan_diff = abs(Pan - last_pan)
                            tilt_diff = abs(Tilt - last_tilt)
                            print(f"角度变化：Pan={pan_diff:.2f}°, Tilt={tilt_diff:.2f}° (阈值={ANGLE_THRESHOLD}°)")
                            # 可选择：任意轴超过阈值 或 两轴综合超过阈值
                            # 方式1：任意轴超过阈值
                            need_send = pan_diff > ANGLE_THRESHOLD or tilt_diff > ANGLE_THRESHOLD
                            # 方式2：两轴综合超过阈值（可选）
                            # import math
                            # total_diff = math.hypot(pan_diff, tilt_diff)
                            # need_send = total_diff > ANGLE_THRESHOLD
                        
                        # 执行发送逻辑
                        if need_send:
                            ptz_control.control_ptz_absolute(Pan, Tilt)
                            # 更新上次角度
                            last_ptz_angles["pan"] = Pan
                            last_ptz_angles["tilt"] = Tilt
                            print(f"✅ 发送云台命令：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                        else:
                            print(f"❌ 角度变化未超过阈值({ANGLE_THRESHOLD}°)，不发送命令")

                        # 计算足球距离与建议变焦等级
                        # football_d = ptz_control.football_pixel2distance(w) 
                        # print(f"足球宽度={w}像素，距离约为{football_d:.2f}米")
                        # football_zoom = ptz_control.distance2zoom(football_d)
                        # print(f"建议变焦等级={football_zoom}X")
                        # ptz_control.control_ptz_zoom(football_zoom)   

                        # 更新历史帧ID为当前值
                        last_cache_frame_id = current_cache_frame_id
                        print(f"✅ <10帧数据更新，云台控制执行完成")
                    else:
                        # 数据未变化 → 跳过控制
                        print(f"\n📌 主线程（第{frame_id}帧）：缓存<10帧（{cache_len}帧），数据未变化（帧ID：{current_cache_frame_id}），跳过云台控制")
        # 处理完<10帧场景，继续循环
        pass

        # ========== 核心修改3：处理≥10帧场景（首次/数据变化执行控制，数据不变跳过） ==========
        # 主线程获取最新预测结果（≥10帧时才有值）
        # 取当前摄像头的预测结果
        last_predict = threads.last_predict_results.get(camera_index, None)
        if last_predict is not None and cache_len >= settings.USE_PREDICTION_AFTER_FRAMES:
            # 云台控制逻辑
            third_frame = predict_utils.get_third_future_frame(last_predict)  
            
            if third_frame:
                # 加锁读取当前缓存的帧ID列表（用于对比是否变化）
                with threads.cache_locks[camera_index]:
                    current_cache_frame_ids = [item['frame_id'] for item in threads.target_frames_caches[camera_index]] if threads.target_frames_caches[camera_index] else []
                
                with control_lock:
                    # 场景1：首次执行≥10帧的云台控制
                    if is_first_api_control:
                        count_num += 1
                        print(f"\n🟢 【首次】缓存≥10帧 - 接口预测结果：")
                        print(f"🔮🔮🔮第{count_num}个预测点（首次）：检测值={third_frame}")
                        print(f"✂️ 裁剪偏移：x={settings.CALIB_OFFSET_X}, y={settings.CALIB_OFFSET_Y}")
                        print(f"🌿 偏移后坐标=({third_frame['x1']+settings.CALIB_OFFSET_X}, {third_frame['y1']+settings.CALIB_OFFSET_Y})")

                        # ========== 执行云台控制（替换为你的实际代码） ========== 这里需要重新计算
                        # 获取目标的像素坐标（图像中心）
                        center_x = third_frame['x1']
                        center_y = third_frame['y1']
                        # 使用坐标映射器计算云台角度
                        print(f"\n🎯 摄像头{camera_index}：检测到目标")
                        print(f"像素坐标：({center_x}, {center_y})")

                        # 左摄像头拟合方程：pan = 0.026689 × u' + -73.58
                        # 右摄像头拟合方程：pan = 0.023512 × u' + -6.37

                        Pan = None
                        if camera_index == 2:
                            Pan = 0.026689 * center_x - 73.58
                        else:
                            Pan = 0.023512 * center_x - 6.37
                        Tilt = -30
                        print(f"计算云台角度：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                        
                        # 获取上次角度
                        last_pan = last_ptz_angles["pan"]
                        last_tilt = last_ptz_angles["tilt"]

                        # 判断是否需要发送命令
                        need_send = False
                        if last_pan is None or last_tilt is None:
                            need_send = True
                        else:
                            pan_diff = abs(Pan - last_pan)
                            tilt_diff = abs(Tilt - last_tilt)
                            print(f"角度变化：Pan={pan_diff:.2f}°, Tilt={tilt_diff:.2f}° (阈值={ANGLE_THRESHOLD}°)")
                            # 可选择：任意轴超过阈值 或 两轴综合超过阈值
                            # 方式1：任意轴超过阈值
                            need_send = pan_diff > ANGLE_THRESHOLD or tilt_diff > ANGLE_THRESHOLD
                            # 方式2：两轴综合超过阈值（可选）
                            # import math
                            # total_diff = math.hypot(pan_diff, tilt_diff)
                            # need_send = total_diff > ANGLE_THRESHOLD
                        
                        # 执行发送逻辑
                        if need_send:
                            ptz_control.control_ptz_absolute(Pan, Tilt)
                            # 更新上次角度
                            last_ptz_angles["pan"] = Pan
                            last_ptz_angles["tilt"] = Tilt
                            print(f"✅ 发送云台命令：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                        else:
                            print(f"❌ 角度变化未超过阈值({ANGLE_THRESHOLD}°)，不发送命令")

                        # 计算足球距离与建议变焦等级
                        # football_d =  ptz_control.football_pixel2distance(w) 
                        # print(f"足球宽度={w}像素，距离约为{football_d:.2f}米")
                        # football_zoom = ptz_control.distance2zoom(football_d)
                        # print(f"建议变焦等级={football_zoom}X")
                        # ptz_control.control_ptz_zoom(football_zoom)   

                        # 标记为非首次，记录本次缓存帧ID（用于后续对比）
                        is_first_api_control = False
                        last_api_cache_frame_ids = current_cache_frame_ids  # 新增：记录首次的帧ID
                        print(f"✅ ≥10帧首次云台控制执行完成，记录缓存帧ID：{last_api_cache_frame_ids}")
                    
                    # 场景2：非首次 → 对比缓存数据是否变化
                    else:
                        # 核心：对比当前缓存帧ID与上一次执行时的帧ID
                        if current_cache_frame_ids != last_api_cache_frame_ids:
                            count_num += 1
                            print(f"\n🟡 【更新】缓存≥10帧 - 数据变化（旧ID：{last_api_cache_frame_ids} → 新ID：{current_cache_frame_ids}）：")
                            print(f"🔮🔮🔮第{count_num}个预测点（更新）：检测值={third_frame}")
                            print(f"✂️ 裁剪偏移：x={settings.CALIB_OFFSET_X}, y={settings.CALIB_OFFSET_Y}")
                            print(f"🌿 偏移后坐标=({third_frame['x1']+settings.CALIB_OFFSET_X}, {third_frame['y1']+settings.CALIB_OFFSET_Y})")

                            # ========== 执行云台控制（替换为你的实际代码） ==========
                             # ========== 执行云台控制（替换为你的实际代码） ========== 这里需要重新计算
                            # 获取目标的像素坐标（图像中心）
                            center_x = third_frame['x1']
                            center_y = third_frame['y1']
                            # 使用坐标映射器计算云台角度
                            print(f"\n🎯 摄像头{camera_index}：检测到目标")
                            print(f"像素坐标：({center_x}, {center_y})")

                            # 左摄像头拟合方程：pan = 0.026689 × u' + -73.58
                            # 右摄像头拟合方程：pan = 0.023512 × u' + -6.37

                            Pan = None
                            if camera_index == 2:
                                Pan = 0.026689 * center_x - 73.58
                            else:
                                Pan = 0.023512 * center_x - 6.37
                            Tilt = -30
                            print(f"计算云台角度：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                            

                            # 获取上次角度
                            last_pan = last_ptz_angles["pan"]
                            last_tilt = last_ptz_angles["tilt"]

                            # 判断是否需要发送命令
                            need_send = False
                            if last_pan is None or last_tilt is None:
                                need_send = True
                            else:
                                pan_diff = abs(Pan - last_pan)
                                tilt_diff = abs(Tilt - last_tilt)
                                print(f"角度变化：Pan={pan_diff:.2f}°, Tilt={tilt_diff:.2f}° (阈值={ANGLE_THRESHOLD}°)")
                                # 可选择：任意轴超过阈值 或 两轴综合超过阈值
                                # 方式1：任意轴超过阈值
                                need_send = pan_diff > ANGLE_THRESHOLD or tilt_diff > ANGLE_THRESHOLD
                                # 方式2：两轴综合超过阈值（可选）
                                # import math
                                # total_diff = math.hypot(pan_diff, tilt_diff)
                                # need_send = total_diff > ANGLE_THRESHOLD
                            
                            # 执行发送逻辑
                            if need_send:
                                ptz_control.control_ptz_absolute(Pan, Tilt)
                                # 更新上次角度
                                last_ptz_angles["pan"] = Pan
                                last_ptz_angles["tilt"] = Tilt
                                print(f"✅ 发送云台命令：Pan={Pan:.2f}°, Tilt={Tilt:.2f}°")
                            else:
                                print(f"❌ 角度变化未超过阈值({ANGLE_THRESHOLD}°)，不发送命令")

                            # 计算足球距离与建议变焦等级
                            # football_d =  ptz_control.football_pixel2distance(w) 
                            # print(f"足球宽度={w}像素，距离约为{football_d:.2f}米")
                            # football_zoom = ptz_control.distance2zoom(football_d)
                            # print(f"建议变焦等级={football_zoom}X")
                            # ptz_control.control_ptz_zoom(football_zoom)

                            # 更新历史帧ID为当前值
                            last_api_cache_frame_ids = current_cache_frame_ids
                            print(f"✅ ≥10帧数据更新，云台控制执行完成")
                        else:
                            # 数据未变化 → 跳过控制
                            print(f"\n📌 主线程（第{frame_id}帧）：≥10帧数据未变化（帧ID：{current_cache_frame_ids}），跳过云台控制")
            else:
                print(f"\n📌 主线程（第{frame_id}帧）：无有效third_frame，跳过云台控制")
        else:
            # 无预测结果（缓存<10或≥10但未请求）
            if cache_len >= 10:  # 仅≥10帧时打印提示，<10帧已在上文处理
                print(f"\n📌 主线程（第{frame_id}帧）：暂无预测结果（缓存≥10帧但接口未返回）")


        # 可视化调试（保留原有逻辑）
        frame_result = results[0].plot()
        visualize_results(frame_result, first_target)

        # 显示窗口：根据 settings.DISPLAY_SCALE 缩放后显示（适配高分辨率屏幕）
        display_frame = frame_result
        try:
            scale = getattr(settings, 'DISPLAY_SCALE', 1.0)
            if scale and scale > 0 and scale != 1.0:
                h, w = frame_result.shape[:2]
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                display_frame = cv2.resize(frame_result, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            display_frame = frame_result

        # 如果配置了双路，则不直接显示，而是写入共享 display_frames，由主线程合成显示
        cam2 = getattr(settings, 'CAMERA_INDEX_2', None)
        if cam2 is not None:
            try:
                with threads.display_locks[camera_index]:
                    # store a copy to avoid race conditions
                    threads.display_frames[camera_index] = display_frame.copy()
                    threads.last_display_info[camera_index] = (frame_id, first_target)
            except Exception:
                pass
        else:
            cv2.imshow(window_name, display_frame)
        # cv2.imshow("Original Frame", frame_original)

        # 按键操作（保留原有逻辑）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✅ 用户主动退出")
            threads.is_running = False
            break
        elif key == ord('c'):
            # 查看缓存详情
            print("\n\n===== 缓存详情 =====")
            cache = threads.target_frames_caches[camera_index]
            if len(cache) == 0:
                print("缓存为空")
            else:
                for i, item in enumerate(cache):
                    t = item.get("target_info", None)
                    if t is None:
                        print(f"缓存{i+1}：帧ID={item['frame_id']} | 无有效目标")
                    else:
                        frame_type = "真实帧" if item.get("is_real_frame", True) else "复用帧"
                        print(f"缓存{i+1}：帧ID={item['frame_id']} | {frame_type} | 类别={t['cls_name']} | 中心点=({t['center_x']},{t['center_y']})")
            print("="*50)
        elif key == ord('s'):
            # 强制添加当前帧到缓存
            if first_target is not None:
                cache_item = {"frame_id": frame_id, "target_info": first_target, "is_real_frame": True}
                threads.target_frames_caches[camera_index].append(cache_item)
                print(f"\n🔧 强制添加缓存：当前帧数={len(threads.target_frames_caches[camera_index])}")

    # 等待线程退出（保留原有逻辑）
    threads.is_running = False
    t_capture.join(timeout=settings.THREAD_JOIN_TIMEOUT)
    t_detection.join(timeout=settings.THREAD_JOIN_TIMEOUT)

    # 判断是否开启预测
    if settings.USE_PREDICTION_AFTER_FRAMES == 10:
        t_predict.join(timeout=settings.THREAD_JOIN_TIMEOUT)

    # 释放资源（保留原有逻辑）
    cv2.destroyAllWindows()
    
    # 最终统计（保留原有逻辑）
    print("\n===== 退出统计 =====")
    print(f"总处理帧数：{frame_id}")
    cache_len_final = len(threads.target_frames_caches[camera_index])
    print(f"缓存帧数：{cache_len_final}")
    if cache_len_final > 0:
        first_cache = threads.target_frames_caches[camera_index][0]
        last_cache = threads.target_frames_caches[camera_index][-1]
        print(f"缓存帧范围：{first_cache['frame_id']} ~ {last_cache['frame_id']}")

if __name__ == "__main__":
    # 支持单路或双路启动：如果配置了 CAMERA_INDEX_2 则启动双路
    try:
        cam1 = settings.CAMERA_INDEX
        cam2 = getattr(settings, 'CAMERA_INDEX_2', None)
    except Exception:
        cam1 = 0
        cam2 = None

    if cam2 is None:
        camera_calib_yolov8(camera_index=cam1)
    else:
        t1 = threading.Thread(target=camera_calib_yolov8, args=(cam1,), name=f"main-{cam1}")
        t2 = threading.Thread(target=camera_calib_yolov8, args=(cam2,), name=f"main-{cam2}")
        t1.start()
        t2.start()

        # 合成显示主循环：在主线程中将两路画面拼接显示
        combined_name = f"Combined-{cam1}-{cam2}"
        try:
            cv2.namedWindow(combined_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass

        while threads.is_running:
            try:
                left = None
                right = None
                with threads.display_locks[cam1]:
                    left = threads.display_frames.get(cam1, None)
                with threads.display_locks[cam2]:
                    right = threads.display_frames.get(cam2, None)

                # 如果都为空，短暂等待
                if left is None and right is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        threads.is_running = False
                        break
                    continue

                # 任何一边为空，用黑帧填充
                if left is None and right is not None:
                    h, w = right.shape[:2]
                    left = np.zeros((h, w, 3), dtype=right.dtype)
                if right is None and left is not None:
                    h, w = left.shape[:2]
                    right = np.zeros((h, w, 3), dtype=left.dtype)

                # 保证高度一致，按高度调整宽度
                if left.shape[0] != right.shape[0]:
                    target_h = max(left.shape[0], right.shape[0])
                    def resize_to_h(img, target_h):
                        h, w = img.shape[:2]
                        new_w = max(1, int(w * (target_h / h)))
                        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    left = resize_to_h(left, target_h)
                    right = resize_to_h(right, target_h)

                # 支持左右互换配置
                try:
                    swap = getattr(settings, 'COMBINE_SWAP', False)
                except Exception:
                    swap = False
                if swap:
                    combined = np.hstack((right, left))
                else:
                    combined = np.hstack((left, right))

                # 额外按DISPLAY_SCALE缩放整张合成图（适配高分辨率屏幕）
                try:
                    scale = getattr(settings, 'DISPLAY_SCALE', 1.0)
                    if scale and scale > 0 and scale != 1.0:
                        h, w = combined.shape[:2]
                        combined = cv2.resize(combined, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

                cv2.imshow(combined_name, combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    threads.is_running = False
                    break
                elif key == ord('c'):
                    print("\n\n===== 双路缓存详情 =====")
                    for cam in (cam1, cam2):
                        with threads.cache_locks[cam]:
                            cache = threads.target_frames_caches[cam]
                            print(f"-- Camera {cam} 缓存 {len(cache)} 帧 --")
                            for i, item in enumerate(cache):
                                t = item.get('target_info', None)
                                if t is None:
                                    print(f"缓存{i+1}：帧ID={item['frame_id']} | 无有效目标")
                                else:
                                    frame_type = '真实帧' if item.get('is_real_frame', True) else '复用帧'
                                    print(f"缓存{i+1}：帧ID={item['frame_id']} | {frame_type} | 类别={t['cls_name']} | 中心点=({t['center_x']},{t['center_y']})")
                    print("="*50)
                elif key == ord('s'):
                    # 强制添加两路当前显示的第一个目标到各自缓存
                    for cam in (cam1, cam2):
                        info = threads.last_display_info.get(cam, None)
                        if info and info[1] is not None:
                            with threads.cache_locks[cam]:
                                threads.target_frames_caches[cam].append({"frame_id": info[0], "target_info": info[1], "is_real_frame": True})
                                print(f"强制添加 Camera {cam} 帧ID {info[0]} 到缓存 (当前帧数={len(threads.target_frames_caches[cam])})")

            except Exception as e:
                print(f"合成显示循环异常：{e}")
                break

        # 退出：等待子线程结束
        threads.is_running = False
        t1.join(timeout=settings.THREAD_JOIN_TIMEOUT)
        t2.join(timeout=settings.THREAD_JOIN_TIMEOUT)
        cv2.destroyAllWindows()