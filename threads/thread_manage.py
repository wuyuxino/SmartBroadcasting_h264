"""
线程管理 - 抽帧/检测/预测线程
"""
import threading
import queue
import time
import cv2
from ultralytics import YOLO
import copy
import numpy as np

from config import settings
# 去除矫正与裁剪依赖（不再使用 calib_utils）
from detection import yolo_utils
from predict import predict_utils
from collections import deque

# 自定义最新帧队列
class LatestFrameQueue(queue.Queue):
    def put(self, item, block=True, timeout=None):
        with self.mutex:
            while self._qsize() >= self.maxsize:
                self._get()  # 剔除最旧帧
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

# 自定义最新结果队列
class LatestResultQueue(queue.Queue):
    def put(self, item, block=True, timeout=None):
        with self.mutex:
            while self._qsize() >= self.maxsize:
                self._get()  # 剔除最旧结果
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

# 多路支持：按摄像头索引维护队列、缓存和锁
frame_queues = {}        # camera_index -> LatestFrameQueue
result_queues = {}       # camera_index -> LatestResultQueue
target_frames_caches = {}# camera_index -> deque
cache_locks = {}         # camera_index -> RLock
last_predict_results = {}# camera_index -> last predict result
is_running = True

def init_camera_globals(camera_index):
    """初始化/确保指定摄像头的全局数据结构存在"""
    if camera_index not in frame_queues:
        frame_queues[camera_index] = LatestFrameQueue(maxsize=2)
    if camera_index not in result_queues:
        result_queues[camera_index] = LatestResultQueue(maxsize=2)
    if camera_index not in target_frames_caches:
        target_frames_caches[camera_index] = deque(maxlen=settings.FRAME_CACHE_LEN)
    if camera_index not in cache_locks:
        cache_locks[camera_index] = threading.RLock()
    if camera_index not in last_predict_results:
        last_predict_results[camera_index] = None
    # 显示合成相关结构
    if camera_index not in globals().get('display_frames', {}):
        # store latest frame for display composition
        globals().setdefault('display_frames', {})[camera_index] = None
    if camera_index not in globals().get('display_locks', {}):
        globals().setdefault('display_locks', {})[camera_index] = threading.Lock()
    # last display info (frame_id, first_target)
    if camera_index not in globals().get('last_display_info', {}):
        globals().setdefault('last_display_info', {})[camera_index] = None
    print(f"✅ 初始化摄像头 {camera_index} 全局对象：缓存 {len(target_frames_caches[camera_index])}/{settings.FRAME_CACHE_LEN}")

def init_global_variables(camera_index=settings.CAMERA_INDEX):
    """兼容接口：初始化指定摄像头的全局变量（由主线程调用）"""
    init_camera_globals(camera_index)

def camera_capture_thread(camera_index=settings.CAMERA_INDEX):
    """抽帧线程：负责读取摄像头帧（含抽帧耗时统计）"""
    global is_running
    
    # ========== 1. 初始化统计变量 ==========
    read_cost_list = []  # 存储最近100帧的抽帧耗时（避免内存溢出）
    max_read_cost = 0.0  # 最近100帧的最高抽帧耗时（ms）
    min_read_cost = float('inf')  # 最近100帧的最低抽帧耗时（ms）
    avg_read_cost = 0.0  # 最近100帧的平均抽帧耗时（ms）
    max_history = 100  # 最多保留最近100帧的耗时数据
    print_interval = 100  # 每100帧打印一次（含统计信息）
    
    # 新增：全局统计变量
    global_total_valid_frames = 0  # 统计【总有效抽帧数】（丢弃前20帧后的所有有效帧）
    global_total_read_cost = 0.0   # 统计【总抽帧耗时】（丢弃前20帧后的耗时总和）
    global_max_read_cost = 0.0     # 全局最高抽帧耗时
    global_min_read_cost = float('inf')  # 全局最低抽帧耗时
    global_avg_read_cost = 0.0     # 全局平均抽帧耗时
    
    # 新增：跳过前N帧的统计
    skip_initial_frames = 20  # 跳过前20帧的统计（用于避免启动时的异常高延迟）
    skipped_frames = 0  # 已跳过的帧数

    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.IMAGE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # 可选：强制设置摄像头输出为 MJPG（可显著降低CPU端解码开销），并打印实际返回的 FOURCC 便于诊断
    try:
        if getattr(settings, 'FORCE_CAPTURE_MJPG', False):
            mjpg = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            # 读取回来的 FOURCC 可能以整数形式返回，转换为可读字符串
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            ch0 = chr(fourcc & 0xFF)
            ch1 = chr((fourcc >> 8) & 0xFF)
            ch2 = chr((fourcc >> 16) & 0xFF)
            ch3 = chr((fourcc >> 24) & 0xFF)
            fc_str = f"{ch0}{ch1}{ch2}{ch3}"
            print(f"🔧 请求设置 MJPG，摄像头 {camera_index} 实际 FOURCC: {fc_str} ({fourcc})")
    except Exception as e:
        print(f"⚠️ 强制设置 MJPG 时出错：{e}")

    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_index}")
        is_running = False
        return

    # 不再进行矫正/裁剪，直接使用原始帧作为校正后帧

    frame_id = 0
    while is_running:
        # ========== 时间节点1：开始抽帧 ==========
        start_read = time.time()   # 抽帧开始时间

        # 核心优化1：清空缓冲区，只读最新帧
        grab_count = 0
        while cap.grab():  # 读帧头，清空旧帧
            grab_count += 1
            if grab_count > 2:  # 最多清2帧，避免无限循环
                break
        ret, frame = cap.retrieve()  # 读最新帧
        frame_id += 1
        
        # ========== 时间节点2：抽帧完成 ==========
        read_cost = (time.time() - start_read) * 1000  # 转毫秒
        
        # ========== 2. 抽帧耗时统计（仅统计有效帧） ==========
        if ret and frame is not None:
            # 检查是否需要跳过此帧的统计
            should_skip = skipped_frames < skip_initial_frames
            
            if not should_skip:
                # ===== 全局统计 =====
                global_total_valid_frames += 1
                global_total_read_cost += read_cost
                
                # 更新全局最高/最低耗时
                if read_cost > global_max_read_cost:
                    global_max_read_cost = read_cost
                if read_cost < global_min_read_cost:
                    global_min_read_cost = read_cost
                    
                # 计算全局平均耗时
                if global_total_valid_frames > 0:
                    global_avg_read_cost = global_total_read_cost / global_total_valid_frames
                
                # ===== 最近100帧统计 =====
                # 将当前耗时加入列表（仅保留最近100帧）
                read_cost_list.append(read_cost)
                if len(read_cost_list) > max_history:
                    read_cost_list.pop(0)
                # 计算最近100帧的统计值
                if len(read_cost_list) > 0:
                    max_read_cost = max(read_cost_list)
                    min_read_cost = min(read_cost_list)
                    avg_read_cost = sum(read_cost_list) / len(read_cost_list)
            else:
                # 跳过此帧统计
                skipped_frames += 1
                # 打印跳过信息（可选）
                if skipped_frames <= 5:  # 只打印前5次跳过信息，避免刷屏
                    print(f"🔄 跳过前{skip_initial_frames}帧统计中的第{skipped_frames}帧（耗时{read_cost:.2f}ms）")
                elif skipped_frames == skip_initial_frames:
                    print(f"✅ 已完成前{skip_initial_frames}帧跳过，开始正式统计...")
        
        if not ret:
            print(f"❌ 第{frame_id}帧：无法读取摄像头帧，重试...")
            retry_count = 0
            while retry_count < 3 and not ret:
                ret, frame = cap.read()
                retry_count += 1
            if not ret:
                print("❌ 摄像头读取失败，退出程序")
                is_running = False
                break

        # ========== 时间节点3：不再进行矫正或裁剪，直接使用原始帧 ==========
        start_process = time.time()
        frame_calib = frame

        # ========== 3. 打印耗时+统计信息（每10帧） ==========
        # 只有当跳过阶段已完成且有一定统计数据时才打印
        if (skipped_frames >= skip_initial_frames and 
            frame_id % print_interval == 0 and 
            len(read_cost_list) > 0):
            print(f"""
🔍 帧统计 | 帧ID：{frame_id} | 线程：{threading.current_thread().name}
├─ 最近{len(read_cost_list)}帧抽帧耗时：
│  ├─ 当前帧：{read_cost:.2f}ms
│  ├─ 最高帧：{max_read_cost:.2f}ms
│  ├─ 最低帧：{min_read_cost:.2f}ms
│  └─ 平均帧：{avg_read_cost:.2f}ms
├─ 全局抽帧耗时（共{global_total_valid_frames}帧，已跳过前{skip_initial_frames}帧）：
│  ├─ 最高帧：{global_max_read_cost:.2f}ms
│  ├─ 最低帧：{global_min_read_cost:.2f}ms
│  └─ 平均帧：{global_avg_read_cost:.2f}ms
└─ 尺寸：{frame_calib.shape[1]}×{frame_calib.shape[0]}
            """)

        # 放入自定义队列（自动删旧存新，无满队列问题）
        # 注意：所有帧（包括跳过的帧）都放入队列供后续处理
        frame_queues[camera_index].put((frame_id, frame, frame_calib))

    # ========== 4. 线程退出时打印最终统计 ==========
    print("\n==================== 抽帧线程退出 | 最终统计 ====================")
    if global_total_valid_frames > 0:
        print(f"统计信息（已跳过前{skip_initial_frames}帧启动延迟）：")
        print(f"总有效抽帧数：{global_total_valid_frames}帧")
        print(f"全局抽帧耗时统计：")
        print(f"  - 最高耗时：{global_max_read_cost:.2f}ms")
        print(f"  - 最低耗时：{global_min_read_cost:.2f}ms")
        print(f"  - 平均耗时：{global_avg_read_cost:.2f}ms")
        print(f"  - 总耗时：{global_total_read_cost:.2f}ms")
        
        # 可选：显示最近100帧的统计（作为性能参考）
        if len(read_cost_list) > 0:
            print(f"\n最近{len(read_cost_list)}帧参考统计：")
            print(f"  - 最高耗时：{max_read_cost:.2f}ms")
            print(f"  - 最低耗时：{min_read_cost:.2f}ms")
            print(f"  - 平均耗时：{avg_read_cost:.2f}ms")
    else:
        print(f"无有效抽帧数据（已跳过前{skip_initial_frames}帧）")
    print("===============================================================\n")
    
    cap.release()
    print("✅ 抽帧线程退出")

def yolo_detection_thread(camera_index=settings.CAMERA_INDEX):
    """检测线程：修复None值运算错误+稳定推理耗时（支持多路，每路传入camera_index）"""
    global is_running
    init_camera_globals(camera_index)
    
    # ========== 优化1：固定推理设备+兼容式半精度判断 ==========
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 正确判断FP16（半精度）支持（兼容所有Torch版本）
    def is_fp16_supported():
        if not torch.cuda.is_available():
            return False
        # 获取GPU算力（算力≥5.0支持FP16）
        capability = torch.cuda.get_device_capability(0)
        return capability[0] >= 5  # 算力5.0及以上支持FP16
    
    use_half = is_fp16_supported()
    print(f"🔧 推理设备：{device} | 半精度支持：{use_half}（GPU算力≥5.0）")

    # ========== 优化2：初始化YOLO模型（固定参数） ==========
    try:
        model = YOLO("model/b_best.pt")
        # 模型移至固定设备，预热推理
        model.to(device)
        print(f"✅ 模型加载成功，类别列表：{model.names}")
    except Exception as e:
        print(f"❌ 加载模型失败：{e}，使用官方YOLOv8n")
        model = YOLO("yolov8n.pt")
        model.to(device)
        print(f"✅ 官方模型类别列表：{model.names}")

    # 诊断信息：打印CUDA与设备详情，便于判断是否在GPU上推理
    try:
        print(f"🔧 推理设备：{device} | 半精度支持：{use_half}（GPU算力≥5.0）")
        print(f"🔎 torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"🔎 CUDA device name: {torch.cuda.get_device_name(0)} | capability: {torch.cuda.get_device_capability(0)}")
            except Exception:
                pass
    except Exception:
        # 保护性兜底，避免诊断打印阻塞主逻辑
        pass

    # ========== 优化3：固定推理参数（核心修复：处理None值） ==========
    # 1. 优先用默认固定尺寸（32倍数），避免依赖未初始化的裁剪参数
    DEFAULT_IMGSZ = settings.MODEL_WARMUP_SIZE  # YOLOv8默认，32的倍数
    # 2. 尝试获取裁剪尺寸，若为None则用默认值
    calib_w = getattr(settings, "CALIB_OFFSET_W", None) or DEFAULT_IMGSZ[0]
    calib_h = getattr(settings, "CALIB_OFFSET_H", None) or DEFAULT_IMGSZ[1]
    # 3. 确保尺寸为32的倍数（处理None/0/非整数）
    FIXED_IMGSZ = (
        round(int(calib_w) / 32) * 32,
        round(int(calib_h) / 32) * 32
    )
    # 兜底：防止尺寸为0
    FIXED_IMGSZ = (
        FIXED_IMGSZ[0] if FIXED_IMGSZ[0] > 0 else DEFAULT_IMGSZ[0],
        FIXED_IMGSZ[1] if FIXED_IMGSZ[1] > 0 else DEFAULT_IMGSZ[1]
    )
    print(f"🔧 固定推理尺寸：{FIXED_IMGSZ}（32倍数）| 裁剪参数初始值：w={calib_w}, h={calib_h}")

    # ========== 优化4：模型预热（消除首次推理高耗时） ==========
    warmup_frame = np.zeros((FIXED_IMGSZ[1], FIXED_IMGSZ[0], 3), dtype=np.uint8)
    for _ in range(5):  # 预热5次，稳定推理耗时
        model(warmup_frame, conf=settings.CONF_THRESHOLD, verbose=False, 
              imgsz=FIXED_IMGSZ, half=use_half, device=device)
    print("✅ 模型预热完成，推理耗时已稳定")

    # ========== 检测主循环 ==========
    while is_running:
        # ========== 优化5：拆分计时（仅统计推理耗时，排除队列等待） ==========
        try:
            # 1. 先取帧（单独计时，排除到推理耗时外）
            frame_id, frame_original, frame_calib = frame_queues[camera_index].get(timeout=1.0)
        except queue.Empty:
            continue

        # 2. 推理前准备（确保帧尺寸稳定）
        if frame_calib is None or frame_calib.shape[0] == 0 or frame_calib.shape[1] == 0:
            continue  # 跳过空帧，避免推理异常

        # 3. 动态更新推理尺寸（可选：若裁剪参数已初始化，更新尺寸）
        current_calib_w = getattr(settings, "CALIB_OFFSET_W", None)
        current_calib_h = getattr(settings, "CALIB_OFFSET_H", None)
        if current_calib_w is not None and current_calib_h is not None and current_calib_w > 0 and current_calib_h > 0:
            dynamic_imgsz = (
                round(int(current_calib_w) / 32) * 32,
                round(int(current_calib_h) / 32) * 32
            )
            if dynamic_imgsz != FIXED_IMGSZ:
                FIXED_IMGSZ = dynamic_imgsz
                print(f"🔧 动态更新推理尺寸：{FIXED_IMGSZ}（裁剪参数已初始化）")

        # 4. 精准计时：按设置决定是否对帧进行缩放以加速检测
        infer_start = time.time()

        # YOLO检测：使用缩放后的图像和对应 imgsz，返回结果坐标为缩放图的坐标
        results = model(
            frame_calib,
            conf=settings.CONF_THRESHOLD,
            verbose=False,
            imgsz=settings.MODEL_WARMUP_SIZE,  # 始终使用预热尺寸，确保稳定
            half=use_half,
            device=device,
            batch=1,
            max_det=10,
            iou=0.7
        )

        # 将检测坐标按比例映射回原始裁剪帧尺寸，供后续逻辑使用
        try:
            scale_x = orig_w / float(target_w)
            scale_y = orig_h / float(target_h)
            first_target, all_targets = yolo_utils.get_first_detected_target(results, model, frame_id, scale=(scale_x, scale_y))
        except Exception:
            # 回退：不缩放坐标
            first_target, all_targets = yolo_utils.get_first_detected_target(results, model, frame_id)
        
        infer_end = time.time()
        infer_time = (infer_end - infer_start) * 1000
        # 打印仅推理耗时，排除队列等待
        print(f" 2️⃣ 2️⃣ 2️⃣ 帧ID：{frame_id} | 模型检测耗时：{infer_time:.2f} ms (稳定区间)")
        
        # 写入结果队列（自动删旧存新，无阻塞风险）
        result_queues[camera_index].put(
            (frame_id, frame_original, frame_calib, results, first_target, all_targets)
        )

    print("✅ 检测线程退出")

def predict_thread(camera_index=settings.CAMERA_INDEX):
    """预测线程：≥10帧且数据变化时请求接口，<10帧不操作（支持多路）"""
    global is_running
    init_camera_globals(camera_index)
    print(f"✅ 预测线程启动（摄像头{camera_index}）：等待缓存就绪...")
    
    # 等待缓存初始化
    while (camera_index not in target_frames_caches or target_frames_caches[camera_index] is None) and is_running:
        time.sleep(0.01)
    if not is_running:
        return

    # 记录上一次请求的帧ID列表（用于对比数据是否变化）
    last_request_frame_ids = None  

    while is_running:
        # 步骤1：加锁读取缓存
        with cache_locks[camera_index]:
            if target_frames_caches[camera_index] is None or len(target_frames_caches[camera_index]) == 0:
                time.sleep(0.01)
                continue
            current_cache = copy.deepcopy(list(target_frames_caches[camera_index]))  
        
        cache_len = len(current_cache)
        # ========== 核心逻辑1：<10帧 → 不请求、不返回值，直接跳过 ==========
        if cache_len < 10:
            time.sleep(0.01)
            continue
        
        # ========== 核心逻辑2：≥10帧 → 对比数据是否变化 ==========
        current_frame_ids = [item['frame_id'] for item in current_cache]
        
        # 场景1：首次≥10帧（无历史ID）→ 执行请求
        # 场景2：非首次但帧ID变化 → 执行请求
        if last_request_frame_ids is None or current_frame_ids != last_request_frame_ids:
            print(f"📌 缓存≥10帧且数据更新（当前帧ID：{current_frame_ids}），执行接口请求")
            
            # 组装请求数据
            try:
                request_data = predict_utils.assemble_predict_data(
                    cache=current_cache,
                    use_kalman=False,
                    conf_thresh=settings.CONF_THRESHOLD
                )
                if len(request_data["frame_data"]) == 0:
                    print("❌ 请求数据为空，跳过接口调用")
                    time.sleep(0.01)
                    continue
            except Exception as e:
                print(f"❌ 组装请求数据失败：{e}")
                time.sleep(0.01)
                continue
            
            # 调用预测接口
            start_api = time.time()
            response = predict_utils.call_predict_api(request_data)
            api_cost = (time.time() - start_api) * 1000
            print(f" 3️⃣ 3️⃣ 3️⃣ 预测接口耗时：{api_cost:.2f}ms")

            # 处理响应：仅请求成功且业务返回success时更新结果和历史ID
            if response and response.status_code == 200:
                try:
                    predict_result = response.json()
                    # 新增：判断业务层面是否成功（新接口的status字段）
                    business_status = predict_result.get('status', '')
                    if business_status != 'success':
                        print(f"⚠️ 接口返回200但业务失败 → 状态：{business_status}，结果：{predict_result}")
                    else:
                        print(f"✅ 预测完成（摄像头{camera_index}） → 状态：{business_status}")
                        with cache_locks[camera_index]:
                            last_predict_results[camera_index] = {"type": "api_result", "data": predict_result}
                            print(f"传入的最新检测帧（10帧）--->{current_cache}，预测结果--->{predict_result}")
                        # 仅业务成功时更新历史ID（保证数据准确性）
                        last_request_frame_ids = current_frame_ids
                except json.JSONDecodeError:
                    print(f"❌ 响应解析失败 → 非JSON格式，响应内容：{response.text}")
                except Exception as e:
                    print(f"❌ 响应处理异常 → {e}，响应内容：{response.text}")
            else:
                status_code = response.status_code if response else "无响应"
                print(f"❌ 请求失败（状态码{status_code}），保留历史ID等待下次变化")
        
        # 场景3：≥10帧但数据未变化 → 不请求、不返回值
        else:
            print(f"⚠️ 缓存≥10帧但数据未变化（帧ID：{current_frame_ids}），跳过请求")
        
        time.sleep(0.001)

    print("✅ 预测线程退出")