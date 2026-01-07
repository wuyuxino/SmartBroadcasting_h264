import os
import torch
import numpy as np
import queue
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging
import logging.handlers
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from collections import defaultdict
import time
import types

# 导入你的模型类
from model import TrajectoryPredictor, KalmanFilter  
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ===================== 动态路径初始化（核心修改） =====================
# 1. 获取当前脚本（predict_service.py）的所在目录（绝对路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # service/predict/
# 2. 获取项目根目录（SmartBroadcasting/，向上两级）
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))  # 从service/predict/向上到SmartBroadcasting/


# ===================== 全局配置（Windows 高并发兼容版） =====================
CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "service", "predict", "train_results", "best_model.pth"),
    "log_save_dir": os.path.join(PROJECT_ROOT, "service", "predict", "future_3frames_predictions"),
    "img_w": 3744,
    "img_h": 1920,
    # 高并发配置（Windows 兼容）
    "max_workers": 8,  # 线程池大小（CPU核心数*2）
    "kf_pool_size": 32,  # 卡尔曼滤波器池大小
    "batch_size": 4,  # 推理批处理大小
    "timeout": 1.0,  # 请求超时时间（秒）
    "max_concurrent_gpu": 4,  # 最大并发GPU请求数
    "rate_limit": "10000/second",  # 限流调整为20QPS
}

# ===================== 全局变量（高并发安全） =====================
# 关键：确保DEVICE是torch.device对象，且不被覆盖
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
SCALER = None
TARGET_SCALER = None
MODEL_CONFIG = None
KF_POOL = []  # 改为池化列表
KF_POOL_LOCK = Lock()  # 滤波器池锁
LOG_LISTENER = None
EXECUTOR = None  # 全局线程池
GPU_SEMAPHORE = None  # GPU并发控制信号量
REQUEST_METRICS = defaultdict(int)  # 请求统计
METRICS_LOCK = Lock()

# ===================== 工具函数（新增参数校验） =====================
def validate_model_config(config: Dict) -> Dict:
    """校验并清理模型配置（强制所有参数为整数，防止Conv1d参数错误）"""
    required_keys = ['input_dim', 'pred_len', 'd_model', 'nhead', 'num_layers', 'conv_channels']
    clean_config = {}
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"模型配置缺少关键键：{key}")
        
        val = config[key]
        # 过滤函数对象
        if isinstance(val, (types.FunctionType, types.MethodType)):
            raise TypeError(f"模型配置中 {key} 是函数对象（预期数值）")
        # 关键修正：所有模型参数强制转为整数（Conv1d要求整数）
        try:
            clean_config[key] = int(val)  # 包括d_model在内，全部转整数
        except (ValueError, TypeError) as e:
            raise TypeError(f"模型配置 {key} 无法转换为整数：{val}（类型：{type(val)}）→ 错误：{e}")
    
    # 补充默认值
    clean_config.setdefault('dropout', 0.0)
    return clean_config

def numpy_to_python(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

def update_metrics(key: str):
    """线程安全更新请求指标"""
    with METRICS_LOCK:
        REQUEST_METRICS[key] += 1

class AsyncRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """异步日志处理器"""
    def emit(self, record):
        try:
            asyncio.get_event_loop().call_soon_threadsafe(super().emit, record)
        except Exception:
            self.handleError(record)

# ===================== 异步日志初始化（高并发版） =====================
def init_async_logger():
    """高并发异步日志初始化"""
    log_dir = CONFIG["log_save_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "api_high_concurrency.log")

    # 1. 创建异步日志处理器
    file_handler = AsyncRotatingFileHandler(
        log_path, maxBytes=50*1024*1024, backupCount=10, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    ))

    # 2. 配置根日志
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(logging.StreamHandler())

    return file_handler

# ===================== 卡尔曼滤波器池管理（高并发版） =====================
def init_kf_pool():
    """初始化卡尔曼滤波器池"""
    global KF_POOL
    KF_POOL = [
        KalmanFilter(dt=1.0, std_acc=1.0, std_meas=0.1)
        for _ in range(CONFIG["kf_pool_size"])
    ]
    logging.info(f"✅ 卡尔曼滤波器池初始化完成，大小：{CONFIG['kf_pool_size']}")

def get_kf_from_pool() -> KalmanFilter:
    """从池获取滤波器（线程安全）"""
    with KF_POOL_LOCK:
        if KF_POOL:
            return KF_POOL.pop()
        # 池空时创建临时滤波器
        return KalmanFilter(dt=1.0, std_acc=1.0, std_meas=0.1)

def return_kf_to_pool(kf: KalmanFilter):
    """归还滤波器到池（线程安全）"""
    with KF_POOL_LOCK:
        if len(KF_POOL) < CONFIG["kf_pool_size"]:
            KF_POOL.append(kf)

# ===================== 模型预加载（新增错误防护） =====================
def preload_resources():
    """高并发模型预加载（含参数校验+错误防护）"""
    global MODEL, SCALER, TARGET_SCALER, MODEL_CONFIG, EXECUTOR, GPU_SEMAPHORE
    
    try:
        # 1. 初始化GPU并发控制
        GPU_SEMAPHORE = asyncio.Semaphore(CONFIG["max_concurrent_gpu"]) if DEVICE.type == "cuda" else None
        
        # 2. 初始化线程池
        EXECUTOR = ThreadPoolExecutor(
            max_workers=CONFIG["max_workers"],
            thread_name_prefix="infer_worker"
        )
        
        # 3. 加载并校验checkpoint
        logging.info(f"📥 加载模型 checkpoint：{CONFIG['model_path']}")
        checkpoint = torch.load(CONFIG["model_path"], map_location=DEVICE, weights_only=False)
        
        # 关键：校验模型配置（强制所有参数为整数）
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint 缺少 'config' 键")
        MODEL_CONFIG = validate_model_config(checkpoint['config'])
        logging.info(f"✅ 模型配置校验通过（全整数）：{MODEL_CONFIG}")
        
        # 4. 初始化标准化器（增加类型校验）
        for scaler_key in ['scaler_mean', 'scaler_scale', 'target_scaler_mean', 'target_scaler_scale']:
            if scaler_key not in checkpoint:
                raise ValueError(f"Checkpoint 缺少 {scaler_key}")
            if not isinstance(checkpoint[scaler_key], (list, np.ndarray)):
                raise TypeError(f"{scaler_key} 不是列表/数组：{type(checkpoint[scaler_key])}")

        SCALER = StandardScaler()
        SCALER.mean_ = np.array(checkpoint['scaler_mean'], dtype=np.float32)
        SCALER.scale_ = np.array(checkpoint['scaler_scale'], dtype=np.float32)
        SCALER.scale_[SCALER.scale_ < 1e-6] = 1e-6

        TARGET_SCALER = StandardScaler()
        TARGET_SCALER.mean_ = np.array(checkpoint['target_scaler_mean'], dtype=np.float32)
        TARGET_SCALER.scale_ = np.array(checkpoint['target_scaler_scale'], dtype=np.float32)
        TARGET_SCALER.scale_[TARGET_SCALER.scale_ < 1e-6] = 1e-6

        # 5. 初始化Transformer模型（关键：确保所有参数都是整数）
        logging.info(f"🔧 初始化模型，设备：{DEVICE}，参数全整数")
        # 显式提取整数参数，避免浮点数传入
        input_dim = int(MODEL_CONFIG['input_dim'])
        pred_len = int(MODEL_CONFIG['pred_len'])
        d_model = int(MODEL_CONFIG['d_model'])
        nhead = int(MODEL_CONFIG['nhead'])
        num_layers = int(MODEL_CONFIG['num_layers'])
        conv_channels = int(MODEL_CONFIG['conv_channels'])
        
        model = TrajectoryPredictor(
            input_dim=input_dim,
            output_dim=pred_len * 2,  # 显式计算，确保整数
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            conv_channels=conv_channels,
            dropout=0.0  # 推理时禁用dropout
        )
        # 先加载权重，再移到设备（避免设备不匹配）
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        MODEL = model.to(DEVICE)  # 关键：确保model是实例，而非函数
        MODEL.eval()  # 纯eval模式
        
        # 6. GPU优化（确认DEVICE是torch.device对象）
        if isinstance(DEVICE, torch.device) and DEVICE.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            logging.warning(f"⚠️ DEVICE 不是CUDA设备：{DEVICE}（类型：{type(DEVICE)}）")
        
        # 7. 模型预热（确保输入尺寸是整数）
        dummy_input = torch.randn(1, 10, len(SCALER.mean_), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            for _ in range(5):
                MODEL(dummy_input)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
        
        # 8. 初始化卡尔曼滤波器池
        init_kf_pool()
        
        logging.info(f"""✅ 高并发模型预加载成功！
        - 设备类型：{type(DEVICE)} → {DEVICE}
        - 模型配置（全整数）：{MODEL_CONFIG}
        - 线程池大小：{CONFIG['max_workers']}
        - GPU最大并发：{CONFIG['max_concurrent_gpu']}
        """)
        
    except Exception as e:
        logging.error(f"❌ 模型预加载失败：{str(e)}", exc_info=True)
        # 打印关键变量类型，辅助定位
        logging.error(f"🔍 关键变量类型排查：")
        logging.error(f"   - DEVICE: {type(DEVICE)} → {DEVICE}")
        logging.error(f"   - MODEL_CONFIG: {type(MODEL_CONFIG)} → {MODEL_CONFIG if MODEL_CONFIG else '未加载'}")
        logging.error(f"   - checkpoint['config']: {type(checkpoint.get('config')) if 'checkpoint' in locals() else '未加载'}")
        # 打印Conv1d相关参数
        if 'MODEL_CONFIG' in locals() and MODEL_CONFIG:
            logging.error(f"   - d_model（整数校验后）: {MODEL_CONFIG['d_model']}（类型：{type(MODEL_CONFIG['d_model'])}）")
            logging.error(f"   - conv_channels: {MODEL_CONFIG['conv_channels']}（类型：{type(MODEL_CONFIG['conv_channels'])}）")
        raise RuntimeError(f"模型预加载失败：{str(e)}")

# ===================== FastAPI生命周期管理 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI生命周期管理（启动/关闭）"""
    # 启动阶段
    global LOG_LISTENER
    LOG_LISTENER = init_async_logger()
    try:
        preload_resources()
    except Exception as e:
        logging.critical(f"💥 服务启动失败：{e}")
        raise  # 终止服务启动
    logging.info("🚀 高并发轨迹预测API启动完成（Windows兼容版）")
    yield
    # 关闭阶段
    if EXECUTOR:
        EXECUTOR.shutdown(wait=True)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    logging.info("🛑 高并发轨迹预测API已关闭")

# ===================== FastAPI实例创建 =====================
app = FastAPI(
    title="轨迹预测API（高并发版）",
    description="支持每秒20次请求，Windows兼容",
    version="3.0.0",
    lifespan=lifespan
)

# ===================== 中间件配置 =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# ===================== 限流配置 =====================
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[CONFIG["rate_limit"]],
    storage_uri="memory://"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===================== 数据模型 =====================
class FrameData(BaseModel):
    frame_id: int = Field(..., description="帧ID（唯一、递增）")
    x1: float = Field(..., description="目标框左上角x坐标")
    y1: float = Field(..., description="目标框左上角y坐标")
    w: float = Field(..., description="目标框宽度")
    h: float = Field(..., description="目标框高度")
    conf: float = Field(..., ge=0.0, le=1.0, description="置信度（0-1）")

class PredictRequest(BaseModel):
    frame_data: List[FrameData] = Field(..., description="帧数据列表（需≥10帧有效数据）")
    use_kalman: bool = Field(False, description="是否启用卡尔曼滤波")
    conf_thresh: float = Field(0.5, ge=0.0, le=1.0, description="置信度过滤阈值")

class PredictResponse(BaseModel):
    code: int = Field(..., description="状态码：200成功/500失败")
    msg: str = Field(..., description="状态信息")
    data: Optional[Dict[str, Any]] = Field(None, description="预测结果（JSON格式）")
    error: Optional[str] = Field(None, description="错误详情")
    latency: Optional[float] = Field(None, description="请求耗时（秒）")

# ===================== 核心推理函数 =====================
def process_frame_data(frame_data: List[FrameData], conf_thresh: float) -> tuple[List[np.ndarray], List[int]]:
    """处理帧数据"""
    sorted_frames = sorted(frame_data, key=lambda x: x.frame_id)
    if len(sorted_frames) < 10:
        raise ValueError(f"帧数据不足（需≥10帧），实际仅{len(sorted_frames)}帧")

    max_frames = len(sorted_frames)
    real_frames = np.zeros((max_frames, 4), dtype=np.float32)
    frame_ids = []
    valid_idx = 0

    for frame in sorted_frames:
        if frame.conf < conf_thresh:
            continue
        real_frames[valid_idx, 0] = frame.x1 + frame.w / 2
        real_frames[valid_idx, 1] = frame.y1 + frame.h / 2
        real_frames[valid_idx, 2] = frame.w
        real_frames[valid_idx, 3] = frame.h
        frame_ids.append(frame.frame_id)
        valid_idx += 1

    real_frames = real_frames[:valid_idx]
    if len(real_frames) < 10:
        raise ValueError(f"有效帧不足（需≥10帧），实际仅{len(real_frames)}帧")
    
    real_frames_list = [real_frames[i] for i in range(len(real_frames))]
    return real_frames_list, frame_ids

def preprocess_history_batch(history_list: List[List[np.ndarray]]) -> torch.Tensor:
    """批量预处理历史数据"""
    batch_data = []
    for history_frames in history_list:
        history = history_frames[-10:] if len(history_frames) > 10 else history_frames
        if len(history) < 10:
            history = [history[0]] * (10 - len(history)) + history
        
        history_np = np.array(history, dtype=np.float32)
        delta = np.zeros_like(history_np)
        delta[1:] = history_np[1:] - history_np[:-1]
        input_feat = np.concatenate([history_np, delta], axis=1)
        batch_data.append(SCALER.transform(input_feat))
    
    batch_tensor = torch.FloatTensor(np.array(batch_data)).to(DEVICE)
    return batch_tensor

def predict_future_batch(
        history_list: List[List[np.ndarray]],
        kf_list: List[Optional[KalmanFilter]],
        current_frame_ids: List[int]
) -> Tuple[List[List[Dict]], List[List[int]], List[List[np.ndarray]]]:
    """批量预测未来3帧"""
    # 批量预处理
    input_seq = preprocess_history_batch(history_list)
    
    # 批量推理
    with torch.no_grad():
        pred_norm = MODEL(input_seq)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    
    # 批量反标准化
    pred_raw_all = TARGET_SCALER.inverse_transform(
        pred_norm.cpu().numpy().reshape(-1, 6)
    ).reshape(-1, 3, 2)
    
    # 批量处理结果
    all_pred_results = []
    all_future_ids = []
    all_pred_coords = []
    
    for i in range(len(history_list)):
        history_frames = history_list[i]
        kf = kf_list[i]
        current_frame_id = current_frame_ids[i]
        pred_raw = pred_raw_all[i]
        
        future_frame_ids = [current_frame_id + 1 + t for t in range(3)]
        pred_results = []
        prev_w = history_frames[-1][2]
        prev_h = history_frames[-1][3]
        pred_coords = []
        
        for t in range(3):
            pred_cx, pred_cy = pred_raw[t]
            future_id = future_frame_ids[t]

            if kf is not None:
                kf.predict()
                pred_cx, pred_cy = kf.update(np.array([pred_cx, pred_cy]))

            # 边界裁剪
            pred_cx = np.clip(pred_cx, 0, CONFIG["img_w"])
            pred_cy = np.clip(pred_cy, 0, CONFIG["img_h"])
            x1 = pred_cx - prev_w / 2
            y1 = pred_cy - prev_h / 2
            x1 = np.clip(x1, 0, CONFIG["img_w"] - prev_w)
            y1 = np.clip(y1, 0, CONFIG["img_h"] - prev_h)

            pred_results.append({
                "frame_id": int(future_id),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "w": round(float(prev_w), 2),
                "h": round(float(prev_h), 2),
                "conf": 0.90
            })
            pred_coords.append(np.array([x1 + prev_w/2, y1 + prev_h/2]))
        
        all_pred_results.append(pred_results)
        all_future_ids.append(future_frame_ids)
        all_pred_coords.append(pred_coords)
    
    return all_pred_results, all_future_ids, all_pred_coords

def continuous_infer_core(
        frame_data: List[FrameData],
        use_kalman: bool = False,
        conf_thresh: float = 0.5
) -> Dict[str, Any]:
    """核心推理函数"""
    start_time = time.time()
    try:
        real_frames, real_frame_ids = process_frame_data(frame_data, conf_thresh)

        history_frames = real_frames[:10].copy()
        history_frame_ids = real_frame_ids[:10].copy()
        all_predictions = []
        pred_records = []
        total_frames = len(real_frame_ids) - 10
        processed_count = 0

        # 获取卡尔曼滤波器
        kf = get_kf_from_pool() if use_kalman else None
        if kf and use_kalman:
            kf.init_state(history_frames[-1][:2])

        # 单请求推理
        while len(history_frames) == 10:
            current_frame_id = history_frame_ids[-1]
            processed_count += 1

            # 单样本预测
            pred_results, future_ids, pred_coords = predict_future_batch(
                history_list=[history_frames],
                kf_list=[kf],
                current_frame_ids=[current_frame_id]
            )
            pred_results = pred_results[0]
            future_ids = future_ids[0]
            pred_coords = pred_coords[0]

            all_predictions.append({
                "current_frame_id": int(current_frame_id),
                "future_frames": pred_results
            })

            # 记录预测结果
            for t in range(3):
                future_id = future_ids[t]
                pred_cx, pred_cy = pred_coords[t]
                real_cx, real_cy = None, None
                
                if future_id in real_frame_ids:
                    real_idx = real_frame_ids.index(future_id)
                    real_cx, real_cy = real_frames[real_idx][0], real_frames[real_idx][1]
                
                pred_records.append({
                    "current_frame": int(current_frame_id),
                    "future_frame": int(future_id),
                    "pred_cx": float(pred_cx),
                    "pred_cy": float(pred_cy),
                    "real_cx": float(real_cx) if real_cx else None,
                    "real_cy": float(real_cy) if real_cy else None
                })

            # 滑动窗口
            history_frames.pop(0)
            history_frame_ids.pop(0)
            first_future_id = future_ids[0]
            
            if first_future_id in real_frame_ids:
                real_idx = real_frame_ids.index(first_future_id)
                history_frames.append(real_frames[real_idx])
                history_frame_ids.append(first_future_id)
            else:
                break

        # 归还卡尔曼滤波器
        if kf:
            return_kf_to_pool(kf)

        # 计算准确率
        valid_errors = []
        for record in pred_records:
            if record['real_cx'] and record['real_cy']:
                error = np.sqrt((record['pred_cx'] - record['real_cx']) ** 2 + 
                                (record['pred_cy'] - record['real_cy']) ** 2)
                valid_errors.append(error)

        accuracy = {}
        if valid_errors:
            frame1_errors = valid_errors[::3] if len(valid_errors) >= 3 else []
            frame2_errors = valid_errors[1::3] if len(valid_errors) >= 3 else []
            frame3_errors = valid_errors[2::3] if len(valid_errors) >= 3 else []

            accuracy = {
                "总平均误差(px)": round(float(np.mean(valid_errors)), 2),
                "未来1帧平均误差(px)": round(float(np.mean(frame1_errors)), 2) if frame1_errors else 0.0,
                "未来2帧平均误差(px)": round(float(np.mean(frame2_errors)), 2) if frame2_errors else 0.0,
                "未来3帧平均误差(px)": round(float(np.mean(frame3_errors)), 2) if frame3_errors else 0.0,
                "最大误差(px)": round(float(np.max(valid_errors)), 2),
                "最小误差(px)": round(float(np.min(valid_errors)), 2),
                "≤5px成功率(%)": round(float(sum(1 for e in valid_errors if e <= 5) / len(valid_errors) * 100), 2),
                "≤10px成功率(%)": round(float(sum(1 for e in valid_errors if e <= 10) / len(valid_errors) * 100), 2),
                "有效预测帧数": len(valid_errors)
            }
        else:
            accuracy = {"提示": "无有效真实值"}

        latency = time.time() - start_time
        update_metrics("success")
        
        return {
            "all_predictions": numpy_to_python(all_predictions),
            "accuracy": numpy_to_python(accuracy),
            "processed_frames": processed_count,
            "total_frames": total_frames,
            "device": str(DEVICE),
            "latency": round(latency, 4)
        }
    
    except Exception as e:
        update_metrics("error")
        raise e

# ===================== API接口 =====================
@app.post("/predict", response_model=PredictResponse, summary="轨迹预测接口（高并发版）")
@limiter.limit(CONFIG["rate_limit"])
async def predict_trajectory(
    request: Request,
    predict_req: PredictRequest,
    background_tasks: BackgroundTasks
):
    """异步推理接口，支持超时控制和GPU并发限制"""
    start_time = time.time()
    try:
        # GPU并发控制
        if GPU_SEMAPHORE:
            async with GPU_SEMAPHORE:
                # 提交到线程池执行推理
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        EXECUTOR,
                        continuous_infer_core,
                        predict_req.frame_data,
                        predict_req.use_kalman,
                        predict_req.conf_thresh
                    ),
                    timeout=CONFIG["timeout"]
                )
        else:
            # CPU推理
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    EXECUTOR,
                    continuous_infer_core,
                    predict_req.frame_data,
                    predict_req.use_kalman,
                    predict_req.conf_thresh
                ),
                timeout=CONFIG["timeout"]
            )
        
        latency = time.time() - start_time
        return PredictResponse(
            code=200,
            msg="预测成功",
            data={
                "预测结果": result["all_predictions"],
                "准确率统计": result["accuracy"],
                "处理帧数": result["processed_frames"],
                "总帧数": result["total_frames"],
                "使用设备": result["device"],
                "推理耗时(秒)": result["latency"],
                "请求总耗时(秒)": round(latency, 4)
            },
            latency=round(latency, 4)
        )
    
    except asyncio.TimeoutError:
        update_metrics("timeout")
        logging.error(f"请求超时（{CONFIG['timeout']}秒）")
        return PredictResponse(
            code=500,
            msg="请求超时",
            error=f"请求处理超时（超过{CONFIG['timeout']}秒）",
            latency=round(time.time() - start_time, 4)
        )
    except Exception as e:
        error_msg = str(e)
        logging.error(f"预测失败：{error_msg}", exc_info=True)
        return PredictResponse(
            code=500,
            msg="预测失败",
            error=error_msg,
            latency=round(time.time() - start_time, 4)
        )

@app.get("/health", summary="健康检查（高并发版）")
async def health_check():
    """增强型健康检查"""
    gpu_mem = None
    if isinstance(DEVICE, torch.device) and DEVICE.type == "cuda":
        gpu_mem = {
            "已用(MB)": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "最大分配(MB)": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
            "缓存(MB)": round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
        }
    
    with METRICS_LOCK:
        metrics = dict(REQUEST_METRICS)
    
    return {
        "code": 200,
        "msg": "服务正常（高并发模式）",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cuda可用": torch.cuda.is_available(),
        "DEVICE类型": str(type(DEVICE)),
        "DEVICE值": str(DEVICE),
        "模型已加载": MODEL is not None,
        "卡尔曼滤波器池大小": len(KF_POOL),
        "线程池大小": CONFIG["max_workers"],
        "GPU最大并发": CONFIG["max_concurrent_gpu"],
        "当前限流": CONFIG["rate_limit"],
        "GPU内存使用": gpu_mem,
        "请求统计": {
            "成功数": metrics.get("success", 0),
            "错误数": metrics.get("error", 0),
            "超时数": metrics.get("timeout", 0),
            "总请求数": metrics.get("success", 0) + metrics.get("error", 0) + metrics.get("timeout", 0)
        }
    }

@app.get("/metrics", summary="性能指标监控")
async def get_metrics():
    """获取实时性能指标"""
    with METRICS_LOCK:
        metrics = dict(REQUEST_METRICS)
    
    return {
        "code": 200,
        "请求统计": metrics,
        "卡尔曼滤波器池使用率": 1 - len(KF_POOL)/CONFIG["kf_pool_size"],
        "GPU并发数": CONFIG["max_concurrent_gpu"] - GPU_SEMAPHORE._value if GPU_SEMAPHORE else 0,
        "线程池大小": CONFIG["max_workers"],
        "批处理大小": CONFIG["batch_size"]
    }

# ===================== 启动服务 =====================
if __name__ == "__main__":
    # 最终校验DEVICE类型
    if not isinstance(DEVICE, torch.device):
        logging.critical(f"💥 DEVICE 不是 torch.device 对象：{type(DEVICE)} → {DEVICE}")
        exit(1)
    
    # Windows兼容启动
    uvicorn.run(
        app="__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="asyncio",
        access_log=True,
        log_level="info",
        timeout_keep_alive=5,
    )