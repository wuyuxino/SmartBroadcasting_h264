import os
import time
import torch
import numpy as np
import glob
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from config import settings
import json

# 导入模型
from model import PhysicsAwareTrajectoryPredictor, EnhancedKalmanFilter

app = Flask(__name__)

# 全局变量存储模型和配置
model: Optional[PhysicsAwareTrajectoryPredictor] = None
config: Optional[Dict] = None
scaler: Optional[StandardScaler] = None
target_scaler: Optional[StandardScaler] = None
device: Optional[torch.device] = None
kf: Optional[EnhancedKalmanFilter] = None
log_file = None

# ===================== 日志配置 =====================
def init_logger(log_path: str):
    """初始化日志"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(f"\n{'=' * 50} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'=' * 50}\n")
    return log_file


def log_info(msg: str, to_terminal: bool = False):
    """写入日志"""
    if log_file:
        log_file.write(f"{datetime.now().strftime('%H:%M:%S')} - {msg}\n")
        log_file.flush()
    if to_terminal:
        print(msg)


# ===================== 模型加载 =====================
def load_model_service(model_path: str):
    """服务启动时加载模型"""
    global model, config, scaler, target_scaler, device, kf
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        log_info("模型加载成功", to_terminal=True)
        config = checkpoint['config']

        # 初始化标准化器
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'])
        scaler.scale_ = np.array(checkpoint['scaler_scale'])
        scaler.scale_[scaler.scale_ < 1e-6] = 1e-6

        target_scaler = StandardScaler()
        target_scaler.mean_ = np.array(checkpoint['target_scaler_mean'])
        target_scaler.scale_ = np.array(checkpoint['target_scaler_scale'])
        target_scaler.scale_[target_scaler.scale_ < 1e-6] = 1e-6

        # 初始化模型
        model = PhysicsAwareTrajectoryPredictor(
            input_dim=config.get('input_dim', 12),
            output_dim=config.get('pred_len', 3) * 2,
            d_model=config.get('d_model', 512),
            nhead=config.get('nhead', 16),
            num_layers=config.get('num_layers', 8),
            conv_channels=config.get('conv_channels', 256),
            dropout=config.get('dropout', 0.2),
            max_acceleration=config.get('max_acceleration', 100.0),
            max_velocity_change=config.get('max_velocity_change', 50.0)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        
        # 初始化卡尔曼滤波器
        if config.get('use_kalman', False):
            kf = EnhancedKalmanFilter(
                dt=1.0,
                std_acc=1.0,
                std_meas=0.1,
                max_acceleration=config.get('max_acceleration', 100.0)
            )

        log_info("模型服务初始化完成", to_terminal=True)
        return True
    except Exception as e:
        log_info(f"模型加载失败：{str(e)}", to_terminal=True)
        return False


# ===================== 数据处理 =====================
def preprocess_history(frame_data: List[np.ndarray]) -> torch.Tensor:
    """预处理历史数据"""
    global scaler
    if not scaler:
        raise RuntimeError("标准化器未初始化")

    # 确保有10帧历史数据
    frame_data = frame_data[-10:] if len(frame_data) > 10 else frame_data
    if len(frame_data) < 10:
        frame_data = [frame_data[0]] * (10 - len(frame_data)) + frame_data

    history = np.array(frame_data, dtype=np.float32)  # (10, 4)

    # 计算速度和加速度
    velocity = np.zeros_like(history)
    velocity[1:] = history[1:] - history[:-1]

    acceleration = np.zeros_like(history)
    if len(history) >= 3:
        acceleration[2:] = velocity[2:] - velocity[1:-1]

    # 组合特征
    input_feat = np.concatenate([history, velocity, acceleration], axis=1)  # (10, 12)
    input_feat = scaler.transform(input_feat)
    return torch.FloatTensor(input_feat).unsqueeze(0)  # (1, 10, 12)


# ===================== 预测接口 =====================
@app.route('/predict', methods=['POST'])
def predict():
    """预测接口
    请求体格式:
    {
        "frame_data": [
            {
                "frame_id": int,
                "x1": float,
                "y1": float,
                "w": float,
                "h": float,
                "conf": float
            },
            ...
        ],
        "use_kalman": bool,
        "conf_thresh": float
    }
    """
    global model, target_scaler, device, kf

    try:
        # 解析请求数据
        data = request.json
        if not data or "frame_data" not in data:
            return jsonify({"error": "缺少frame_data"}), 400

        # 提取基础参数
        frame_data_list = data["frame_data"]
        use_kalman = data.get("use_kalman", False)
        conf_thresh = data.get("conf_thresh", 0.25)
        img_w = int(settings.IMAGE_WIDTH) # 配置同步
        img_h = int(settings.IMAGE_HEIGHT) # 配置同步

        # 解析帧数据：提取cx, cy, w, h（过滤低置信度）
        frame_data = []
        for frame_dict in frame_data_list:
            # 从字典中提取字段（确保类型正确）
            try:
                x1 = float(frame_dict["x1"])
                y1 = float(frame_dict["y1"])
                w = float(frame_dict["w"])
                h = float(frame_dict["h"])
                conf = float(frame_dict["conf"])
            except (KeyError, ValueError) as e:
                return jsonify({"error": f"帧数据格式错误：{str(e)}"}), 400

            # 过滤低置信度帧
            if conf < conf_thresh:
                continue

            # 计算中心点坐标
            cx = x1 + w / 2
            cy = y1 + h / 2

            # 保存为 [cx, cy, w, h]
            frame_data.append(np.array([cx, cy, w, h], dtype=np.float32))

        # 验证有效帧数
        if len(frame_data) < 10:
            return jsonify({"error": f"有效历史帧数不足（需≥10，实际{len(frame_data)}）"}), 400

        # 获取当前最大frame_id作为基准
        current_frame_id = max(int(frame["frame_id"]) for frame in frame_data_list)

        # 预处理
        input_seq = preprocess_history(frame_data).to(device)

        # 预测
        with torch.no_grad():
            pred_output = model(input_seq)
            pred_constrained = pred_output[0]  # 使用约束后的预测结果

        # ========== 新增日志：反标准化前的数值和target_scaler参数 ==========
        # log_info(f"当前帧 {current_frame_id} 反标准化前预测值：{pred_constrained.cpu().numpy()}")
        # log_info(f"target_scaler均值：{target_scaler.mean_}，标准差：{target_scaler.scale_}")

        # 反标准化
        pred_raw_all = target_scaler.inverse_transform(
            pred_constrained.cpu().numpy().reshape(1, 6)
        ).reshape(3, 2)

        # ========== 新增日志：反标准化后预测中心点 ==========
        # log_info(f"当前帧 {current_frame_id} 反标准化后预测中心点：{pred_raw_all}")

        # 生成结果
        future_frame_ids = [current_frame_id + 1 + t for t in range(3)]
        results = []
        prev_w, prev_h = frame_data[-1][2], frame_data[-1][3]  # 用最后一帧的宽高

        # 初始化卡尔曼滤波器（如果启用）
        current_kf = None
        if use_kalman:
            current_kf = EnhancedKalmanFilter(
                dt=1.0,
                std_acc=1.0,
                std_meas=0.1,
                max_acceleration=config.get('max_acceleration', 100.0)
            )
            current_kf.init_state(frame_data[-1][:2])  # 用最后一帧的中心点初始化

        for t in range(3):
            pred_cx, pred_cy = pred_raw_all[t]
            future_id = future_frame_ids[t]

            # 卡尔曼滤波处理
            if use_kalman and current_kf:
                current_kf.predict()
                pred_cx, pred_cy = current_kf.update(np.array([pred_cx, pred_cy]))

            # 边界裁剪
            pred_cx = np.clip(pred_cx, 0, img_w)
            pred_cy = np.clip(pred_cy, 0, img_h)
            x1 = max(0, min(pred_cx - prev_w / 2, img_w - prev_w))
            y1 = max(0, min(pred_cy - prev_h / 2, img_h - prev_h))

            results.append({
                "future_frame_id": future_id,
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "w": round(float(prev_w), 2),
                "h": round(float(prev_h), 2),
                "confidence": 0.9,
                "pred_cx": round(float(pred_cx), 2),
                "pred_cy": round(float(pred_cy), 2)
            })

        return jsonify({
            "status": "success",
            "predictions": results,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        log_info(f"预测失败：{str(e)}")
        return jsonify({"error": str(e)}), 500


# ===================== 服务启动 =====================
if __name__ == "__main__":
    # 配置参数（请根据实际路径修改）
    CONFIG = {
        "model_path": "D:\\work\\code\\SmartBroadcasting\\service\\predict\\train_results\\best_model.pth",
        "log_path": "D:\\work\\code\\SmartBroadcasting\\service\\predict\\server_log.txt",
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False
    }

    # 初始化日志
    log_file = init_logger(CONFIG["log_path"])

    # 加载模型
    load_success = load_model_service(CONFIG["model_path"])
    if not load_success:
        log_file.close()
        exit(1)

    # 启动服务
    try:
        app.run(
            host=CONFIG["host"],
            port=CONFIG["port"],
            debug=CONFIG["debug"],
            threaded=True
        )
    finally:
        if log_file:
            log_file.close()