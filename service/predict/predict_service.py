import os
import torch
import numpy as np
from flask import Flask, request, jsonify
import json
from typing import List, Dict, Optional
from datetime import datetime

app = Flask(__name__)

# 全局变量
device: torch.device = None
model = None
norm_stats: Dict = None

# ===================== 模型定义 =====================
class KFDeepLearningModel(torch.nn.Module):
    def __init__(self):
        super(KFDeepLearningModel, self).__init__()
        self.Q_log = torch.nn.Parameter(torch.log(torch.eye(4, dtype=torch.float32) * 0.1))
        self.R_log = torch.nn.Parameter(torch.log(torch.eye(2, dtype=torch.float32) * 1.0))

        self.F = torch.tensor([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float32)
        self.H = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=torch.float32)
        self.init_P = torch.eye(4, dtype=torch.float32) * 1000.0

    @property
    def Q(self):
        return torch.exp(self.Q_log) + 1e-6 * torch.eye(4, dtype=torch.float32).to(self.Q_log.device)

    @property
    def R(self):
        return torch.exp(self.R_log) + 1e-6 * torch.eye(2, dtype=torch.float32).to(self.R_log.device)

    def forward(self, history_obs: torch.Tensor, norm_stats: dict = None, denorm: bool = True) -> torch.Tensor:
        F = self.F.to(history_obs.device)
        H = self.H.to(history_obs.device)
        init_P = self.init_P.to(history_obs.device)

        x0, y0 = history_obs[0, 0], history_obs[0, 1]
        X = torch.tensor([x0, y0, 0.0, 0.0], dtype=torch.float32).to(history_obs.device).reshape(4, 1)
        P = init_P.clone()

        for obs in history_obs:
            X = F @ X
            P = F @ P @ F.T + self.Q
            z = obs.reshape(2, 1)
            S = H @ P @ H.T + self.R
            K = P @ H.T @ torch.inverse(S)
            X = X + K @ (z - H @ X)
            P = (torch.eye(4).to(history_obs.device) - K @ H) @ P

        current_x, current_y = X[0, 0], X[1, 0]
        vx, vy = X[2, 0], X[3, 0]
        future_pred = []
        for k in range(1, 4):
            pred_x = current_x + k * vx
            pred_y = current_y + k * vy
            pred_tensor = torch.cat([pred_x.unsqueeze(0), pred_y.unsqueeze(0)])
            future_pred.append(pred_tensor)
        pred_norm = torch.stack(future_pred)

        if denorm and norm_stats is not None:
            pred = self.denormalize_coords(pred_norm, norm_stats)
            return pred
        return pred_norm
    
    def denormalize_coords(self, coords_norm: torch.Tensor, stats: dict) -> torch.Tensor:
        mean_x = torch.tensor(stats["mean_x"], dtype=coords_norm.dtype).to(coords_norm.device)
        mean_y = torch.tensor(stats["mean_y"], dtype=coords_norm.dtype).to(coords_norm.device)
        std_x = torch.tensor(stats["std_x"], dtype=coords_norm.dtype).to(coords_norm.device)
        std_y = torch.tensor(stats["std_y"], dtype=coords_norm.dtype).to(coords_norm.device)

        coords = coords_norm.clone()
        coords[:, 0] = coords[:, 0] * std_x + mean_x
        coords[:, 1] = coords[:, 1] * std_y + mean_y
        return coords

# ===================== 辅助函数 =====================
def normalize_coords(coords: torch.Tensor, stats: dict) -> torch.Tensor:
    """归一化坐标"""
    mean_x = torch.tensor(stats["mean_x"], dtype=coords.dtype).to(coords.device)
    mean_y = torch.tensor(stats["mean_y"], dtype=coords.dtype).to(coords.device)
    std_x = torch.tensor(stats["std_x"], dtype=coords.dtype).to(coords.device)
    std_y = torch.tensor(stats["std_y"], dtype=coords.dtype).to(coords.device)

    coords_norm = coords.clone()
    coords_norm[:, 0] = (coords_norm[:, 0] - mean_x) / std_x
    coords_norm[:, 1] = (coords_norm[:, 1] - mean_y) / std_y
    return coords_norm

def load_norm_stats(stats_path: str) -> dict:
    """加载归一化统计量"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"归一化统计量文件不存在：{stats_path}")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

# ===================== 模型加载 =====================
def load_model_service(model_path: str, stats_path: str):
    """服务启动时加载模型和统计量"""
    global device, model, norm_stats
    
    try:
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ 使用设备: {device}")
        
        # 加载归一化统计量
        norm_stats = load_norm_stats(stats_path)
        print(f"📊 加载归一化统计量: mean_x={norm_stats['mean_x']:.2f}, mean_y={norm_stats['mean_y']:.2f}")
        
        # 加载模型
        model = KFDeepLearningModel()
        model = model.to(device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        print(f"✅ 模型加载成功 | 训练最优验证损失：{checkpoint.get('best_val_loss', 'N/A'):.6f}")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return False

# ===================== 数据处理 =====================
def validate_frame_data(frame_data_list: List[Dict]) -> bool:
    """验证输入数据格式"""
    required_fields = ['frame_id', 'x', 'y']
    
    for frame_dict in frame_data_list:
        # 检查必需字段
        for field in required_fields:
            if field not in frame_dict:
                return False, f"缺少字段: {field}"
        
        # 检查数据类型
        try:
            int(frame_dict['frame_id'])
            float(frame_dict['x'])
            float(frame_dict['y'])
        except ValueError:
            return False, "字段类型错误: frame_id应为整数, x,y应为浮点数"
    
    return True, ""

def preprocess_history(frame_data_list: List[Dict]) -> torch.Tensor:
    """预处理历史数据，提取5帧坐标"""
    global norm_stats
    
    # 按帧号排序
    sorted_frames = sorted(frame_data_list, key=lambda x: x['frame_id'])
    
    # 提取最后5帧（或全部如果不足5帧）
    if len(sorted_frames) >= 5:
        history_frames = sorted_frames[-5:]
    else:
        print(f"⚠️  历史帧数不足5帧（实际{len(sorted_frames)}帧），使用所有可用帧")
        history_frames = sorted_frames
    
    # 提取坐标
    coords = []
    for frame in history_frames:
        coords.append([frame['x'], frame['y']])
    
    # 转换为tensor并归一化
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    coords_norm = normalize_coords(coords_tensor, norm_stats)
    
    return coords_norm, history_frames

# ===================== 预测接口 =====================
@app.route('/predict', methods=['POST'])
def predict():
    """预测接口
    请求体格式:
    {
        "frame_data": [
            {
                "frame_id": int,      # 帧号
                "x": float,           # x坐标
                "y": float            # y坐标
            },
            ...
        ],
        "seg_name": str               # 可选的片段名称（用于记录）
    }
    """
    global model, device, norm_stats
    
    try:
        # 解析请求数据
        data = request.json
        if not data or "frame_data" not in data:
            return jsonify({"error": "缺少frame_data字段"}), 400
        
        frame_data_list = data["frame_data"]
        seg_name = data.get("seg_name", "unknown")
        
        # 验证数据格式
        is_valid, error_msg = validate_frame_data(frame_data_list)
        if not is_valid:
            return jsonify({"error": f"数据格式错误: {error_msg}"}), 400
        
        # 检查数据量
        if len(frame_data_list) < 1:
            return jsonify({"error": "frame_data不能为空"}), 400
        
        # 预处理历史数据
        try:
            history_norm, history_frames = preprocess_history(frame_data_list)
            history_norm = history_norm.to(device)
        except Exception as e:
            return jsonify({"error": f"数据预处理失败: {str(e)}"}), 400
        
        # 获取当前最大帧号
        current_frame_id = max(frame['frame_id'] for frame in frame_data_list)
        
        # 模型预测
        with torch.no_grad():
            future_pred = model(history_norm, norm_stats=norm_stats, denorm=True)
            future_pred_np = future_pred.cpu().numpy()
        
        # 生成未来帧号（从当前帧+1开始）
        future_frame_ids = [current_frame_id + 1 + i for i in range(3)]
        
        # 准备历史数据信息
        history_info = []
        for i, frame in enumerate(history_frames):
            history_info.append({
                "frame_id": frame['frame_id'],
                "x": round(float(frame['x']), 2),
                "y": round(float(frame['y']), 2)
            })
        
        # 准备预测结果
        predictions = []
        for i in range(3):
            predictions.append({
                "future_frame_id": future_frame_ids[i],
                "x": round(float(future_pred_np[i, 0]), 2),
                "y": round(float(future_pred_np[i, 1]), 2)
            })
        
        # 构建响应
        response = {
            "status": "success",
            "seg_name": seg_name,
            "current_frame_id": current_frame_id,
            "history_frames_used": len(history_frames),
            "history": history_info,
            "predictions": predictions,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_info": {
                "history_frames": 5,
                "future_frames": 3,
                "device": str(device)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global model, norm_stats
    
    if model is None or norm_stats is None:
        return jsonify({"status": "unhealthy", "message": "模型未加载"}), 503
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "stats_loaded": norm_stats is not None,
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    global model, norm_stats
    
    if model is None or norm_stats is None:
        return jsonify({"error": "模型未加载"}), 503
    
    info = {
        "model_type": "KFDeepLearningModel",
        "input_frames": 5,
        "output_frames": 3,
        "normalization_stats": {
            "mean_x": norm_stats.get("mean_x"),
            "mean_y": norm_stats.get("mean_y"),
            "std_x": norm_stats.get("std_x"),
            "std_y": norm_stats.get("std_y")
        },
        "device": str(device)
    }
    
    return jsonify(info)

# ===================== 服务启动 =====================
if __name__ == "__main__":
    # 配置文件路径 - 请根据实际情况修改
    CONFIG = {
        "model_path": "./trained_kf_model.pth",      # 你的模型文件路径
        "stats_path": "./norm_stats.json",           # 归一化统计量文件路径
        "host": "0.0.0.0",
        "port": 8000,                                # 服务端口
        "debug": False
    }
    
    # 加载模型
    print("=" * 50)
    print("🚀 开始加载模型...")
    load_success = load_model_service(CONFIG["model_path"], CONFIG["stats_path"])
    
    if not load_success:
        print("❌ 模型加载失败，服务退出")
        exit(1)
    
    print("=" * 50)
    print(f"✅ 模型服务准备就绪")
    print(f"📡 服务地址: http://{CONFIG['host']}:{CONFIG['port']}")
    print(f"📌 预测接口: POST http://{CONFIG['host']}:{CONFIG['port']}/predict")
    print(f"📌 健康检查: GET  http://{CONFIG['host']}:{CONFIG['port']}/health")
    print(f"📌 模型信息: GET  http://{CONFIG['host']}:{CONFIG['port']}/model_info")
    print("=" * 50)
    
    # 启动Flask服务
    app.run(
        host=CONFIG["host"],
        port=CONFIG["port"],
        debug=CONFIG["debug"],
        threaded=True
    )