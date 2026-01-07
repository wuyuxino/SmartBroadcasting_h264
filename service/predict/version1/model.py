import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim: int = 8,
                 output_dim: int = 6,  # 3帧×2坐标
                 d_model: int = 512,
                 nhead: int = 16,
                 num_layers: int = 8,
                 conv_channels: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # 卷积特征提取
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=d_model,
            kernel_size=3,
            padding=1
        )
        self.conv_norm = nn.LayerNorm(d_model)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层（预测3帧）
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        x_conv = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x_conv = F.relu(self.conv1(x_conv))  # (batch, conv_channels, seq_len)
        x_conv = F.relu(self.conv2(x_conv))  # (batch, d_model, seq_len)
        x_conv = x_conv.permute(0, 2, 1)  # (batch, seq_len, d_model)
        x_conv = self.conv_norm(x_conv)

        x_transformer = self.transformer_encoder(x_conv)  # (batch, seq_len, d_model)
        x_pool = x_transformer.mean(dim=1)  # 全局池化 (batch, d_model)
        pred = self.fc(x_pool)  # (batch, 6)

        return pred


class KalmanFilter:
    """卡尔曼滤波器（用于平滑预测结果）"""
    def __init__(self, dt: float = 1.0,
                 u_x: float = 0.0, u_y: float = 0.0,
                 std_acc: float = 1.0,
                 std_meas: float = 0.1):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])  # [x, vx, y, vy]

        self.F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0.5 * dt**2, 0],
            [dt, 0],
            [0, 0.5 * dt**2],
            [0, dt]
        ])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3, 0, 0],
            [0.5 * dt**3, dt**2, 0, 0],
            [0, 0, 0.25 * dt**4, 0.5 * dt**3],
            [0, 0, 0.5 * dt**3, dt**2]
        ]) * std_acc**2
        self.R = np.eye(2) * std_meas**2
        self.P = np.eye(4)

    def init_state(self, pos: np.ndarray):
        self.x = np.array([[pos[0]], [0.0], [pos[1]], [0.0]])

    def predict(self):
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z: np.ndarray):
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P
        return self.x[:2].flatten()
