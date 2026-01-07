import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PhysicsAwareTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim: int = 12,
                 output_dim: int = 6,
                 d_model: int = 512,
                 nhead: int = 16,
                 num_layers: int = 8,
                 conv_channels: int = 256,
                 dropout: float = 0.2,
                 max_acceleration: float = 100.0,
                 max_velocity_change: float = 50.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_acceleration = max_acceleration
        self.max_velocity_change = max_velocity_change


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


        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)


        self.fc_position = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )


        self.fc_velocity = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )


        self.history_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def compute_physics_constraints(self, pred_positions, history_features, dt: float = 1.0):

        batch_size = pred_positions.shape[0]


        pred_positions_reshaped = pred_positions.view(batch_size, 3, 2)


        pred_velocities = torch.zeros(batch_size, 3, 2, device=pred_positions.device)
        pred_velocities[:, 1:, :] = pred_positions_reshaped[:, 1:, :] - pred_positions_reshaped[:, :-1, :]

        pred_accelerations = torch.zeros(batch_size, 3, 2, device=pred_positions.device)
        if pred_velocities.shape[1] > 1:
            pred_accelerations[:, 1:, :] = pred_velocities[:, 1:, :] - pred_velocities[:, :-1, :]


        smoothed_positions = pred_positions_reshaped.clone()
        for i in range(1, 3):
            smoothed_positions[:, i, :] = 0.7 * pred_positions_reshaped[:, i, :] + 0.3 * pred_positions_reshaped[:,
                                                                                         i - 1, :]

        return smoothed_positions.view(batch_size, -1), {
            'velocities': pred_velocities,
            'accelerations': pred_accelerations
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]


        history_last_3 = x[:, -3:, :].reshape(batch_size, -1)
        history_features = self.history_encoder(history_last_3)


        x_conv = x.permute(0, 2, 1)
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = x_conv.permute(0, 2, 1)
        x_conv = self.conv_norm(x_conv)

        x_transformer = self.transformer_encoder(x_conv)
        x_pool = x_transformer.mean(dim=1)


        pred_raw = self.fc_position(x_pool)

        pred_constrained, physics_info = self.compute_physics_constraints(pred_raw, history_features)


        pred_velocity = self.fc_velocity(x_pool)

        return pred_constrained, pred_raw, pred_velocity, physics_info


class TrajectoryPredictor(nn.Module):
    """兼容原有接口的包装类"""

    def __init__(self, input_dim: int = 12,
                 output_dim: int = 6,
                 d_model: int = 512,
                 nhead: int = 16,
                 num_layers: int = 8,
                 conv_channels: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.physics_model = PhysicsAwareTrajectoryPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            conv_channels=conv_channels,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_constrained, _, _, _ = self.physics_model(x)
        return pred_constrained


class EnhancedKalmanFilter:
    """增强的卡尔曼滤波器，包含加速度估计"""

    def __init__(self, dt: float = 1.0,
                 u_x: float = 0.0, u_y: float = 0.0,
                 std_acc: float = 1.0,
                 std_meas: float = 0.1,
                 max_acceleration: float = 100.0):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.x = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])  # [x, vx, ax, y, vy, ay]
        self.max_acceleration = max_acceleration


        self.F = np.array([
            [1, dt, 0.5 * dt ** 2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt ** 2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])


        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 1]
        ])


        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])


        self.Q = np.eye(6) * std_acc ** 2
        self.Q[2, 2] = 0.1 * std_acc ** 2
        self.Q[5, 5] = 0.1 * std_acc ** 2


        self.R = np.eye(2) * std_meas ** 2
        self.P = np.eye(6)

    def init_state(self, pos: np.ndarray, vel: np.ndarray = None, acc: np.ndarray = None):
        if vel is None:
            vel = np.array([0.0, 0.0])
        if acc is None:
            acc = np.array([0.0, 0.0])
        self.x = np.array([[pos[0]], [vel[0]], [acc[0]], [pos[1]], [vel[1]], [acc[1]]])

    def predict(self):
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

        # 加速度约束
        acc_x, acc_y = self.x[2, 0], self.x[5, 0]
        acc_norm = np.sqrt(acc_x ** 2 + acc_y ** 2)
        if acc_norm > self.max_acceleration:
            scale = self.max_acceleration / acc_norm
            self.x[2, 0] *= scale
            self.x[5, 0] *= scale

        return self.x[[0, 3], :].flatten()

    def update(self, z: np.ndarray):
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P
        return self.x[[0, 3], :].flatten()

    def get_velocity(self):
        return np.array([self.x[1, 0], self.x[4, 0]])

    def get_acceleration(self):
        return np.array([self.x[2, 0], self.x[5, 0]])