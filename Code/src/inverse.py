"""逆向网络的简单基线：谱(1101) → 参数(6)

三种结构，均输出 Sigmoid（落在归一化参数区间 [0,1]）：
- InverseMLP      残差 MLP，对标 ForwardMLP 的反向
- InverseCNN      1D 卷积逐层下采样 + 全连接头，显式利用谱的局部结构
- InverseLinear   单层线性，作为容量下界参照

风格对齐 forward.py：残差块 + LayerNorm + GELU。
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )

    def forward(self, x):
        return x + self.block(x)


class InverseMLP(nn.Module):
    """残差 MLP：1101 → width → [ResidualBlock × n_blocks] → 6"""

    def __init__(self, in_dim: int = 1101, out_dim: int = 6,
                 width: int = 512, n_blocks: int = 3):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.Linear(width, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class InverseCNN(nn.Module):
    """1D-CNN：Conv1d(stride=2) 逐层下采样 → 自适应池化 → 全连接头

    谱是 1101 点的一维序列，相邻频点强相关；卷积核显式建模这种局部结构，
    这是MLP第一层做不到的（它把每个频点当独立特征）。
    """

    def __init__(self, in_dim: int = 1101, out_dim: int = 6,
                 channels=(32, 64, 128), kernel: int = 7,
                 pool: int = 2, fc: int = 256, dropout: float = 0.1):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel, stride=2, padding=kernel // 2),
                nn.BatchNorm1d(c_out), nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        # 逐层算出卷积后的长度，用固定核 MaxPool 下采样。
        # 不用 AdaptiveAvgPool1d：MPS 后端不支持非整除的输入尺寸（会直接抛错）。
        length = in_dim
        for _ in channels:
            length = (length + 2 * (kernel // 2) - kernel) // 2 + 1
        length //= pool
        self.pool = nn.MaxPool1d(pool)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c_in * length, fc), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc, out_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:                      # (B, 1101) → (B, 1, 1101)
            x = x.unsqueeze(1)
        return self.head(self.pool(self.conv(x)))


class InverseLinear(nn.Module):
    """单层线性映射：容量下界参照"""

    def __init__(self, in_dim: int = 1101, out_dim: int = 6):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.out(x)


MODELS = {
    "mlp": InverseMLP,
    "cnn": InverseCNN,
    "linear": InverseLinear,
}


def build_model(cfg_model: dict) -> nn.Module:
    """按配置里的 name 字段构造模型"""
    cfg_model = dict(cfg_model)
    name = cfg_model.pop("name")
    return MODELS[name](**cfg_model)


if __name__ == "__main__":
    spec = torch.rand(4, 1101)
    for name, cls in MODELS.items():
        m = cls()
        n = sum(p.numel() for p in m.parameters())
        y = m(spec)
        print(f"{name:8s} 参数量 {n / 1e6:6.3f}M | 输出 {tuple(y.shape)} | 值域 [{y.min():.3f}, {y.max():.3f}]")
