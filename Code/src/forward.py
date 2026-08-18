"""前向网络：结构参数(6) → S21 透射谱(1101)

设计依据 spec §4：残差 MLP，~90 万参数，Sigmoid 输出匹配谱值域 [0,1]。
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


class ForwardMLP(nn.Module):
    def __init__(self, in_dim: int = 6, out_dim: int = 1101,
                 width: int = 512, n_blocks: int = 3):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.Linear(width, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


if __name__ == "__main__":
    m = ForwardMLP()
    n_params = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(4, 6))
    print(f"参数量: {n_params / 1e6:.2f}M | 输出形状: {tuple(y.shape)} | 值域: [{y.min():.3f}, {y.max():.3f}]")
