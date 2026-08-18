"""逆向网络 cVAE：以谱 c(1101) 为条件，生成结构参数 x(6)

设计依据 spec §5：
- 编码器 q(z|x,c)：输入 [x(6) ∥ c(1101)] → 512 → 256 → μ/logσ² (各 latent_dim 维)
- 解码器 p(x|z,c)：输入 [z ∥ c] → 512 → 256 → x̂(6)，Sigmoid（归一化参数区间）
- 损失 = MSE(x, x̂) + β·KL，β 退火由训练脚本控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, param_dim: int = 6, spec_dim: int = 1101, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(param_dim + spec_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + spec_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, param_dim), nn.Sigmoid(),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    @torch.no_grad()
    def generate(self, c, n_samples: int):
        """对条件谱 c (B, spec_dim) 各采 n_samples 个候选参数 → (B, n_samples, param_dim)"""
        B = c.shape[0]
        z = torch.randn(B, n_samples, self.latent_dim, device=c.device)
        c_exp = c.unsqueeze(1).expand(B, n_samples, c.shape[-1])
        return self.decode(z.reshape(-1, self.latent_dim),
                           c_exp.reshape(-1, c.shape[-1])).reshape(B, n_samples, -1)


def cvae_loss(x, x_hat, mu, logvar, beta: float, free_bits: float = 0.0):
    """重构 MSE（batch 均值）+ β·KL。

    KL 先按 z 维度计算；free_bits > 0 时每维至少计 free_bits nats（free-bits 技巧），
    保留编码器活性、缓解后验坍塌。
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z)
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum(dim=-1).mean()
    return recon + beta * kl, recon, kl


if __name__ == "__main__":
    m = CVAE()
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.rand(4, 6)
    c = torch.rand(4, 1101)
    x_hat, mu, logvar = m(x, c)
    loss, recon, kl = cvae_loss(x, x_hat, mu, logvar, beta=0.1)
    g = m.generate(c, n_samples=50)
    print(f"参数量: {n_params / 1e6:.2f}M | 重建: {tuple(x_hat.shape)} | 生成: {tuple(g.shape)}")
    print(f"loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")
