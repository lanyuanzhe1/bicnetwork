"""训练 cVAE 逆向网络（谱 → 参数）

用法（在 Code/ 目录下）：
    python src/train_cvae.py [--config configs/cvae.yaml]

β 退火：前 beta_warmup_epochs 从 0 线性升到 beta_max，之后固定（防 KL 坍塌）。
产物写入 runs/cvae_<时间戳>/：best.pth、losses.csv（含 recon/kl 分项）、loss_curve.png、config.yaml
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import get_device, load_config, load_npz, make_run_dir, set_seed
from cvae import CVAE, cvae_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "cvae.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"设备: {device}")

    data = load_npz(cfg["data"]["npz_path"])
    X = torch.from_numpy(data["params_norm"])   # 参数（生成目标）
    C = torch.from_numpy(data["spectra"])       # 谱（条件）
    train_ds = TensorDataset(X[data["idx_train"]], C[data["idx_train"]])
    val_ds = TensorDataset(X[data["idx_val"]], C[data["idx_val"]])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])
    print(f"train={len(train_ds)}, val={len(val_ds)}")

    model = CVAE(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["max_epochs"])

    run_dir = make_run_dir(cfg["output"]["runs_dir"], "cvae", args.config)
    print(f"运行目录: {run_dir}")

    warmup = cfg["train"]["beta_warmup_epochs"]
    beta_max = cfg["train"]["beta_max"]
    free_bits = cfg["train"].get("free_bits", 0.0)
    best_val = float("inf")
    patience_left = cfg["train"]["patience"]
    history = []

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        beta = beta_max * min(1.0, epoch / warmup)

        model.train()
        tr = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for xb, cb in train_loader:
            xb, cb = xb.to(device), cb.to(device)
            x_hat, mu, logvar = model(xb, cb)
            loss, recon, kl = cvae_loss(xb, x_hat, mu, logvar, beta, free_bits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in (("loss", loss), ("recon", recon), ("kl", kl)):
                tr[k] += v.item() * len(xb)
        scheduler.step()

        model.eval()
        va = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        with torch.no_grad():
            for xb, cb in val_loader:
                xb, cb = xb.to(device), cb.to(device)
                x_hat, mu, logvar = model(xb, cb)
                loss, recon, kl = cvae_loss(xb, x_hat, mu, logvar, beta, free_bits)
                for k, v in (("loss", loss), ("recon", recon), ("kl", kl)):
                    va[k] += v.item() * len(xb)

        row = {"epoch": epoch, "beta": beta}
        for k in tr:
            row[f"train_{k}"] = tr[k] / len(train_ds)
            row[f"val_{k}"] = va[k] / len(val_ds)
        history.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | β={beta:.3f} | train {row['train_loss']:.5f} "
                  f"(recon {row['train_recon']:.5f}, kl {row['train_kl']:.3f}) | "
                  f"val {row['val_loss']:.5f}")

        if row["val_loss"] < best_val - 1e-7:
            best_val = row["val_loss"]
            patience_left = cfg["train"]["patience"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pth"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping @ epoch {epoch}, best val {best_val:.6f}")
                break

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "losses.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    for ax, key, title in zip(axes, ["loss", "recon", "kl"], ["Total loss", "Recon MSE", "KL"]):
        ax.plot([h[f"train_{key}"] for h in history], label="train")
        ax.plot([h[f"val_{key}"] for h in history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    print(f"完成。最优验证损失 {best_val:.6f}，产物在 {run_dir}")


if __name__ == "__main__":
    main()
