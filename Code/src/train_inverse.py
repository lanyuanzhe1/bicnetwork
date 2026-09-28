"""训练逆向网络的简单基线（谱 → 参数）

用法（在 Code/ 目录下）：
    python src/train_inverse.py --config configs/inverse_mlp.yaml



产物写入 runs/inverse_<时间戳>/：best.pth、losses.csv、loss_curve.png、metrics.json、config.yaml
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import CODE_DIR, get_device, load_config, load_npz, make_run_dir, set_seed
from forward import ForwardMLP
from inverse import build_model


def latest_run(pattern: str) -> str:
    runs = sorted(glob.glob(os.path.join(CODE_DIR, "runs", pattern)))
    if not runs:
        raise FileNotFoundError(f"runs/ 下没有找到 {pattern}，请先训练前向网络")
    return runs[-1]


def load_forward(device) -> ForwardMLP:
    """加载最新的前向模型（闭环评估要用）"""
    run = latest_run("forward_*")
    cfg = load_config(os.path.join(run, "config.yaml"))
    fwd = ForwardMLP(**cfg["model"]).to(device)
    fwd.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=device))
    fwd.eval()
    print(f"闭环用的前向模型: {run}")
    return fwd


@torch.no_grad()
def evaluate(model, fwd, X, Y, freq, data, device):
    """测试集评估：参数误差 + 网格命中 + 闭环谱误差"""
    model.eval()
    preds = []
    for i in range(0, len(Y), 256):
        preds.append(model(Y[i:i + 256].to(device)).cpu())
    P = torch.cat(preds)                                   # (N,6) 归一化参数

    param_mse = ((P - X) ** 2).mean().item()
    param_mae = (P - X).abs().mean().item()
    per_col_mae = (P - X).abs().mean(dim=0).numpy()

    # 闭环：预测参数 → 前向模型 → 谱
    spec_pred = []
    for i in range(0, len(P), 256):
        spec_pred.append(fwd(P[i:i + 256].to(device)).cpu())
    S = torch.cat(spec_pred)                               # (N,1101)
    loop_mse = ((S - Y) ** 2).mean(dim=1).numpy()

    # 谐振谷频率误差
    df = float(freq[1] - freq[0])
    freq_err = (S.argmin(dim=1) - Y.argmin(dim=1)).abs().numpy() * df

    # 网格命中率：反归一化 → 吸附到最近网格水平 → 6/6 全中
    p_min, p_max = data["param_min"], data["param_max"]
    levels = [sorted(set(np.round(data["params_raw"][:, j], 6).tolist())) for j in range(6)]
    true_raw = X.numpy() * (p_max - p_min) + p_min
    pred_raw = P.numpy() * (p_max - p_min) + p_min
    hits = 0
    for i in range(len(pred_raw)):
        snap = np.array([min(lv, key=lambda v: abs(v - pred_raw[i][j]))
                         for j, lv in enumerate(levels)])
        hits += int(np.array_equal(snap, true_raw[i]))

    return {
        "param_mse_norm": param_mse,
        "param_mae_norm": param_mae,
        "param_mae_per_col": [float(v) for v in per_col_mae],
        "grid_hit_rate": hits / len(pred_raw),
        "loop_spec_mse_median": float(np.median(loop_mse)),
        "loop_spec_mse_mean": float(loop_mse.mean()),
        "loop_spec_mse_p90": float(np.percentile(loop_mse, 90)),
        "resonance_freq_err_thz_median": float(np.median(freq_err)),
    }, P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"设备: {device} | 模型: {cfg['model']['name']}")

    data = load_npz(cfg["data"]["npz_path"])
    freq = data["freq_grid"]
    X = torch.from_numpy(data["params_norm"])
    Y = torch.from_numpy(data["spectra"])
    train_ds = TensorDataset(Y[data["idx_train"]], X[data["idx_train"]])   # 注意：输入谱，标签参数
    val_ds = TensorDataset(Y[data["idx_val"]], X[data["idx_val"]])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])
    print(f"train={len(train_ds)}, val={len(val_ds)}")

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.3f}M")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["max_epochs"])

    # 目录名带宽度后缀：inverse_mlp512 / inverse_mlp1024，避免两种 MLP run 混在一起
    tag = cfg["model"]["name"]
    if "width" in cfg["model"]:
        tag += str(cfg["model"]["width"])
    run_dir = make_run_dir(cfg["output"]["runs_dir"], f"inverse_{tag}", args.config)
    print(f"运行目录: {run_dir}")

    best_val, patience_left, history = float("inf"), cfg["train"]["patience"], []
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for sb, xb in train_loader:
            sb, xb = sb.to(device), xb.to(device)
            loss = criterion(model(sb), xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(sb)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sb, xb in val_loader:
                sb, xb = sb.to(device), xb.to(device)
                val_loss += criterion(model(sb), xb).item() * len(sb)
        val_loss /= len(val_ds)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - 1e-7:
            best_val, patience_left = val_loss, cfg["train"]["patience"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pth"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping @ epoch {epoch}, best val {best_val:.6f}")
                break

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "losses.csv"), index=False)
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot([h["train_loss"] for h in history], label="train")
    plt.plot([h["val_loss"] for h in history], label="val")
    plt.xlabel("Epoch"); plt.ylabel("param MSE"); plt.yscale("log"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss_curve.png")); plt.close()

    # 测试集评估（含闭环）
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=device))
    fwd = load_forward(device)
    idx = data["idx_test"]
    metrics, P, S = evaluate(model, fwd, X[idx], Y[idx], freq, data, device)
    metrics["model"] = cfg["model"]["name"]
    metrics["n_params"] = n_params
    metrics["best_val_loss"] = best_val
    print("测试集: " + json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 示例图：目标谱 vs 闭环重构谱
    Yt = Y[idx]
    rng = np.random.RandomState(cfg["train"]["seed"])
    picks = rng.choice(len(Yt), size=4, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for ax, i in zip(axes.flat, picks):
        ax.plot(freq, Yt[i].numpy(), "b-", lw=1.2, label="target")
        ax.plot(freq, S[i], "r--", lw=1.2, label="loop-reconstructed")
        ax.set_xlabel("Frequency / THz"); ax.set_ylabel("S21 amplitude"); ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle(f"Inverse ({cfg['model']['name']}): target vs closed-loop spectrum")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "examples.png")); plt.close()

    print(f"完成。产物在 {run_dir}")


if __name__ == "__main__":
    main()
