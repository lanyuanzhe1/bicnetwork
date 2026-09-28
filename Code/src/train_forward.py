"""训练前向网络（参数 → 谱）

用法（在 Code/ 目录下）：
    python src/train_forward.py [--config configs/forward.yaml]

产物写入配置指定的 runs 目录：best.pth、losses.csv、loss_curve.png、metrics.json、config.yaml
"""

import argparse
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

from common import get_device, load_config, load_npz, make_run_dir, set_seed
from forward import ForwardMLP


def main(config_path=None):
    if config_path is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "forward.yaml"))
        config_path = ap.parse_args().config
    cfg = load_config(config_path)

    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"设备: {device}")

    data = load_npz(cfg["data"]["npz_path"])
    X = torch.from_numpy(data["params_norm"])
    Y = torch.from_numpy(data["spectra"])
    train_ds = TensorDataset(X[data["idx_train"]], Y[data["idx_train"]])
    val_ds = TensorDataset(X[data["idx_val"]], Y[data["idx_val"]])
    test_ds = TensorDataset(X[data["idx_test"]], Y[data["idx_test"]])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])
    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = ForwardMLP(**cfg["model"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["max_epochs"])

    run_dir = make_run_dir(cfg["output"]["runs_dir"], "forward", config_path)
    print(f"运行目录: {run_dir}")

    best_val = float("inf")
    patience_left = cfg["train"]["patience"]
    history = []

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            patience_left = cfg["train"]["patience"]
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
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=device))
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            test_mse += criterion(model(xb), yb).item() * len(xb)
    test_mse /= len(test_ds)
    metrics = {"best_val_mse": best_val, "test_spec_mse": test_mse,
               "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"完成。最优验证 MSE {best_val:.6f}，测试集光谱 MSE {test_mse:.6f}，产物在 {run_dir}")


if __name__ == "__main__":
    main()
