"""BIC1 tandem 对照：参数损失与冻结前向代理的光谱损失共同训练逆向网络。"""

import argparse
import csv
import json
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BIC1.train_inverse import evaluate, load_forward
from common import get_device, load_config, load_npz, make_run_dir, set_seed, write_run_readme
from inverse import InverseMLP


def tandem_loss(inverse, forward, spectrum, target_params,
                param_weight: float, spec_weight: float):
    predicted_params = inverse(spectrum)
    param_loss = nn.functional.mse_loss(predicted_params, target_params)
    predicted_spectrum = forward(predicted_params)
    spec_loss = nn.functional.mse_loss(predicted_spectrum, spectrum)
    return param_weight * param_loss + spec_weight * spec_loss, param_loss, spec_loss


def epoch_losses(model, forward, loader, device, weights, optimizer=None):
    totals = np.zeros(3, dtype=np.float64)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for spectrum, target in loader:
            spectrum, target = spectrum.to(device), target.to(device)
            losses = tandem_loss(model, forward, spectrum, target, **weights)
            if optimizer is not None:
                optimizer.zero_grad()
                losses[0].backward()
                optimizer.step()
            totals += np.array([loss.item() for loss in losses]) * len(spectrum)
    return totals / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(CODE_DIR, "BIC1", "tandem", "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device()
    data = load_npz(cfg["data"]["npz_path"])
    if data["spectra"].shape[1] != 1101 or data["params_norm"].shape[1] != 6:
        raise ValueError("预处理数据应为 1101 点光谱和六列参数")
    if not np.isin(data["params_raw"][:, 5], (5, 10, 15, 20)).all():
        raise ValueError("BIC1 的 Y 必须属于 5/10/15/20")

    spectra = torch.from_numpy(data["spectra"])
    params = torch.from_numpy(data["params_norm"])
    train = TensorDataset(spectra[data["idx_train"]], params[data["idx_train"]])
    val = TensorDataset(spectra[data["idx_val"]], params[data["idx_val"]])
    train_loader = DataLoader(train, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg["train"]["batch_size"])

    inverse = InverseMLP(**cfg["model"]).to(device)
    forward = load_forward(cfg, device)
    forward.requires_grad_(False)
    forward.eval()
    optimizer = torch.optim.AdamW(
        inverse.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["max_epochs"])
    weights = cfg["loss"]
    run_dir = make_run_dir(cfg["output"]["runs_dir"], "tandem", args.config)
    print(f"设备: {device} | train={len(train)} val={len(val)} | 输出: {run_dir}", flush=True)

    best = float("inf")
    history = []
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        inverse.train()
        train_losses = epoch_losses(inverse, forward, train_loader, device, weights, optimizer)
        scheduler.step()
        inverse.eval()
        val_losses = epoch_losses(inverse, forward, val_loader, device, weights)
        history.append((epoch, *train_losses, *val_losses))
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train {train_losses[0]:.6f} "
                  f"(param {train_losses[1]:.6f}, spec {train_losses[2]:.6f}) | "
                  f"val {val_losses[0]:.6f} "
                  f"(param {val_losses[1]:.6f}, spec {val_losses[2]:.6f})", flush=True)
        if val_losses[0] < best:
            best = float(val_losses[0])
            best_epoch = epoch
            torch.save(inverse.state_dict(), os.path.join(run_dir, "best.pth"))

    with open(os.path.join(run_dir, "losses.csv"), "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("epoch", "train_total", "train_param", "train_spec",
                         "val_total", "val_param", "val_spec"))
        writer.writerows(history)
    plt.plot([row[0] for row in history], [row[1] for row in history], label="train total")
    plt.plot([row[0] for row in history], [row[4] for row in history], label="val total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE sum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    inverse.load_state_dict(torch.load(
        os.path.join(run_dir, "best.pth"), map_location=device, weights_only=True))
    idx = data["idx_test"]
    metrics = evaluate(
        inverse, forward, spectra[idx], params[idx],
        torch.from_numpy(data["param_min"]),
        torch.from_numpy(data["param_max"]), device)
    metrics.update(best_val_total=best, best_epoch=best_epoch, n_test=len(idx),
                   forward_run=cfg["evaluation"]["forward_run"], loss_weights=weights)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as out:
        json.dump(metrics, out, indent=2, ensure_ascii=False)
    write_run_readme(run_dir)
    print("测试结果: " + json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"完成。产物在 {run_dir}", flush=True)


if __name__ == "__main__":
    main()
