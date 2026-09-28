"""BIC-0 逆向基线：整条 1101 点光谱 → 六列参数。"""

import argparse
import csv
import json
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE_DIR, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import get_device, load_config, load_npz, make_run_dir, set_seed
from forward import ForwardMLP
from inverse import InverseMLP


def bic0_forward_input(predicted_params):
    """用于 BIC-0 结构/闭环的参数副本；不修改逆向网络原始输出。"""
    params = predicted_params.clone()
    params[:, 5] = 0
    return params


def load_forward(cfg, device):
    run = cfg["evaluation"]["forward_run"]
    run = run if os.path.isabs(run) else os.path.join(CODE_DIR, run)
    forward_cfg = load_config(os.path.join(run, "config.yaml"))
    if forward_cfg["data"]["npz_path"] != cfg["data"]["npz_path"]:
        raise ValueError("前向模型与逆向训练使用的数据文件不一致")
    model = ForwardMLP(**forward_cfg["model"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, forward, spectra, params, device):
    model.eval()
    raw_predictions, reconstructions = [], []
    for start in range(0, len(spectra), 256):
        spec = spectra[start:start + 256].to(device)
        raw = model(spec)
        raw_predictions.append(raw.cpu())
        reconstructions.append(forward(bic0_forward_input(raw)).cpu())
    predicted = torch.cat(raw_predictions)
    rebuilt = torch.cat(reconstructions)
    return {
        "raw_param_mae_norm_per_column": (predicted - params).abs().mean(dim=0).tolist(),
        "bic0_loop_spec_mse": ((rebuilt - spectra) ** 2).mean().item(),
        "raw_predicted_y_norm_mean": predicted[:, 5].mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(CODE_DIR, "BIC0", "inverse.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device()
    data = load_npz(cfg["data"]["npz_path"])
    params = torch.from_numpy(data["params_norm"])
    spectra = torch.from_numpy(data["spectra"])
    if spectra.shape[1] != 1101 or params.shape[1] != 6 or not torch.all(params[:, 5] == 0):
        raise ValueError("预处理数据不是预期的 BIC-0：应为 1101 点谱、六列参数且 Y=0")

    train = TensorDataset(spectra[data["idx_train"]], params[data["idx_train"]])
    val = TensorDataset(spectra[data["idx_val"]], params[data["idx_val"]])
    train_loader = DataLoader(train, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg["train"]["batch_size"])
    model = InverseMLP(**cfg["model"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["max_epochs"])
    # 先确认评估用的是这份 BIC-0 前向模型，再启动耗时训练。
    forward = load_forward(cfg, device)
    run_dir = make_run_dir(cfg["output"]["runs_dir"], "inverse_mlp", args.config)
    print(f"设备: {device} | train={len(train)} val={len(val)} | 输出: {run_dir}")

    best, patience_left, history = float("inf"), cfg["train"]["patience"], []
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        total = 0.0
        for spec, target in train_loader:
            spec, target = spec.to(device), target.to(device)
            loss = criterion(model(spec), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(spec)
        train_loss = total / len(train)
        scheduler.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for spec, target in val_loader:
                spec, target = spec.to(device), target.to(device)
                total += criterion(model(spec), target).item() * len(spec)
        val_loss = total / len(val)
        history.append((epoch, train_loss, val_loss))
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f}")
        if val_loss < best - 1e-7:
            best, patience_left = val_loss, cfg["train"]["patience"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pth"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping @ epoch {epoch}")
                break

    with open(os.path.join(run_dir, "losses.csv"), "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("epoch", "train_loss", "val_loss"))
        writer.writerows(history)
    plt.plot([item[0] for item in history], [item[1] for item in history], label="train")
    plt.plot([item[0] for item in history], [item[2] for item in history], label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Parameter MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=device,
                                     weights_only=True))
    idx = data["idx_test"]
    metrics = evaluate(model, forward, spectra[idx], params[idx], device)
    metrics["best_val_loss"] = best
    metrics["n_test"] = len(idx)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as out:
        json.dump(metrics, out, indent=2, ensure_ascii=False)
    print("测试诊断: " + json.dumps(metrics, ensure_ascii=False))
    print(f"完成。产物在 {run_dir}")


if __name__ == "__main__":
    main()
