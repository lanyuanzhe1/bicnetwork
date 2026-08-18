"""测试集评估：前向指标 + cVAE best-of-K 逆向闭环

用法（在 Code/ 目录下）：
    python src/evaluate.py                                  # 自动用 runs/ 里最新的 forward/cvae 运行
    python src/evaluate.py --forward-run runs/forward_xxx --cvae-run runs/cvae_xxx

设计依据 spec §4.3 / §5.3：
- 前向：MSE / MAE / R² / 谐振谷频率误差 / 谷深误差
- 逆向：每条目标谱采 K 个候选 → 前向网络重构谱 → 取谱误差最小者（best-of-K）
  主指标 = 谱重构误差；辅助 = 网格命中率、候选多样性
产物写入 runs/eval_<时间戳>/：metrics.json、示例对比图
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import CODE_DIR, get_device, load_config, load_npz, set_seed
from cvae import CVAE
from forward import ForwardMLP

K_CANDIDATES = 50
N_EXAMPLES = 4


def latest_run(pattern: str) -> str:
    runs = sorted(glob.glob(os.path.join(CODE_DIR, "runs", pattern)))
    if not runs:
        raise FileNotFoundError(f"runs/ 下没有找到 {pattern}，请先训练")
    return runs[-1]


def load_models(forward_run: str, cvae_run: str, device):
    fcfg = load_config(os.path.join(forward_run, "config.yaml"))
    ccfg = load_config(os.path.join(cvae_run, "config.yaml"))
    fwd = ForwardMLP(**fcfg["model"]).to(device)
    fwd.load_state_dict(torch.load(os.path.join(forward_run, "best.pth"), map_location=device))
    cvae = CVAE(**ccfg["model"]).to(device)
    cvae.load_state_dict(torch.load(os.path.join(cvae_run, "best.pth"), map_location=device))
    fwd.eval()
    cvae.eval()
    return fwd, cvae


def eval_forward(fwd, X, Y, freq, device):
    """返回指标字典与测试集预测谱"""
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 256):
            preds.append(fwd(X[i:i + 256].to(device)).cpu())
        P = torch.cat(preds)

    mse = ((P - Y) ** 2).mean().item()
    mae = (P - Y).abs().mean().item()
    ss_res = ((P - Y) ** 2).sum().item()
    ss_tot = ((Y - Y.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot

    df = float(freq[1] - freq[0])
    idx_pred = P.argmin(dim=1)
    idx_true = Y.argmin(dim=1)
    freq_err = (idx_pred - idx_true).abs().numpy() * df          # THz
    depth_err = (P.min(dim=1).values - Y.min(dim=1).values).abs().numpy()

    metrics = {
        "mse": mse, "mae": mae, "r2": r2,
        "resonance_freq_err_thz_median": float(np.median(freq_err)),
        "resonance_freq_err_thz_p90": float(np.percentile(freq_err, 90)),
        "dip_depth_err_median": float(np.median(depth_err)),
    }
    return metrics, P


def eval_inverse(fwd, cvae, X, Y, data, device, k: int):
    """best-of-K 闭环。返回指标、best 候选参数（归一化）、best 重构谱、候选谱误差矩阵"""
    n = len(X)
    best_params = np.zeros((n, 6), dtype=np.float32)
    best_specs = np.zeros((n, Y.shape[1]), dtype=np.float32)
    best_errs = np.zeros(n, dtype=np.float32)
    all_errs = np.zeros((n, k), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, 64):
            cb = Y[i:i + 64].to(device)                          # (B,1101)
            B = len(cb)
            cand = cvae.generate(cb, k)                          # (B,K,6)
            spec_pred = fwd(cand.reshape(-1, 6)).reshape(B, k, -1)  # (B,K,1101)
            errs = ((spec_pred - cb.unsqueeze(1)) ** 2).mean(dim=-1)  # (B,K)
            best = errs.argmin(dim=1)                            # (B,)
            arange = torch.arange(B, device=device)
            best_params[i:i + B] = cand[arange, best].cpu().numpy()
            best_specs[i:i + B] = spec_pred[arange, best].cpu().numpy()
            best_errs[i:i + B] = errs[arange, best].cpu().numpy()
            all_errs[i:i + B] = errs.cpu().numpy()

    # 网格命中率：反归一化 → 四舍五入到最近网格水平 → 与真值 6/6 全中
    p_min, p_max = data["param_min"], data["param_max"]
    levels = [sorted(set(np.round(data["params_raw"][:, j], 6).tolist())) for j in range(6)]

    def snap(params_raw_row):
        return np.array([min(lv, key=lambda v: abs(v - params_raw_row[j]))
                         for j, lv in enumerate(levels)])

    true_raw = X.numpy() * (p_max - p_min) + p_min
    best_raw = best_params * (p_max - p_min) + p_min
    hits = sum(np.array_equal(snap(best_raw[i]), true_raw[i]) for i in range(n))

    # 多样性：50 个候选的谱误差展布（std），衡量生成是否坍塌到单点
    diversity = float(all_errs.std(axis=1).mean())

    metrics = {
        "k": k,
        "spectral_recon_mse_median": float(np.median(best_errs)),
        "spectral_recon_mse_mean": float(best_errs.mean()),
        "spectral_recon_mse_p90": float(np.percentile(best_errs, 90)),
        "grid_hit_rate": hits / n,
        "candidate_err_spread_mean": diversity,
    }
    return metrics, best_params, best_specs, best_errs


def plot_examples(freq, Y, P, path, title, extra_texts=None, seed=42):
    rng = np.random.RandomState(seed)
    picks = rng.choice(len(Y), size=min(N_EXAMPLES, len(Y)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    for ax, i in zip(axes.flat, picks):
        ax.plot(freq, Y[i].numpy(), "b-", lw=1.2, label="Ground truth")
        ax.plot(freq, P[i], "r--", lw=1.2, label="Prediction")
        ax.set_xlabel("Frequency / THz")
        ax.set_ylabel("S21 amplitude")
        ax.set_ylim(0, 1)
        if extra_texts:
            ax.set_title(extra_texts[i], fontsize=8)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-run", default=None)
    ap.add_argument("--cvae-run", default=None)
    ap.add_argument("--k", type=int, default=K_CANDIDATES)
    args = ap.parse_args()

    forward_run = args.forward_run or latest_run("forward_*")
    cvae_run = args.cvae_run or latest_run("cvae_*")
    print(f"前向运行: {forward_run}\n逆向运行: {cvae_run}")

    set_seed(42)
    device = get_device()
    data = load_npz("data/processed/data.npz")
    freq = data["freq_grid"]
    idx = data["idx_test"]
    X = torch.from_numpy(data["params_norm"][idx])
    Y = torch.from_numpy(data["spectra"][idx])
    print(f"测试集: {len(X)} 样本 | 设备: {device}")

    fwd, cvae = load_models(forward_run, cvae_run, device)

    # 前向评估
    fwd_metrics, P = eval_forward(fwd, X, Y, freq, device)
    print("前向: " + json.dumps(fwd_metrics, indent=2))

    # 逆向 best-of-K 闭环
    inv_metrics, best_params, best_specs, best_errs = eval_inverse(
        fwd, cvae, X, Y, data, device, args.k)
    print("逆向: " + json.dumps(inv_metrics, indent=2))

    # 运行目录与产物
    from datetime import datetime
    run_dir = os.path.join(CODE_DIR, "runs", f"eval_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"forward_run": forward_run, "cvae_run": cvae_run,
                   "forward": fwd_metrics, "inverse": inv_metrics}, f, indent=2, ensure_ascii=False)

    plot_examples(freq, Y, P, os.path.join(run_dir, "forward_examples.png"),
                  "Forward: ground truth vs predicted spectra (4 random test samples)")

    p_min, p_max = data["param_min"], data["param_max"]
    extra = {}
    for i in range(len(Y)):
        pr = best_params[i] * (p_max - p_min) + p_min
        extra[i] = f"best-of-{args.k} spec MSE={best_errs[i]:.2e} | params={np.round(pr, 1).tolist()}"
    plot_examples(freq, Y, best_specs, os.path.join(run_dir, "inverse_examples.png"),
                  f"Inverse: target vs best-of-{args.k} reconstructed spectra (4 random test samples)",
                  extra_texts=extra)

    print(f"产物在 {run_dir}")


if __name__ == "__main__":
    main()
