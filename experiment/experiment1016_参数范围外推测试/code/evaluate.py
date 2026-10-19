"""用 10 月 16 日新测试集分别评估 BIC1 前向、逆向和代理闭环。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Code" / "src"))
from forward import ForwardMLP
from inverse import InverseMLP

NAMES = ("Px", "Py", "A", "B", "L", "Y")


def load_test_set(dataset: Path):
    """严格读取编号 1–180；缺任何一项或频率轴不一致就退出。"""
    expected = set(range(1, 181))
    for folder, pattern, parse in (
        ("PARA", "para*.txt", lambda p: int(p.stem[4:])),
        ("TXT", "*.txt", lambda p: int(p.stem)),
        ("FIG", "*.bmp", lambda p: int(p.stem)),
    ):
        found = {parse(p) for p in (dataset / folder).glob(pattern)}
        if found != expected:
            raise ValueError(f"{folder} ID 不完整: 缺 {sorted(expected - found)}，多 {sorted(found - expected)}")

    params, spectra, freq = [], [], None
    for sample_id in range(1, 181):
        p = np.atleast_1d(np.loadtxt(dataset / "PARA" / f"para{sample_id}.txt"))
        s = np.loadtxt(dataset / "TXT" / f"{sample_id}.txt", skiprows=2)
        if p.shape != (6,) or s.shape != (1101, 2) or not np.isfinite(p).all() or not np.isfinite(s).all():
            raise ValueError(f"样本 {sample_id} 的参数或光谱形状/数值异常")
        if freq is None:
            freq = s[:, 0]
        elif not np.allclose(s[:, 0], freq, rtol=1e-6, atol=1e-7):
            raise ValueError(f"样本 {sample_id} 的频率轴不同")
        params.append(p)
        spectra.append(s[:, 1])
    params = np.asarray(params, dtype=np.float32)
    spectra = np.asarray(spectra, dtype=np.float32)
    if len(np.unique(params, axis=0)) != 180 or spectra.min() < 0 or spectra.max() > 1:
        raise ValueError("参数重复或 |S21| 超出 [0,1]")
    return np.arange(1, 181), params, spectra, freq.astype(np.float32)


def load_model(run: Path, model_class):
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    model = model_class(**cfg["model"])
    model.load_state_dict(torch.load(run / "best.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def compute_predictions(forward, inverse_models, params_raw, spectra, p_min, p_max):
    """前向只看真实参数；逆向只看真实谱；闭环只看逆向预测参数。"""
    params_norm = (params_raw - p_min) / (p_max - p_min)
    forward_spectra = forward(torch.from_numpy(params_norm)).cpu().numpy()
    inverse = {}
    for name, model in inverse_models.items():
        predicted_norm = model(torch.from_numpy(spectra)).cpu().numpy()
        inverse[name] = {
            "params_norm": predicted_norm,
            "params_raw": predicted_norm * (p_max - p_min) + p_min,
            "loop_spectra": forward(torch.from_numpy(predicted_norm)).cpu().numpy(),
        }
    return {"forward_spectra": forward_spectra, "inverse": inverse}


def spectral_metrics(prediction, truth):
    diff = prediction - truth
    per_sample_mse = np.mean(diff ** 2, axis=1)
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "median_sample_mse": float(np.median(per_sample_mse)),
        "p90_sample_mse": float(np.percentile(per_sample_mse, 90)),
    }


def save_examples(path, freq, ids, truth, predictions, forward_error, in_range):
    novel = np.flatnonzero(~in_range)
    median_novel = novel[np.argsort(forward_error[novel])[len(novel) // 2]]
    examples = [30, int(median_novel), int(novel[np.argmax(forward_error[novel])])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = {"CST test": "black", "forward": "#3274a1", "direct loop": "#e1812c", "tandem loop": "#3a923a"}
    for ax, index, title in zip(axes, examples, ("old-range overlap", "novel median forward error", "novel worst forward error")):
        for label, curves in predictions.items():
            ax.plot(freq, curves[index], label=label, color=colors[label], linewidth=1.1)
        ax.set_title(f"ID {ids[index]} — {title}")
        ax.set_ylabel("|S21|")
        ax.set_ylim(0, 1.02)
    axes[-1].set_xlabel("Frequency (THz)")
    axes[0].legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [int(ids[index]) for index in examples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "data_1016_for_test")
    parser.add_argument("--out", type=Path, default=EXPERIMENT / "results")
    parser.add_argument("--forward-run", type=Path,
                        default=REPO / "Code/runs/bic1/forward_20260929_101011")
    parser.add_argument("--direct-run", type=Path,
                        default=REPO / "Code/runs/bic1/inverse_mlp_20260929_201411")
    parser.add_argument("--tandem-run", type=Path,
                        default=REPO / "Code/runs/bic1/tandem/tandem_20260929_233231")
    args = parser.parse_args()
    ids, params, spectra, freq = load_test_set(args.dataset)

    old = np.load(REPO / "Code/data/processed/bic1.npz")
    p_min, p_max = old["param_min"], old["param_max"]
    if not np.allclose(freq, old["freq_grid"], atol=1e-7):
        raise ValueError("新旧数据的频率轴不同")
    forward_run = args.forward_run.resolve()
    inverse_runs = {
        "direct": args.direct_run.resolve(),
        "tandem": args.tandem_run.resolve(),
    }
    forward = load_model(forward_run, ForwardMLP)
    inverse_models = {name: load_model(run, InverseMLP) for name, run in inverse_runs.items()}
    predictions = compute_predictions(forward, inverse_models, params, spectra, p_min, p_max)

    in_range = np.all((params >= p_min) & (params <= p_max), axis=1)
    forward_spec = predictions["forward_spectra"]
    metrics = {
        "test_set": str(args.dataset.resolve()),
        "n_samples": len(ids),
        "frequency_points": len(freq),
        "old_range_overlap": int(in_range.sum()),
        "outside_old_range": int((~in_range).sum()),
        "outside_old_range_per_column": dict(zip(NAMES, np.sum((params < p_min) | (params > p_max), axis=0).tolist())),
        "model_runs": {"forward": str(forward_run.relative_to(REPO)), **{name: str(run.relative_to(REPO)) for name, run in inverse_runs.items()}},
        "forward": spectral_metrics(forward_spec, spectra),
        "inverse": {},
    }
    forward_error = np.mean((forward_spec - spectra) ** 2, axis=1)
    metrics["forward"]["overlap_mse"] = float(np.mean(forward_error[in_range]))
    metrics["forward"]["novel_mse"] = float(np.mean(forward_error[~in_range]))

    rows = []
    for i, sample_id in enumerate(ids):
        row = {"id": int(sample_id), "in_old_range": bool(in_range[i]), "forward_spec_mse": float(forward_error[i])}
        row.update({f"true_{name}": float(params[i, j]) for j, name in enumerate(NAMES)})
        rows.append(row)
    archive = {"ids": ids, "freq_thz": freq, "params_true": params, "spectra_true": spectra, "forward_spectra": forward_spec}
    for name, result in predictions["inverse"].items():
        delta = result["params_raw"] - params
        norm_truth = (params - p_min) / (p_max - p_min)
        loop = spectral_metrics(result["loop_spectra"], spectra)
        metrics["inverse"][name] = {
            "param_mse_old_normalization": float(np.mean((result["params_norm"] - norm_truth) ** 2)),
            "param_mae_raw_per_column_um": dict(zip(NAMES, np.mean(np.abs(delta), axis=0).tolist())),
            "predicted_param_range_um": {col: [float(result["params_raw"][:, j].min()), float(result["params_raw"][:, j].max())] for j, col in enumerate(NAMES)},
            "loop": loop,
        }
        archive[f"{name}_params_pred"] = result["params_raw"]
        archive[f"{name}_loop_spectra"] = result["loop_spectra"]
        loop_error = np.mean((result["loop_spectra"] - spectra) ** 2, axis=1)
        for i, row in enumerate(rows):
            row[f"{name}_param_mse_old_normalization"] = float(np.mean((result["params_norm"][i] - norm_truth[i]) ** 2))
            row[f"{name}_loop_spec_mse"] = float(loop_error[i])
            row.update({f"{name}_pred_{col}": float(result["params_raw"][i, j]) for j, col in enumerate(NAMES)})

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", **archive)
    with (args.out / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    examples = save_examples(args.out / "spectra_examples.png", freq, ids, spectra,
        {"CST test": spectra, "forward": forward_spec,
         "direct loop": predictions["inverse"]["direct"]["loop_spectra"],
         "tandem loop": predictions["inverse"]["tandem"]["loop_spectra"]},
        forward_error, in_range)
    metrics["example_ids"] = examples
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"forward_mse": metrics["forward"]["mse"],
                      "direct_loop_mse": metrics["inverse"]["direct"]["loop"]["mse"],
                      "tandem_loop_mse": metrics["inverse"]["tandem"]["loop"]["mse"],
                      "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
