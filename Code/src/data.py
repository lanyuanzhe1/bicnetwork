"""通用数据预处理：读取 PARA/TXT 原始文件 → 归一化 → 8:1:1 划分 → npz

用法（在 Code/ 目录下）：
    python src/data.py                      # 默认数据集在 ../dataset
    python src/data.py --dataset-dir /path/to/dataset

设计依据：docs/superpowers/specs/2026-08-18-bic-forward-inverse-pipeline-design.md §3
- 默认全量模式跳过缺失谱文件的 id（2532、2645）
- 变化的参数列 min-max 归一化到 [0,1]；常量列保持原值；谱不做缩放
- 随机 8:1:1，固定种子 42，划分索引随 npz 固化
"""

import argparse
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(CODE_DIR, "..", "dataset")
DEFAULT_OUT = os.path.join(CODE_DIR, "data", "processed", "data.npz")

N_FREQ = 1101          # 每条谱的采样点数
SKIP_ROWS = 2          # TXT 谱文件的头部行数（1 行表头 + 1 行分隔线）
SEED = 42


def read_with_retry(path: str, attempts: int = 6, **loadtxt_kwargs) -> np.ndarray:
    """np.loadtxt 的 OneDrive 容错版：占位文件首次读取会超时，重试等待其完成下载"""
    for k in range(attempts):
        try:
            return np.loadtxt(path, **loadtxt_kwargs)
        except (TimeoutError, OSError) as e:
            if k == attempts - 1:
                raise RuntimeError(
                    f"读取 {path} 多次超时。请在 Finder 中确认 dataset 已完全下载到本地"
                    "（右键 → 始终保留在此设备上）") from e
            wait = 5 * (k + 1)
            print(f"  {os.path.basename(path)} 读取超时（疑似 OneDrive 占位），{wait}s 后重试 [{k + 1}/{attempts}]",
                  flush=True)
            time.sleep(wait)


def load_dataset(dataset_dir: str, first_id: int = 1, last_id: int = 3840,
                 strict: bool = False):
    """读取指定 ID 范围，返回 ids, params_raw (N,6), spectra (N,1101), freq_grid (1101,)"""
    para_dir = os.path.join(dataset_dir, "PARA")
    txt_dir = os.path.join(dataset_dir, "TXT")

    ids, params, spectra = [], [], []
    freq_grid = None
    skipped = []

    for i in range(first_id, last_id + 1):
        para_path = os.path.join(para_dir, f"para{i}.txt")
        spec_path = os.path.join(txt_dir, f"{i}.txt")
        if not (os.path.exists(para_path) and os.path.exists(spec_path)):
            if strict:
                raise FileNotFoundError(f"样本 {i} 缺少 PARA 或 TXT: {para_path}, {spec_path}")
            skipped.append(i)
            continue

        p = np.atleast_1d(read_with_retry(para_path))
        if p.shape != (6,) or not np.isfinite(p).all():
            raise ValueError(f"para{i}.txt 参数形状或数值异常: {p.shape}")

        s = read_with_retry(spec_path, skiprows=SKIP_ROWS)
        if s.shape != (N_FREQ, 2) or not np.isfinite(s).all():
            raise ValueError(f"{i}.txt 光谱形状或数值异常: {s.shape}")
        if freq_grid is None:
            freq_grid = s[:, 0]
        else:
            # 容忍文件打印精度造成的末位差异
            if not np.allclose(s[:, 0], freq_grid, rtol=1e-4, atol=1e-6):
                raise ValueError(f"{i}.txt 频率网格不一致")

        ids.append(i)
        params.append(p)
        spectra.append(s[:, 1])
        if len(ids) % 500 == 0:
            print(f"  已读取 {len(ids)} 个样本...", flush=True)

    if skipped:
        print(f"跳过缺失文件的 id: {skipped}")
    if freq_grid is None:
        raise ValueError(f"{dataset_dir} 中没有可用的 PARA/TXT 配对")
    return (np.array(ids), np.array(params, dtype=np.float32),
            np.array(spectra, dtype=np.float32), freq_grid.astype(np.float32))


def normalize_params(params_raw: np.ndarray):
    """归一化变化的参数列；常量列保持原值。"""
    p_min = params_raw.min(axis=0)
    p_max = params_raw.max(axis=0)
    varying = p_max > p_min
    params_norm = params_raw.copy()
    params_norm[:, varying] = (params_raw[:, varying] - p_min[varying]) / (p_max[varying] - p_min[varying])
    return params_norm.astype(np.float32), p_min, p_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    ids, params_raw, spectra, freq_grid = load_dataset(args.dataset_dir)
    n = len(ids)
    print(f"有效样本数: {n}")
    print(f"谱值域: [{spectra.min():.4f}, {spectra.max():.4f}]")
    print(f"频率范围: [{freq_grid[0]:.4f}, {freq_grid[-1]:.4f}] THz, {len(freq_grid)} 点")

    params_norm, p_min, p_max = normalize_params(params_raw)
    print(f"参数网格水平: {[sorted(set(params_raw[:, j].tolist())) for j in range(6)]}")

    # 随机 8:1:1 划分，固定种子
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    print(f"划分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)} (seed={SEED})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=ids,
        params_raw=params_raw,
        params_norm=params_norm,
        param_min=p_min.astype(np.float32),
        param_max=p_max.astype(np.float32),
        spectra=spectra,
        freq_grid=freq_grid,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        seed=np.array([SEED]),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"已保存: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
