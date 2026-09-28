"""BIC-0 原始 PARA/TXT → 六列参数与整条光谱的 npz。"""

import argparse
import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
from src.data import SEED, load_dataset, normalize_params

DEFAULT_DATASET = os.path.join(CODE_DIR, "..", "周期分类数据集", "BIC-0")
DEFAULT_OUT = os.path.join(CODE_DIR, "data", "processed", "bic0.npz")


def build_dataset(dataset_dir: str, last_id: int = 768) -> dict:
    ids, params_raw, spectra, freq_grid = load_dataset(
        dataset_dir, last_id=last_id, strict=True)
    if not np.all(params_raw[:, 5] == 0):
        raise ValueError("BIC-0 的 Y 参数必须全部为 0")
    params_norm, p_min, p_max = normalize_params(params_raw)
    n = len(ids)
    perm = np.random.RandomState(SEED).permutation(n)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return dict(
        ids=ids, params_raw=params_raw, params_norm=params_norm,
        param_min=p_min.astype(np.float32), param_max=p_max.astype(np.float32),
        spectra=spectra, freq_grid=freq_grid,
        idx_train=perm[:n_train], idx_val=perm[n_train:n_train + n_val],
        idx_test=perm[n_train + n_val:], seed=np.array([SEED]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    data = build_dataset(args.dataset_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"已保存 {len(data['ids'])} 个样本至 {args.out}")


if __name__ == "__main__":
    main()
