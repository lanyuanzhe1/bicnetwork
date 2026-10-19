"""BIC1 原始 PARA/TXT → 六列参数与完整光谱的 npz。"""

import argparse
import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
from src.data import SEED, load_dataset, normalize_params

DEFAULT_DATASET = os.path.join(CODE_DIR, "..", "周期分类数据集", "BIC")
DEFAULT_OUT = os.path.join(CODE_DIR, "data", "processed", "bic1.npz")
FIRST_ID = 769
LAST_ID = 3840
VALID_Y = (5, 10, 15, 20)
EXPECTED_MISSING_IDS = (2532, 2645)
EXPECTED_Y_COUNTS = {5: 768, 10: 768, 15: 766, 20: 768}


def build_dataset(dataset_dir: str, first_id: int = FIRST_ID,
                  last_id: int = LAST_ID,
                  expected_missing_ids=EXPECTED_MISSING_IDS,
                  expected_y_counts=EXPECTED_Y_COUNTS) -> dict:
    para_dir = os.path.join(dataset_dir, "PARA")
    txt_dir = os.path.join(dataset_dir, "TXT")
    expected_ids = range(first_id, last_id + 1)
    missing_para = [
        sample_id for sample_id in expected_ids
        if not os.path.exists(os.path.join(para_dir, f"para{sample_id}.txt"))
    ]
    missing_txt = [
        sample_id for sample_id in expected_ids
        if not os.path.exists(os.path.join(txt_dir, f"{sample_id}.txt"))
    ]
    if missing_para:
        raise ValueError(f"BIC1 缺失 PARA: {missing_para}")
    if tuple(missing_txt) != tuple(expected_missing_ids):
        raise ValueError(
            f"BIC1 TXT 缺失 ID 应为 {list(expected_missing_ids)}，"
            f"实际为 {missing_txt}")

    ids, params_raw, spectra, freq_grid = load_dataset(
        dataset_dir, first_id=first_id, last_id=last_id, strict=False)
    if not np.isin(params_raw[:, 5], VALID_Y).all():
        found = sorted(set(params_raw[:, 5].tolist()))
        raise ValueError(f"BIC1 的 Y 参数应为 {list(VALID_Y)} 中的值，实际为 {found}")
    actual_y_counts = {
        y: int(np.count_nonzero(params_raw[:, 5] == y))
        for y in sorted(set(params_raw[:, 5].tolist()))
    }
    if actual_y_counts != dict(expected_y_counts):
        raise ValueError(
            f"BIC1 Y 分层计数应为 {dict(expected_y_counts)}，"
            f"实际为 {actual_y_counts}")

    params_norm, p_min, p_max = normalize_params(params_raw)
    n = len(ids)
    perm = np.random.RandomState(SEED).permutation(n)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return dict(
        ids=ids,
        skipped_ids=np.array(missing_txt, dtype=np.int64),
        params_raw=params_raw,
        params_norm=params_norm,
        param_min=p_min.astype(np.float32),
        param_max=p_max.astype(np.float32),
        spectra=spectra,
        freq_grid=freq_grid,
        idx_train=perm[:n_train],
        idx_val=perm[n_train:n_train + n_val],
        idx_test=perm[n_train + n_val:],
        seed=np.array([SEED]),
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
    if len(data["skipped_ids"]):
        print(f"已跳过缺失样本: {data['skipped_ids'].tolist()}")


if __name__ == "__main__":
    main()
