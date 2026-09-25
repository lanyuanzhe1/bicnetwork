#!/usr/bin/env bash
# 数据集更新后的一键全量重训：数据 → 前向 → 三个逆向基线 → 汇总
#
# 用法（在 Code/ 目录下，先 conda activate bic）：
#   bash tools/run_full_retrain.sh
#   bash tools/run_full_retrain.sh --dataset-dir /path/to/new_dataset
#
# 注意：顺序不能颠倒——
#   1. data.py 重建 data.npz（划分种子 42，随 npz 固化）
#   2. 必须先重训前向：逆向的闭环评估加载的是 runs/forward_* 里最新的一个，
#      数据集换了还用旧前向，闭环数字就是错的
#   3. 三个逆向基线（mlp512 / mlp1024 / cnn）
#   4. 汇总对比表
set -euo pipefail
cd "$(dirname "$0")/.."

python src/data.py "$@"
python src/train_forward.py --config configs/forward.yaml
python src/train_inverse.py --config configs/inverse_mlp.yaml
python src/train_inverse.py --config configs/inverse_mlp_wide.yaml
python src/train_inverse.py --config configs/inverse_cnn.yaml
python tools/summarize_inverse.py
