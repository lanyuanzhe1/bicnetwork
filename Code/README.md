# BIC 超表面 前向 + cVAE 逆向设计管线

设计文档：`../docs/superpowers/specs/2026-08-18-bic-forward-inverse-pipeline-design.md`

- **前向网络**：结构参数 (6) → S21 透射谱 (1101 点, 0.5–1.3 THz)，残差 MLP
- **逆向网络**：cVAE，以谱为条件生成参数；评估时 best-of-50 采样 + 前向网络筛选

## 环境

```bash
conda activate bic        # python 3.11 + torch 2.13 (MPS) + numpy/pandas/matplotlib/pyyaml
```

## 运行顺序（在 `Code/` 目录下）

```bash
python src/data.py            # 1. 数据预处理 → data/processed/data.npz（只需跑一次）
python src/train_forward.py   # 2. 训练前向网络 → runs/forward_<时间戳>/
python src/train_cvae.py      # 3. 训练 cVAE → runs/cvae_<时间戳>/
python src/evaluate.py        # 4. 测试集评估 + 出图 → runs/eval_<时间戳>/
```

超参数在 `configs/*.yaml` 中调整；`runs/` 已 gitignore。
