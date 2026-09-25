# -*- coding: utf-8 -*-
"""汇总 runs/inverse_*/metrics.json + config.yaml → 逆向基线对比表

用法（在 Code/ 目录下）：
    python tools/summarize_inverse.py

只统计有 metrics.json 的 run（训练中途崩溃的 run 自动跳过）。
历史 run 目录名不带宽度后缀（inverse_mlp_*），以 config.yaml 里的 width 为准。
"""
import glob
import json
import os

import yaml

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_NAMES = ["Px", "Py", "A", "B", "L", "Y"]

rows = []
for f in sorted(glob.glob(os.path.join(CODE, "runs/inverse_*/metrics.json"))):
    run_dir = os.path.dirname(f)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        continue
    d = json.load(open(f, encoding="utf-8"))
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    m = cfg["model"]
    tag = m["name"] + (str(m["width"]) if "width" in m else "")
    rows.append((tag, os.path.basename(run_dir), d))

print(f"{'模型':<14}{'参数量':>9}{'参数MSE':>10}{'网格命中':>10}{'闭环谱MSE':>12}{'频率误差THz':>12}")
print("-" * 70)
for tag, run, d in rows:
    print(f"{tag:<14}{d['n_params'] / 1e6:>8.2f}M{d['param_mse_norm']:>10.4f}"
          f"{d['grid_hit_rate']:>10.1%}{d['loop_spec_mse_median']:>12.5f}"
          f"{d['resonance_freq_err_thz_median']:>12.5f}")

print("\n各参数 MAE（归一化，越小越好）")
print(f"{'模型':<14}" + "".join(f"{n:>9}" for n in PARAM_NAMES))
print("-" * 70)
for tag, run, d in rows:
    if "param_mae_per_col" in d:
        print(f"{tag:<14}" + "".join(f"{v:>9.4f}" for v in d["param_mae_per_col"]))

print("\nrun 目录：")
for tag, run, d in rows:
    print(f"  {tag:<14}{run}")

# 参照系（历史数字，来自不同 run 时需手工核对）
print("\n参照：cVAE best-of-50（旧 baseline，runs/eval_20260819_150938）")
print(f"{'cvae(best50)':<14}{'—':>9}{'—':>10}{0.2779:>10.1%}{0.00429:>12.5f}{'—':>12}")
print(f"{'前向模型误差下限':<14}{'—':>9}{'—':>10}{'—':>10}{0.00299:>12.5f}{0.00509:>12.5f}")
