"""训练脚本共享工具：配置加载、种子、设备、运行目录"""

import os
import random
import shutil
import csv
import json
from datetime import datetime

import numpy as np
import torch
import yaml

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_run_dir(runs_dir: str, prefix: str, config_path: str) -> str:
    """创建 runs/<prefix>_<时间戳>/，并复制配置文件留档"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CODE_DIR, runs_dir, f"{prefix}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    write_run_readme(run_dir)
    return run_dir


def write_run_readme(run_dir: str):
    """根据当前产物更新单次训练说明；训练中断时仍保留配置。"""
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as stream:
        config_text = stream.read().rstrip()
    cfg = yaml.safe_load(config_text)
    losses_path = os.path.join(run_dir, "losses.csv")
    metrics_path = os.path.join(run_dir, "metrics.json")
    lines = [f"# {os.path.basename(run_dir)}", "", "## 训练记录", ""]
    if os.path.exists(losses_path):
        with open(losses_path, newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        val_key = "val_total" if history and "val_total" in history[0] else "val_loss"
        best_row = min(history, key=lambda row: float(row[val_key]))
        lines += [f"- 实际训练轮数：{len(history)} / {cfg['train']['max_epochs']}",
                  f"- 最佳验证轮次：{best_row['epoch']}",
                  f"- 最佳验证损失：{float(best_row[val_key]):.10g}"]
    else:
        lines += [f"- 计划训练轮数：{cfg['train']['max_epochs']}", "- 状态：训练中或未完成"]
    lines += ["- 检查点：`best.pth`（验证损失最低的轮次）", ""]
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as stream:
            metrics = json.load(stream)
        lines += ["## 评估结果", "", "```json",
                  json.dumps(metrics, ensure_ascii=False, indent=2), "```", ""]
    lines += ["## 本次配置", "", "```yaml", config_text, "```", "", "## 文件", ""]
    for name, description in (("config.yaml", "配置快照"), ("losses.csv", "逐轮损失"),
                              ("loss_curve.png", "损失曲线"), ("best.pth", "最佳模型"),
                              ("metrics.json", "评估指标")):
        if os.path.exists(os.path.join(run_dir, name)):
            lines.append(f"- `{name}`：{description}")
    lines.append("")
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def load_npz(npz_path: str) -> dict:
    path = npz_path if os.path.isabs(npz_path) else os.path.join(CODE_DIR, npz_path)
    return dict(np.load(path))
