"""训练脚本共享工具：配置加载、种子、设备、运行目录"""

import os
import random
import shutil
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
    return run_dir


def load_npz(npz_path: str) -> dict:
    path = npz_path if os.path.isabs(npz_path) else os.path.join(CODE_DIR, npz_path)
    return dict(np.load(path))
