"""训练应跑满配置轮次，同时保留最佳验证权重。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from BIC1.train_forward import train


class FullEpochTrainingTest(unittest.TestCase):
    def test_forward_runs_all_epochs_when_validation_does_not_improve(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(
                root / "tiny.npz",
                params_norm=np.zeros((3, 6), dtype=np.float32),
                spectra=np.full((3, 1101), 0.5, dtype=np.float32),
                idx_train=np.array([0]), idx_val=np.array([1]), idx_test=np.array([2]),
            )
            cfg = {
                "data": {"npz_path": str(root / "tiny.npz")},
                "model": {"in_dim": 6, "out_dim": 1101, "width": 8, "n_blocks": 0},
                "train": {"seed": 42, "batch_size": 1, "max_epochs": 3,
                          "lr": 0.0, "weight_decay": 0.0, "patience": 1},
                "output": {"runs_dir": str(root / "runs")},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            train(str(config_path))
            runs = list((root / "runs").glob("forward_*"))
            self.assertEqual(len(runs), 1)
            with (runs[0] / "losses.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 3])
            self.assertTrue((runs[0] / "best.pth").is_file())
            metrics = json.loads((runs[0] / "metrics.json").read_text())
            self.assertEqual(metrics["best_val_mse"], min(float(row["val_loss"]) for row in rows))
            readme = (runs[0] / "README.md").read_text(encoding="utf-8")
            self.assertIn("实际训练轮数：3 / 3", readme)
            self.assertIn("最佳验证轮次：1", readme)
            self.assertIn("lr: 0.0", readme)
            self.assertIn("test_spec_mse", readme)


if __name__ == "__main__":
    unittest.main()
