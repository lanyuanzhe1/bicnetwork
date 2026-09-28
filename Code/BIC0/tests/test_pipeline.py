"""BIC-0 数据边界与逆向闭环约束。"""

import os
import tempfile
import unittest

import numpy as np

from BIC0.data import build_dataset


class DataTests(unittest.TestCase):
    def test_build_dataset_keeps_y_zero_and_full_spectrum(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            frequency = np.linspace(0.5, 1.3, 1101)
            for sample_id in (1, 2):
                np.savetxt(os.path.join(root, "PARA", f"para{sample_id}.txt"),
                           [110 + sample_id, 120, 5, 40, 30, 0])
                np.savetxt(os.path.join(root, "TXT", f"{sample_id}.txt"),
                           np.column_stack([frequency, np.full(1101, 0.5)]),
                           header="frequency\nspectrum")
            result = build_dataset(root, last_id=2)
            self.assertEqual(result["spectra"].shape, (2, 1101))
            np.testing.assert_array_equal(result["params_norm"][:, 5], [0, 0])

    def test_build_dataset_rejects_nonzero_y(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            np.savetxt(os.path.join(root, "PARA", "para1.txt"), [110, 120, 5, 40, 30, 1])
            np.savetxt(os.path.join(root, "TXT", "1.txt"),
                       np.column_stack([np.linspace(0.5, 1.3, 1101), np.full(1101, 0.5)]),
                       header="frequency\nspectrum")
            with self.assertRaisesRegex(ValueError, "Y"):
                build_dataset(root, last_id=1)


class InverseTests(unittest.TestCase):
    def test_clamped_y_does_not_change_raw_prediction(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("base 环境未安装 torch")
        from BIC0.train_inverse import bic0_forward_input

        raw = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.7]])
        projected = bic0_forward_input(raw)
        self.assertEqual(projected[0, 5].item(), 0)
        self.assertAlmostEqual(raw[0, 5].item(), 0.7, places=6)
        torch.testing.assert_close(projected[:, :5], raw[:, :5])


if __name__ == "__main__":
    unittest.main()
