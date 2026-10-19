"""BIC1 数据边界与逆向闭环约束。"""

import os
import tempfile
import unittest

import numpy as np

from BIC1.data import build_dataset


def write_sample(root, sample_id, params):
    np.savetxt(os.path.join(root, "PARA", f"para{sample_id}.txt"), params)
    frequency = np.linspace(0.5, 1.3, 1101)
    spectrum = np.full(1101, 0.5 + sample_id * 1e-6)
    np.savetxt(
        os.path.join(root, "TXT", f"{sample_id}.txt"),
        np.column_stack([frequency, spectrum]),
        header="frequency\nspectrum",
    )


class DataTests(unittest.TestCase):
    def test_build_dataset_skips_missing_spectrum_and_normalizes_nonzero_y(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            write_sample(root, 769, [110, 110, 5, 40, 30, 5])
            np.savetxt(
                os.path.join(root, "PARA", "para770.txt"),
                [120, 120, 10, 50, 40, 10],
            )
            write_sample(root, 771, [140, 140, 15, 70, 60, 20])

            result = build_dataset(
                root,
                first_id=769,
                last_id=771,
                expected_missing_ids=(770,),
                expected_y_counts={5: 1, 20: 1},
            )

            np.testing.assert_array_equal(result["ids"], [769, 771])
            np.testing.assert_array_equal(result["skipped_ids"], [770])
            self.assertEqual(result["spectra"].shape, (2, 1101))
            np.testing.assert_array_equal(result["params_raw"][:, 5], [5, 20])
            np.testing.assert_allclose(result["params_norm"][:, 5], [0, 1])

    def test_build_dataset_rejects_zero_y(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            write_sample(root, 769, [110, 110, 5, 40, 30, 0])

            with self.assertRaisesRegex(ValueError, "Y"):
                build_dataset(
                    root,
                    first_id=769,
                    last_id=769,
                    expected_missing_ids=(),
                    expected_y_counts={5: 1},
                )

    def test_build_dataset_rejects_unexpected_missing_id(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            write_sample(root, 769, [110, 110, 5, 40, 30, 5])

            with self.assertRaisesRegex(ValueError, "缺失"):
                build_dataset(
                    root,
                    first_id=769,
                    last_id=770,
                    expected_missing_ids=(),
                    expected_y_counts={5: 1},
                )

    def test_build_dataset_rejects_incomplete_y_layers(self):
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "PARA"))
            os.mkdir(os.path.join(root, "TXT"))
            write_sample(root, 769, [110, 110, 5, 40, 30, 5])

            with self.assertRaisesRegex(ValueError, "Y 分层"):
                build_dataset(
                    root,
                    first_id=769,
                    last_id=769,
                    expected_missing_ids=(),
                    expected_y_counts={5: 1, 10: 1},
                )


class InverseTests(unittest.TestCase):
    def test_forward_input_keeps_predicted_y(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("当前环境未安装 torch")
        from BIC1.train_inverse import bic1_forward_input

        predicted = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.7]])
        forward_input = bic1_forward_input(predicted)

        torch.testing.assert_close(forward_input, predicted)
        self.assertAlmostEqual(forward_input[0, 5].item(), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
