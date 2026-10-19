"""验证三条评估路径使用各自应有的输入。"""

import unittest

import numpy as np
import torch

from evaluate import compute_predictions


class Forward(torch.nn.Module):
    def forward(self, params):
        return params[:, :2]


class Inverse(torch.nn.Module):
    def forward(self, spectra):
        return torch.stack((spectra[:, 1], spectra[:, 0]), dim=1)


class EvaluationPathsTest(unittest.TestCase):
    def test_forward_inverse_and_loop_are_separate(self):
        params = np.array([[12.0, 23.0], [14.0, 21.0]], dtype=np.float32)
        spectra = np.array([[0.8, 0.3], [0.5, 0.1]], dtype=np.float32)
        results = compute_predictions(
            Forward(), {"direct": Inverse()}, params, spectra,
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([20.0, 30.0], dtype=np.float32),
        )
        np.testing.assert_allclose(results["forward_spectra"], [[0.2, 0.3], [0.4, 0.1]])
        np.testing.assert_allclose(results["inverse"]["direct"]["params_raw"], [[13, 28], [11, 25]])
        np.testing.assert_allclose(results["inverse"]["direct"]["loop_spectra"], [[0.3, 0.8], [0.1, 0.5]])


if __name__ == "__main__":
    unittest.main()
