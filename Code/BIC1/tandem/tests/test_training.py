"""Tandem 损失必须经固定的前向网络更新逆向网络。"""

import unittest

import torch
from torch import nn

from BIC1.tandem.train import tandem_loss


class TandemLossTests(unittest.TestCase):
    def test_spectral_loss_backpropagates_only_to_inverse(self):
        inverse = nn.Linear(2, 1, bias=False)
        forward = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            inverse.weight.copy_(torch.tensor([[0.2, 0.3]]))
            forward.weight.copy_(torch.tensor([[1.0], [2.0]]))
        forward.requires_grad_(False)
        spectrum = torch.tensor([[1.0, 1.0]])
        target_params = torch.tensor([[0.0]])

        total, param_loss, spec_loss = tandem_loss(
            inverse, forward, spectrum, target_params,
            param_weight=0.0, spec_weight=1.0,
        )
        total.backward()

        self.assertGreater(spec_loss.item(), 0)
        self.assertGreater(param_loss.item(), 0)
        self.assertIsNotNone(inverse.weight.grad)
        self.assertGreater(inverse.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(forward.weight.grad)


if __name__ == "__main__":
    unittest.main()
