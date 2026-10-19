# BIC1 tandem 对照

训练跑满配置中的 `max_epochs`（当前 500），每轮按验证集组合损失更新 `best.pth`，结束后用最佳权重评估。

在 `Code/` 目录执行：

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python -m unittest BIC1.tandem.tests.test_training -v
/opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python BIC1/tandem/train.py
```

使用已有 `bic1.npz` 和 `forward_20260929_101011`，训练一个新的逆向 MLP。前向权重固定；训练损失为等权参数 MSE + 前向代理光谱 MSE，光谱损失梯度穿过前向网络更新逆向网络。最佳模型按验证集组合损失保存。结果在 `runs/bic1/tandem/tandem_<timestamp>/`，包含独立的参数和代理光谱指标。代理光谱结果仍需 CST 回验。

2026-09-29 的已完成运行和直接逆向对照见 [comparison_2026-09-29.md](comparison_2026-09-29.md)。
