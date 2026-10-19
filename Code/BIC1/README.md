# BIC1 基线

`BIC1` 只使用 `周期分类数据集/BIC/` 中 Y=5/10/15/20 的数据，不与 BIC-0 合并。原始参数 ID 为 769–3840；预处理只允许光谱 2532、2645 缺失，并把这两个 ID 写入 `skipped_ids`。如果 PARA 缺失、TXT 出现其他缺失，或四个 Y 分层数量不符，预处理会直接报错，避免在 OneDrive 文件未完整落地时训练残缺数据。

六列参数 Px、Py、A、B、L、Y 都按 BIC1 自身范围归一化。逆向模型直接预测六列参数；代理闭环评估不会把 Y 强制改为 0。

前向和逆向训练都会跑满配置中的 `max_epochs`（当前均为 500），每轮根据验证损失更新 `best.pth`，结束后使用最佳权重评估，不再提前停止。

## 执行顺序

以下命令均在 `Code/` 目录执行。

1. 运行测试：

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python -m unittest BIC1.tests.test_pipeline -v
   ```
2. 生成固化数据集：

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python BIC1/data.py
   ```

   预期输出为 `data/processed/bic1.npz`，包含 3070 条有效样本，train/val/test 应为 2456/307/307。
3. 训练前向模型：

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python BIC1/train_forward.py
   ```

   记下新建的 `runs/bic1/forward_<timestamp>/` 目录。
4. 训练逆向模型，显式传入上一步的前向运行：

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python BIC1/train_inverse.py \
     --forward-run runs/bic1/forward_<timestamp>
   ```

前向产物写入 `runs/bic1/forward_<timestamp>/`；逆向产物写入 `runs/bic1/inverse_mlp_<timestamp>/`。逆向 `metrics.json` 分别记录归一化参数 MSE、每列参数 MAE、原始尺度 MAE 和前向代理闭环光谱 MSE。代理闭环不是 CST 回验。

## 运行环境

测试、预处理和训练使用 miniconda `bic` 环境：`/opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python`。该环境包含训练需要的 PyTorch，并可使用 MPS。
