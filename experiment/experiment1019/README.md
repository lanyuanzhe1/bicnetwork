# experiment1019：历史最优配置跑满 500 轮

本实验从 `Code/runs/bic0`、`Code/runs/bic1` 中选取既有模型的配置快照，分别重训 BIC0 前向、BIC0 直接逆向、BIC1 前向、BIC1 直接逆向和 BIC1 tandem。所有模型跑满配置中的 500 轮，保存验证损失最低轮次的 `best.pth`。10/16 的 180 条数据仅用于独立测试，不参与本次训练或模型选择。

## 原始配置来源

| 本次配置 | 来源 run | 选择依据 |
| --- | --- | --- |
| `code/configs/bic0_forward.yaml` | `Code/runs/bic0/forward_20260928_113847` | BIC0 已有前向模型 |
| `code/configs/bic0_inverse.yaml` | `Code/runs/bic0/inverse_mlp_20260929_095236` | BIC0 逆向候选中验证损失最低 |
| `code/configs/bic1_forward.yaml` | `Code/runs/bic1/forward_20260929_101011` | BIC1 已有前向模型 |
| `code/configs/bic1_inverse.yaml` | `Code/runs/bic1/inverse_mlp_20260929_201411` | BIC1 直接逆向候选中验证损失与测试参数 MSE 更低 |
| `code/configs/bic1_tandem.yaml` | `Code/runs/bic1/tandem/tandem_20260929_233231` | BIC1 已有 tandem 模型 |

配置中移除旧的 `patience` 项；种子、模型架构、学习率、批量大小、数据划分与最大轮数沿用原 run。逆向及 tandem 的 `evaluation.forward_run` 更新为本次相应的新前向 run。

## 本次结果速览

五个模型都完成了 500/500 轮；最佳轮次分别是 BIC0 前向 235、BIC0 逆向 446、BIC1 前向 186、BIC1 直接逆向 204、BIC1 tandem 177。实际目录在 `results/runs.json`，每个 run 的 `README.md` 包含配置全文和评估结果。

在旧数据内部测试上，BIC1 tandem 的参数 MSE 从 0.018199 降到 0.016857，代理闭环谱 MSE 从 0.004862 降到 0.004712。但在 10/16 独立测试上，tandem 闭环谱 MSE 从 0.048228 升到 0.049170。新数据中 176/180 条超出旧 BIC1 训练参数范围；完整对照与原因分析见 `报告.md`。

## 文件

- `code/retrain.py`：按顺序训练五个模型，记录实际 run 路径并调用独立测试脚本。
- `code/evaluate_new_runs.py`：调用 `experiment1016_参数范围外推测试/code/evaluate.py`，分别计算前向、逆向和代理闭环误差。
- `code/configs/`：本次实际使用的配置。
- `results/runs.json`：五个新 run 的路径；每个 run 自带 `config.yaml`、`README.md`、逐轮损失、最佳权重和指标。
- `results/metrics.json`、`per_sample.csv`、`predictions.npz`、`spectra_examples.png`：10/16 独立测试结果。
- `报告.md`：旧 run 与新 run、独立测试的对照和解释。

## 复现

从项目根目录，激活 `bic` conda 环境后运行：

```bash
python experiment/experiment1019/code/retrain.py
```

脚本会新建五个 run，不覆盖已有权重。若已有本次 BIC0 前向 run，可传 `--bic0-forward-run` 跳过它。训练脚本在创建 run 时生成初始 README，完成训练后写入实际轮数、最佳验证轮次、指标和完整配置。
