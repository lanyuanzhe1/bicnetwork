# experiment1016：BIC1 新测试集评估

本实验**只评估现有模型，不重新训练**。测试集为仓库根目录 `data_1016_for_test/`，共 180 组。各文件都留在本目录：`code/` 存评估脚本和测试，`results/` 存逐样本输出、汇总指标与光谱图，[报告.md](报告.md)解释结果。

## 三条评估路径

1. **前向**：测试集真实六参数 → 已训练前向网络 → 预测光谱；与测试集 CST 光谱比较。
2. **逆向**：测试集真实光谱 → 已训练逆向网络 → 预测六参数；与测试集 PARA 参数比较。分别评估直接逆向与 tandem 逆向。
3. **代理闭环**：同一条真实光谱 → 逆向网络 → 预测六参数 → 已训练前向网络 → 代理光谱；与输入的测试集 CST 光谱比较。直接逆向与 tandem 各自闭环。

闭环输出是**前向网络预测谱**，没有对预测结构重新运行 CST。

## 输入与模型

- 数据：`data_1016_for_test/{PARA,TXT,FIG}/`，ID 1–180。
- 旧 BIC1 归一化范围与频率轴：`Code/data/processed/bic1.npz`。
- 前向：`Code/runs/bic1/forward_20260929_101011/best.pth`。
- 直接逆向：`Code/runs/bic1/inverse_mlp_20260929_201411/best.pth`。
- Tandem 逆向：`Code/runs/bic1/tandem/tandem_20260929_233231/best.pth`。
- 网络定义：`Code/src/forward.py`、`Code/src/inverse.py`。配置从各 run 的 `config.yaml` 读取。

这些训练权重位于忽略的 `Code/runs/`，不纳入 Git；换机复现时需复制已有权重或重新训练。脚本不修改原始数据或模型权重。

## 运行

从仓库根目录执行，使用 miniconda `bic` 环境（也可用 `conda activate bic` 后的 `python`）：

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python -m unittest discover -s experiment/experiment1016_参数范围外推测试/code -p 'test_*.py' -v
/opt/homebrew/Caskroom/miniconda/base/envs/bic/bin/python experiment/experiment1016_参数范围外推测试/code/evaluate.py
```

脚本严格检查 1–180 的 PARA、TXT、FIG 是否齐全、参数/谱形状、频率轴和有限值。若要指定数据或结果目录，可使用 `--dataset`、`--out`。

## 结果文件

- [`results/metrics.json`](results/metrics.json)：三条路径的汇总指标、模型路径与数据域计数。
- [`results/per_sample.csv`](results/per_sample.csv)：每组参数、前向/闭环误差及逆向预测。
- [`results/predictions.npz`](results/predictions.npz)：真实谱、预测谱、预测参数，便于后续作图或复核。
- [`results/spectra_examples.png`](results/spectra_examples.png)：三个代表样本的谱线对照。

光谱 MSE 在所有样本与 1101 个频点上平均；参数 `param_mse_old_normalization` 使用**旧 BIC1 训练范围**归一化，可大于 1。原始参数 MAE 的单位为 µm。没有把超出旧范围的参数截断到 `[0,1]` 后再送入前向网络。
