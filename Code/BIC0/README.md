# BIC-0 基线

从 `Code/` 目录运行：

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python BIC0/data.py
/opt/homebrew/Caskroom/miniconda/base/bin/python BIC0/train_forward.py
/opt/homebrew/Caskroom/miniconda/base/bin/python BIC0/train_inverse.py
```

`data.py` 读取 1–768 号 PARA/TXT，保留 1101 点光谱和六列结构参数，确认 Y 全为 0 后原样保留；其他五列按各自范围归一化。输出沿用 `Code/data/processed/bic0.npz`，包括固定种子 42 的 train/val/test 索引。

逆向网络是单个残差 MLP，输入整条光谱，训练目标为六列参数（包括 Y=0）；训练损失仅为参数 MSE，没有前向闭环损失。测试闭环时只对逆向输出的副本施加 Y=0，再交给前向模型；原始输出保留用于诊断。参数误差的最终汇总口径暂不确定。

`inverse.yaml` 的 `evaluation.forward_run` 当前指向已有的 BIC-0 前向 checkpoint。重新训练前向模型后，需将其改为新的 `runs/bic0/forward_...` 目录。逆向产物写入 `Code/runs/bic0/inverse_mlp_.../`。正式训练由使用者自行启动。
