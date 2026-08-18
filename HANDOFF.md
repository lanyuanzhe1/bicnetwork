# Handoff — BIC 前向 + cVAE 逆向管线

**日期**：2026-08-19
**状态**：baseline 已跑通并训练完毕，可移交上服务器做正式实验

## 当前状态

- 代码：`Code/`（数据预处理、前向 MLP、cVAE、训练、评估，yaml 配置驱动）
- 数据：`Code/data/processed/data.npz`（3838 样本，8:1:1 种子 42，已固化）
- Baseline 模型与评估图：`Code/runs/`（forward_20260819_013825 / cvae_20260819_014112 / eval_20260819_014708）
- 设计文档：`docs/superpowers/specs/2026-08-18-bic-forward-inverse-pipeline-design.md`
- Git：main 分支初始提交已完成；`runs/`、`dataset/`、`基础文献/`、zip/mp4 已 gitignore

## Baseline 指标（测试集 385 样本）

| 模型 | 指标 |
|---|---|
| 前向（6→1101 残差 MLP, 2.15M 参数） | MSE 0.0030，R²=0.949，谐振频率中位误差 0.005 THz |
| 逆向 cVAE（best-of-50 闭环） | 谱重构中位 MSE 0.0043，网格命中率 27.8%，候选展布 2.75e-4（≈坍塌） |

## 环境

- conda env `bic`：python 3.11 + torch 2.13（MPS 可用）。激活：`conda activate bic`
- 不要往 base 环境装大包（用户要求）

## 运行（在 Code/ 下）

```bash
python src/data.py            # 已完成，npz 已固化，无需重跑
python src/train_forward.py   # ~2 分钟（MPS）
python src/train_cvae.py      # ~4 分钟（MPS）
python src/evaluate.py        # 自动取 runs/ 最新运行
```

## 已知问题与下一步（按优先级）

1. **cVAE 后验坍塌**（KL≈0，候选几乎相同 → 退化确定性回归器）。已用数据证实一对多真实存在（124 对样本谱 MSE<1e-4 但参数差 2–6 网格步），所以多样性值得救。已尝试 free-bits（configs/cvae_freebits.yaml，runs/cvae_20260819_014850）：多样性×3.5 但主指标变差（early stopping 被 KL 地板常数干扰）。服务器实验方向：超参扫描（β_max、free_bits、latent_dim）、early stopping 改看 recon、必要时对比 MDN/diffusion
2. **前向轻微过拟合**（train/val 差 10 倍）：可试 dropout、谱增强（频率微移）
3. **P1–P6 物理含义/单位**：用户找师姐确认（不阻塞训练，论文写作前必须拿到）
4. **服务器训练**：代码即拷即用；数据需 `dataset/` 或直接用 `data.npz`（14.6MB，已入库）

## 备注

- worktree 分支 `worktree-bic-code`（.claude/worktrees/bic-code）留有开发过程提交，可删
- OneDrive 占位文件已全部本地化；若换新机器，先验证 st_blocks 再跑 data.py
