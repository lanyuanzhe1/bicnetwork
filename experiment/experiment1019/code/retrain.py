"""按本次配置依次训练 BIC0、BIC1，并记录新 run 路径。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "Code"
EXPERIMENT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"


def train(script, config, run_parent, prefix, extra=()):
    folder = CODE / run_parent
    before = set(folder.glob(f"{prefix}_*"))
    subprocess.run([sys.executable, "-u", str(CODE / script),
                    "--config", str(CONFIGS / config), *extra], cwd=CODE, check=True)
    new = set(folder.glob(f"{prefix}_*")) - before
    if len(new) != 1 or not (next(iter(new)) / "README.md").exists():
        raise RuntimeError(f"无法唯一确定 {script} 的新运行目录：{new}")
    return next(iter(new))


def set_forward(config, forward_run):
    path = CONFIGS / config
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["evaluation"]["forward_run"] = str(forward_run.relative_to(CODE))
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bic0-forward-run", type=Path,
                        help="已有的本次 BIC0 前向 run；省略则从头训练")
    parser.add_argument("--bic0-inverse-run", type=Path,
                        help="已有的本次 BIC0 逆向 run；省略则从头训练")
    args = parser.parse_args()
    bic0_forward = (args.bic0_forward_run.resolve() if args.bic0_forward_run else
                    train("BIC0/train_forward.py", "bic0_forward.yaml", "runs/bic0", "forward"))
    if not (bic0_forward / "best.pth").is_file():
        raise FileNotFoundError(bic0_forward / "best.pth")
    set_forward("bic0_inverse.yaml", bic0_forward)
    bic0_inverse = (args.bic0_inverse_run.resolve() if args.bic0_inverse_run else
                    train("BIC0/train_inverse.py", "bic0_inverse.yaml", "runs/bic0", "inverse_mlp"))
    if not (bic0_inverse / "metrics.json").is_file():
        raise FileNotFoundError(bic0_inverse / "metrics.json")

    bic1_forward = train("BIC1/train_forward.py", "bic1_forward.yaml", "runs/bic1", "forward")
    set_forward("bic1_inverse.yaml", bic1_forward)
    set_forward("bic1_tandem.yaml", bic1_forward)
    bic1_inverse = train("BIC1/train_inverse.py", "bic1_inverse.yaml", "runs/bic1", "inverse_mlp",
                         ("--forward-run", str(bic1_forward.relative_to(CODE))))
    bic1_tandem = train("BIC1/tandem/train.py", "bic1_tandem.yaml", "runs/bic1/tandem", "tandem")
    runs = {"bic0_forward": bic0_forward, "bic0_inverse": bic0_inverse,
            "bic1_forward": bic1_forward, "bic1_inverse": bic1_inverse,
            "bic1_tandem": bic1_tandem}
    (EXPERIMENT / "results").mkdir(exist_ok=True)
    (EXPERIMENT / "results/runs.json").write_text(
        json.dumps({name: str(path.relative_to(REPO)) for name, path in runs.items()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(EXPERIMENT / "code/evaluate_new_runs.py"),
                    "--forward-run", str(bic1_forward), "--direct-run", str(bic1_inverse),
                    "--tandem-run", str(bic1_tandem)], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
