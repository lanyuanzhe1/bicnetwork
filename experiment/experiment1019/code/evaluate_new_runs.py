"""使用本次训练的三个 BIC1 run 重测独立的 10/16 数据。"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-run", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--tandem-run", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([
        sys.executable, str(REPO / "experiment/experiment1016_参数范围外推测试/code/evaluate.py"),
        "--dataset", str(REPO / "data_1016_for_test"),
        "--out", str(RESULTS),
        "--forward-run", str(args.forward_run),
        "--direct-run", str(args.direct_run),
        "--tandem-run", str(args.tandem_run),
    ], check=True)


if __name__ == "__main__":
    main()
