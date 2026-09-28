"""BIC-0 前向训练入口，复用通用前向训练器。"""

import argparse
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE_DIR, "src"))

from train_forward import main as train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(CODE_DIR, "BIC0", "forward.yaml"))
    train(parser.parse_args().config)
