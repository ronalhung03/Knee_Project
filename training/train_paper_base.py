from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_method.constants import BASE_BATCH_SIZE, BASE_EPOCHS
from paper_method.training import train_base_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the four IEEE Access 2023 CORN base models.")
    parser.add_argument("--data-dir", required=True, type=Path, help="ImageFolder root with train/, val/, test/.")
    parser.add_argument("--output-dir", default=Path("paper_runs"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=BASE_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=BASE_BATCH_SIZE, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train_base_models(args.data_dir, args.output_dir, args.seed, args.device, args.epochs, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
