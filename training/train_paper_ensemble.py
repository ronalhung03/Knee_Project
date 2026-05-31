from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_method.constants import ENSEMBLE_BATCH_SIZE, ENSEMBLE_EPOCHS
from paper_method.training import train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IEEE Access 2023 fully connected ensemble.")
    parser.add_argument("--data-dir", required=True, type=Path, help="ImageFolder root with train/, val/, test/.")
    parser.add_argument("--output-dir", default=Path("paper_runs"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=ENSEMBLE_EPOCHS, type=int)
    parser.add_argument("--batch-size", default=ENSEMBLE_BATCH_SIZE, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train_ensemble(args.data_dir, args.output_dir, args.seed, args.device, args.epochs, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
