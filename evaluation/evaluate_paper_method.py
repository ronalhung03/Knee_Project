from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_method.evaluation import evaluate_base_models, evaluate_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the IEEE Access 2023 paper-method checkpoints.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("paper_runs"), type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    args = parser.parse_args()

    if not args.skip_base:
        evaluate_base_models(args.data_dir, args.output_dir, args.split, args.seed, args.device, workers=args.workers)
    if not args.skip_ensemble:
        evaluate_ensemble(args.data_dir, args.output_dir, args.split, args.seed, args.device, workers=args.workers)


if __name__ == "__main__":
    main()
