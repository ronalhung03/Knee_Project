from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .constants import (
    BASE_BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def paper_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def paper_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass(frozen=True)
class DatasetBundle:
    train: datasets.ImageFolder
    val: datasets.ImageFolder
    test: datasets.ImageFolder


def load_paper_datasets(data_dir: Path) -> DatasetBundle:
    return DatasetBundle(
        train=datasets.ImageFolder(data_dir / "train", transform=paper_train_transform()),
        val=datasets.ImageFolder(data_dir / "val", transform=paper_eval_transform()),
        test=datasets.ImageFolder(data_dir / "test", transform=paper_eval_transform()),
    )


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = BASE_BATCH_SIZE,
    shuffle: bool = False,
    seed: int = 42,
    workers: int = 4,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _iter_image_files(class_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def create_paper_split(source_dir: Path, output_dir: Path, seed: int = 42, copy: bool = True) -> Path:
    """Create the paper's stratified 7:2:1 train/test/validation split.

    The source directory must contain class folders named 0, 1, 2, 3, and 4.
    The output directory will contain train/, test/, val/ ImageFolder layouts
    plus split_manifest.csv.
    """

    rng = random.Random(seed)
    manifest_rows: list[dict[str, str]] = []
    action = shutil.copy2 if copy else shutil.move

    for class_name in CLASS_NAMES:
        files = list(_iter_image_files(source_dir / class_name))
        rng.shuffle(files)
        train_n = int(len(files) * TRAIN_RATIO)
        test_n = int(len(files) * TEST_RATIO)
        groups = {
            "train": files[:train_n],
            "test": files[train_n : train_n + test_n],
            "val": files[train_n + test_n :],
        }
        for split, split_files in groups.items():
            target_class_dir = output_dir / split / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                dst = target_class_dir / src.name
                action(src, dst)
                manifest_rows.append(
                    {
                        "source": str(src),
                        "target": str(dst),
                        "split": split,
                        "label": class_name,
                    }
                )

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("source", "target", "split", "label"))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def split_main() -> None:
    parser = argparse.ArgumentParser(description="Create the paper's 7:2:1 stratified split.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    args = parser.parse_args()
    manifest = create_paper_split(args.source_dir, args.output_dir, args.seed, copy=not args.move)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    split_main()
