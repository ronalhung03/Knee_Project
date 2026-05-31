from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .constants import BASE_BATCH_SIZE, BASE_MODELS, ENSEMBLE_BATCH_SIZE
from .corn import corn_label_from_logits
from .data import load_paper_datasets, make_loader
from .metrics import compute_metrics
from .models import PaperEnsemble, load_base_checkpoint
from .training import _collect_base_logits, _device, seed_everything


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_base_models(
    data_dir: Path,
    output_dir: Path,
    split: str = "test",
    seed: int = 42,
    device_name: str | None = None,
    batch_size: int = BASE_BATCH_SIZE,
    workers: int = 4,
) -> None:
    seed_everything(seed)
    device = _device(device_name)
    bundle = load_paper_datasets(data_dir)
    dataset = getattr(bundle, split)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed, workers=workers)

    for model_name in BASE_MODELS:
        model = load_base_checkpoint(model_name, output_dir / "base" / model_name / "best.pt", device)
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(device))
                preds = corn_label_from_logits(logits)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.detach().cpu().tolist())
        metrics = compute_metrics(y_true, y_pred)
        _write_json(output_dir / "base" / model_name / f"{split}_metrics.json", asdict(metrics))
        print(f"{model_name} {split}: accuracy={metrics.accuracy:.4f} qwk={metrics.qwk:.4f}")


def evaluate_ensemble(
    data_dir: Path,
    output_dir: Path,
    split: str = "test",
    seed: int = 42,
    device_name: str | None = None,
    batch_size: int = ENSEMBLE_BATCH_SIZE,
    workers: int = 4,
) -> None:
    seed_everything(seed)
    device = _device(device_name)
    bundle = load_paper_datasets(data_dir)
    dataset = getattr(bundle, split)
    loader = make_loader(dataset, batch_size=batch_size, seed=seed, workers=workers)

    base_models = [
        load_base_checkpoint(name, output_dir / "base" / name / "best.pt", device)
        for name in BASE_MODELS
    ]
    features, labels = _collect_base_logits(base_models, loader, device)
    ensemble = PaperEnsemble().to(device)
    state = torch.load(output_dir / "ensemble" / "best.pt", map_location=device)
    ensemble.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    ensemble.eval()

    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            logits = ensemble(features[start : start + batch_size].to(device))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities.append(probs)
            y_pred.extend(np.argmax(probs, axis=1).tolist())
    metrics = compute_metrics(labels.numpy(), y_pred, np.concatenate(probabilities, axis=0))
    _write_json(output_dir / "ensemble" / f"{split}_metrics.json", asdict(metrics))
    print(f"ensemble {split}: accuracy={metrics.accuracy:.4f} qwk={metrics.qwk:.4f}")
