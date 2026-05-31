from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .constants import (
    BASE_BATCH_SIZE,
    BASE_EPOCHS,
    BASE_LR,
    BASE_LR_STEP,
    BASE_MODELS,
    ENSEMBLE_BATCH_SIZE,
    ENSEMBLE_EPOCHS,
    ENSEMBLE_LR,
    ENSEMBLE_LR_STEP,
    LR_GAMMA,
    NUM_CLASSES,
)
from .corn import corn_label_from_logits, corn_loss
from .data import load_paper_datasets, make_loader
from .metrics import compute_metrics
from .models import PaperEnsemble, build_base_model, load_base_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_corn_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    labels_all: list[int] = []
    preds_all: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = corn_loss(logits, labels, NUM_CLASSES)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        preds = corn_label_from_logits(logits)
        labels_all.extend(labels.detach().cpu().tolist())
        preds_all.extend(preds.detach().cpu().tolist())

    return float(np.mean(losses)), compute_metrics(labels_all, preds_all)


def train_base_models(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str | None = None,
    epochs: int = BASE_EPOCHS,
    batch_size: int = BASE_BATCH_SIZE,
    workers: int = 4,
) -> None:
    seed_everything(seed)
    device = _device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_paper_datasets(data_dir)
    train_loader = make_loader(bundle.train, batch_size=batch_size, shuffle=True, seed=seed, workers=workers)
    val_loader = make_loader(bundle.val, batch_size=batch_size, seed=seed, workers=workers)
    test_loader = make_loader(bundle.test, batch_size=batch_size, seed=seed, workers=workers)

    for model_name in BASE_MODELS:
        model = build_base_model(model_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=BASE_LR_STEP, gamma=LR_GAMMA)
        model_dir = output_dir / "base" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        best_val = -1.0

        history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = _run_corn_epoch(model, train_loader, device, optimizer)
            val_loss, val_metrics = _run_corn_epoch(model, val_loader, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
            history.append(row)
            if val_metrics.accuracy > best_val:
                best_val = val_metrics.accuracy
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": asdict(val_metrics)}, model_dir / "best.pt")
            print(
                f"{model_name} epoch {epoch:03d}: "
                f"train_acc={train_metrics.accuracy:.4f} val_acc={val_metrics.accuracy:.4f}"
            )

        model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device)["model"])
        _, test_metrics = _run_corn_epoch(model, test_loader, device)
        _save_json(model_dir / "history.json", history)
        _save_json(model_dir / "test_metrics.json", asdict(test_metrics))


def _collect_base_logits(base_models: list[nn.Module], loader, device):
    labels_all: list[int] = []
    features_all: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = [model(images).detach().cpu() for model in base_models]
            features_all.append(torch.stack(logits, dim=1))
            labels_all.extend(labels.tolist())
    return torch.cat(features_all, dim=0), torch.tensor(labels_all, dtype=torch.long)


def _run_ensemble_epoch(model, features, labels, batch_size, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    preds_all: list[int] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[int] = []
    order = torch.randperm(len(labels)) if is_train else torch.arange(len(labels))

    for start in range(0, len(labels), batch_size):
        idx = order[start : start + batch_size]
        x = features[idx].to(device)
        y = labels[idx].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        losses.append(float(loss.detach().cpu()))
        preds_all.extend(preds.tolist())
        probs_all.append(probs)
        labels_all.extend(y.detach().cpu().tolist())

    probabilities = np.concatenate(probs_all, axis=0) if probs_all else None
    return float(np.mean(losses)), compute_metrics(labels_all, preds_all, probabilities)


def train_ensemble(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str | None = None,
    epochs: int = ENSEMBLE_EPOCHS,
    batch_size: int = ENSEMBLE_BATCH_SIZE,
    workers: int = 4,
) -> None:
    seed_everything(seed)
    device = _device(device_name)
    bundle = load_paper_datasets(data_dir)
    loaders = {
        "train": make_loader(bundle.train, batch_size=batch_size, seed=seed, workers=workers),
        "val": make_loader(bundle.val, batch_size=batch_size, seed=seed, workers=workers),
        "test": make_loader(bundle.test, batch_size=batch_size, seed=seed, workers=workers),
    }
    base_models = [
        load_base_checkpoint(name, output_dir / "base" / name / "best.pt", device)
        for name in BASE_MODELS
    ]
    features = {split: _collect_base_logits(base_models, loader, device) for split, loader in loaders.items()}

    ensemble = PaperEnsemble().to(device)
    optimizer = optim.Adam(ensemble.parameters(), lr=ENSEMBLE_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=ENSEMBLE_LR_STEP, gamma=LR_GAMMA)
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _run_ensemble_epoch(
            ensemble, features["train"][0], features["train"][1], batch_size, device, optimizer
        )
        val_loss, val_metrics = _run_ensemble_epoch(
            ensemble, features["val"][0], features["val"][1], batch_size, device
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
        )
        if val_metrics.accuracy > best_val:
            best_val = val_metrics.accuracy
            torch.save({"model": ensemble.state_dict(), "epoch": epoch, "val": asdict(val_metrics)}, ensemble_dir / "best.pt")
        print(f"ensemble epoch {epoch:03d}: train_acc={train_metrics.accuracy:.4f} val_acc={val_metrics.accuracy:.4f}")

    ensemble.load_state_dict(torch.load(ensemble_dir / "best.pt", map_location=device)["model"])
    _, test_metrics = _run_ensemble_epoch(ensemble, features["test"][0], features["test"][1], batch_size, device)
    _save_json(ensemble_dir / "history.json", history)
    _save_json(ensemble_dir / "test_metrics.json", asdict(test_metrics))
