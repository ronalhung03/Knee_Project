from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .constants import BASE_MODELS, NUM_CLASSES, ORDINAL_LOGITS


def _model_with_weights(factory_name: str, weights_name: str | None = "DEFAULT") -> nn.Module:
    factory = getattr(models, factory_name)
    weights_cls = getattr(models, f"{factory_name.capitalize()}_Weights", None)
    if weights_cls is not None and weights_name is not None:
        try:
            return factory(weights=getattr(weights_cls, weights_name))
        except AttributeError:
            return factory(weights=weights_cls.DEFAULT)
    return factory(pretrained=True)


def _resnet34() -> nn.Module:
    try:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, ORDINAL_LOGITS)
    return model


def _vgg19() -> nn.Module:
    try:
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    except AttributeError:
        model = models.vgg19(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, ORDINAL_LOGITS)
    return model


def _densenet121() -> nn.Module:
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    except AttributeError:
        model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, ORDINAL_LOGITS)
    return model


def _densenet161() -> nn.Module:
    try:
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    except AttributeError:
        model = models.densenet161(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, ORDINAL_LOGITS)
    return model


def build_base_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "resnet34":
        return _resnet34()
    if name == "vgg19":
        return _vgg19()
    if name == "densenet121":
        return _densenet121()
    if name == "densenet161":
        return _densenet161()
    raise ValueError(f"Unsupported paper base model: {name}. Expected one of {BASE_MODELS}.")


class PaperEnsemble(nn.Module):
    """Fully connected ensemble over the four CORN base-model outputs."""

    def __init__(self, num_base_models: int = len(BASE_MODELS), num_classes: int = NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Linear(num_base_models * ORDINAL_LOGITS, num_classes)

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        if base_logits.ndim != 3:
            raise ValueError("Expected base logits shaped [batch, models, ordinal_logits].")
        return self.classifier(base_logits.flatten(start_dim=1))


def load_base_checkpoint(name: str, checkpoint: str | bytes | object, device: torch.device) -> nn.Module:
    model = build_base_model(name).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model

