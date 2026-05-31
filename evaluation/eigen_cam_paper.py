from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_method.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from paper_method.models import load_base_checkpoint
from paper_method.training import _device


def _target_layer(model_name: str, model):
    if model_name == "resnet34":
        return model.layer4[-1]
    if model_name in {"densenet121", "densenet161"}:
        return model.features[-1]
    if model_name == "vgg19":
        return model.features[-1]
    raise ValueError(f"Unsupported model for Eigen-CAM: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Eigen-CAM heatmaps for paper base models.")
    parser.add_argument("--model", required=True, choices=("resnet34", "vgg19", "densenet121", "densenet161"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import cv2
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Eigen-CAM requires optional dependencies. Install them with "
            "`pip install -r requirements-paper.txt`."
        ) from exc

    device = _device(args.device)
    model = load_base_checkpoint(args.model, args.checkpoint, device)
    image = Image.open(args.image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.asarray(image).astype(np.float32) / 255.0
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image).unsqueeze(0).to(device)

    with EigenCAM(model=model, target_layers=[_target_layer(args.model, model)]) as cam:
        grayscale = cam(input_tensor=tensor)[0]
    visualization = show_cam_on_image(rgb, grayscale, use_rgb=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
