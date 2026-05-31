# Knee Osteoarthritis Detection and Classification

This project implements a deep learning pipeline for automatic knee osteoarthritis severity grading from X-ray images using the Kellgren-Lawrence (KL) scale.

The pipeline classifies each knee X-ray into one of five ordered KL grades:

- `0`: normal / healthy
- `1`: doubtful osteoarthritis
- `2`: minimal / mild osteoarthritis
- `3`: moderate osteoarthritis
- `4`: severe osteoarthritis

## Dataset

Use the public Knee Osteoarthritis Severity Grading Dataset:

- Source: Mendeley Data, `Knee Osteoarthritis Severity Grading Dataset`
- DOI: `10.17632/56rmx5bjcr.1`
- Kaggle mirror: `shashwatwork/knee-osteoarthritis-dataset-with-severity`

Verified class distribution:

| KL Grade | Images |
| --- | ---: |
| 0 | 3,857 |
| 1 | 1,770 |
| 2 | 2,578 |
| 3 | 1,286 |
| 4 | 295 |
| Total | 9,786 |

The expected local dataset layout is:

```text
data/knee_oa_severity/
  train/0 ... train/4
  val/0   ... val/4
  test/0  ... test/4
```

The `data/` folder is ignored by git. Keep the dataset local and do not commit image files.

Download the Kaggle mirror into the recommended location:

```powershell
mkdir data
kaggle datasets download shashwatwork/knee-osteoarthritis-dataset-with-severity -p data\knee_oa_severity --unzip
```

## Methodology

The project uses an ordinal deep learning approach for KL-grade classification.

Pipeline:

1. Resize each X-ray to `224 x 224`.
2. Apply training augmentation:
   - brightness and saturation jitter
   - horizontal flip
   - random affine transform
   - ImageNet normalization
3. Fine-tune four ImageNet-pretrained CNN base models:
   - `ResNet-34`
   - `VGG-19`
   - `DenseNet-121`
   - `DenseNet-161`
4. Treat KL grading as ordinal classification.
5. Replace each base model's final layer with `4` ordinal logits for the `5` KL grades.
6. Train base models with CORN ordinal loss.
7. Build a fully connected ensemble over the four base-model ordinal outputs.
8. Train the ensemble with cross entropy.
9. Report accuracy, precision, recall, F1, AUC, QWK, MAE, MSE, confusion matrix, and per-grade accuracy.
10. Generate Eigen-CAM heatmaps for visual interpretation.

Training settings:

| Setting | Value |
| --- | --- |
| Base models | `ResNet-34`, `VGG-19`, `DenseNet-121`, `DenseNet-161` |
| Base loss | CORN ordinal loss |
| Base epochs | `100` |
| Base batch size | `28` |
| Base optimizer | Adam |
| Base learning rate | `0.0001` |
| Base LR decay | every `5` epochs |
| Ensemble input | four base-model ordinal outputs |
| Ensemble classifier | fully connected layer |
| Ensemble loss | cross entropy |
| Ensemble epochs | `25` |
| Ensemble batch size | `28` |
| Ensemble LR decay | every `3` epochs |

## Repository Structure

```text
paper_method/
  constants.py      # hyperparameters and class constants
  corn.py           # CORN loss and prediction helpers
  data.py           # dataset transforms, loaders, and split utility
  models.py         # base CNNs and ensemble model
  training.py       # base-model and ensemble training loops
  evaluation.py     # base-model and ensemble evaluation
  metrics.py        # metric computation

training/
  train_paper_base.py
  train_paper_ensemble.py

evaluation/
  evaluate_paper_method.py
  eigen_cam_paper.py

utils/
  create_paper_split.py

requirements-paper.txt
```

## Setup

Install dependencies:

```powershell
pip install -r requirements-paper.txt
```

`torch` and `torchvision` must match the target hardware. A CUDA-enabled PyTorch install is strongly recommended for real training.

## Run

Train the four base classifiers:

```powershell
python training/train_paper_base.py --data-dir data\knee_oa_severity --output-dir paper_runs
```

Train the ensemble after all four base checkpoints are available:

```powershell
python training/train_paper_ensemble.py --data-dir data\knee_oa_severity --output-dir paper_runs
```

Evaluate base models and ensemble:

```powershell
python evaluation/evaluate_paper_method.py --data-dir data\knee_oa_severity --output-dir paper_runs
```

Generate an Eigen-CAM heatmap:

```powershell
python evaluation/eigen_cam_paper.py --model densenet161 --checkpoint paper_runs\base\densenet161\best.pt --image D:\path\image.png --output paper_runs\cam\densenet161.png
```

## Create a Split

If starting from a raw class-folder dataset containing only folders `0` through `4`, create a stratified `7:2:1` split:

```powershell
python utils/create_paper_split.py --source-dir D:\path\raw_koa --output-dir data\knee_oa_severity
```

By default this copies files. Use `--move` only if you intentionally want to move the source files.

## Local Smoke Test

The full training run is computationally heavy. Use this command only to verify the code path locally:

```powershell
python training/train_paper_base.py --data-dir data\knee_oa_severity --output-dir paper_runs_test --epochs 1 --batch-size 2 --workers 0
```

This is not a meaningful training run. Real training should be done on a CUDA GPU machine.
