# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dissertation research project (University of Nottingham) investigating **adversarial robustness and image compression as a defense mechanism**. The core research question: can image compression pipelines defend neural networks against adversarial attacks, and how do different compression methods and attack types interact?

---

## Experimental Design

### Datasets

| Dataset | Classes | Image Size | Split Used | Batch Size |
|---|---|---|---|---|
| CIFAR-10 | 10 | 32×32 | Test (10,000 images) | 128 |
| CIFAR-100 | 100 | 32×32 | Test (10,000 images) | 128 |
| ImageNet | 1000 | 224×224 | Validation (50,000 images) | 32 |

CIFAR-10 and CIFAR-100 were run on Kaggle (GPU: Tesla P100). ImageNet is run on the University of Nottingham HPC cluster (GPU: NVIDIA GeForce RTX 2080 Ti), using a locally installed copy at `/data/shared/imagenet`.

### Models

| Model Key | Architecture | Dataset | Source |
|---|---|---|---|
| `resnet18_cifar10` | ResNet-18 | CIFAR-10 | timm (pretrained) |
| `resnet50_cifar10` | ResNet-50 | CIFAR-10 | timm (pretrained) |
| `resnet18_cifar100` | ResNet-18 | CIFAR-100 | timm (pretrained) |
| `resnet50_cifar100` | ResNet-50 | CIFAR-100 | timm (pretrained) |
| `resnet18_imagenet` | ResNet-18 | ImageNet | torchvision (DEFAULT weights) |
| `resnet50_imagenet` | ResNet-50 | ImageNet | torchvision (DEFAULT weights) |

For CIFAR-10 and CIFAR-100, two models were evaluated per dataset (ResNet18 and ResNet50). For ImageNet, the same two models are evaluated. ViT variants exist in the registry but were not used in the main runs.

### Compression Methods

| Key | Method | Quality Parameter | Notes |
|---|---|---|---|
| `jpeg` | JPEG (PIL) | Quality: 25, 50, 75 (0–100 scale) | Standard lossy compression |
| `jpeg2000` | JPEG2000 (PIL) | Quality: 25, 50, 75 (mapped to compression rate) | Wavelet-based lossy compression |
| `pca` | PCA per channel | Quality: 25, 50, 75 (% of singular values retained) | Applied per image channel |
| `patchsvd` | Patch-based SVD | Quality: 25, 50, 75 (% of singular values per patch) | 8×8 patches, SVD per patch |
| `lic_roi` | Learned Image Compression with ROI | Quality: 25, 50, 75 (mapped to CompressAI levels 1–6) | CompressAI `cheng2020_attn`; salient regions get higher quality reconstruction |

**Quality mapping for `lic_roi`:** quality values are mapped to CompressAI levels 1–6 via `_map_quality_to_lic_level()`. The ROI is computed via gradient-based saliency (w.r.t. predicted class). High-quality reconstruction is applied to salient regions; low-quality to background. `lic_roi` is included for all three datasets but is most meaningful for ImageNet (224×224) as CompressAI was designed for natural images.

**Note:** `lic_roi` was omitted from CIFAR runs (redundant at 32×32). It is included in the ImageNet run.

### Attack Methods

| Key | Method | Epsilon | Notes |
|---|---|---|---|
| `fgsm` | Fast Gradient Sign Method | 8/255 | Single-step, gradient sign |
| `pgd` | Projected Gradient Descent | 8/255 | 10 steps, α=2/255, random start |
| `apgd` | Auto-PGD with DLR loss | 8/255 | 20 steps, 1 restart, DLR loss |

All attacks operate in pixel space ([0,1]) and project back to the L∞ ball around the original image. ε=8/255 ≈ 0.031 is used throughout (standard ImageNet perturbation budget).

### Pipeline Compositions

For each dataset/model combination, the following pipeline types are evaluated:

1. **Clean** — no perturbation (baseline accuracy)
2. **Compression-only** — single compression at each quality level
3. **Attack-only** — single attack at ε=8/255
4. **Compression → Attack** — compression applied first, then adversarial attack on the compressed image
5. **Attack → Compression** — adversarial attack applied first, then compression as a defence

**Total pipelines per run:**
- 1 clean
- 5 compressions × 3 qualities = 15 compression-only
- 3 attacks × 1 epsilon = 3 attack-only
- 5 × 3 × 3 × 1 = 45 compression→attack
- 3 × 5 × 1 × 3 = 45 attack→compression
- **Total: 109 pipelines**

For CIFAR runs: `lic_roi` excluded → 4 compressions → **85 pipelines**.

### Metrics

For every pipeline × model combination the following are recorded:

| Metric | Description |
|---|---|
| `accuracy` | Top-1 classification accuracy |
| `correct` | Number of correctly classified images |
| `total` | Total images evaluated |
| `psnr_mean` | Mean Peak Signal-to-Noise Ratio (dB) between original and perturbed images |
| `mse_mean` | Mean Squared Error (pixel space, [0,1]) |
| `mae_mean` | Mean Absolute Error (pixel space, [0,1]) |

PSNR, MSE, and MAE are computed in pixel space after denormalisation. For clean and attack-only pipelines, these measure the perturbation magnitude. For compression pipelines, they measure compression distortion.

---

## Code Architecture

All logic for CIFAR runs lives in `ProjectNotebook.ipynb`. The ImageNet HPC run uses `HPC/run_pipeline.py` (a self-contained script derived from the notebook).

### Core Abstractions

- `Perturbation` (ABC) — base class with abstract `apply(model, images, labels, device)` method
- `PerturbationPipeline` — chains multiple `Perturbation` instances sequentially
- `ModelSpec` (dataclass) — bundles model loader, `num_classes`, `input_size`, `dataset`
- `PipelineSpec` (dataclass) — describes one experimental configuration (pipeline + metadata)
- `CompressionSpec` / `AttackSpec` — registry entries with a `make(param)` factory

### Key Functions

- `build_pipeline_grid()` — generates all 109 (or 85) pipeline combinations
- `evaluate_pipeline()` — runs one pipeline over a dataloader, returns accuracy + metrics
- `run_pipeline_grid()` — iterates all models × pipelines, returns a `pd.DataFrame`

### Results Output

Results are saved as `results.xlsx` with columns: `model`, `pipeline`, `compression`, `attack`, `quality`, `epsilon`, `accuracy`, `correct`, `total`, `psnr_mean`, `mse_mean`, `mae_mean`.

---

## Completed Runs

| Dataset | Models | Pipelines | Status |
|---|---|---|---|
| CIFAR-10 | ResNet18, ResNet50 | 85 (no lic_roi) | Complete |
| CIFAR-100 | ResNet18, ResNet50 | 85 (no lic_roi) | Complete |
| ImageNet | ResNet18, ResNet50 | 109 (with lic_roi) | In progress (HPC) |

---

## Environment

- Kaggle: Python 3.11 (Kaggle default environment, no venv)
- HPC: Python 3.11 virtual environment at `~/Diss-Notebook/venv/`
- Key packages: PyTorch, torchvision, timm, compressai, detectors, pandas, openpyxl
- HPC cluster: University of Nottingham CS cluster (SLURM), `cs` partition, RTX 2080 Ti
