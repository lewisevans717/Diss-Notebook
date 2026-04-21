# Exploring the Interaction of Compression and Adversarial Attacks in Deep Learning Image Classifiers

**BSc Computer Science with Artificial Intelligence — COMP3003 Dissertation**  
**University of Nottingham, 2025–2026**  
**Supervised by Dr Shreyank Narayana Gowda**

---

## Overview

This repository contains the experimentation framework and results for a dissertation investigating the bidirectional relationship between lossy image compression and adversarial robustness in deep learning image classifiers. The central finding is that pipeline ordering — whether compression is applied before or after an adversarial attack — is the single most consequential variable determining classifier robustness.

The framework evaluates over 300 unique pipeline configurations across three datasets (CIFAR-10, CIFAR-100, ImageNet), two architectures (ResNet-18, ResNet-50), five compression methods (JPEG, JPEG2000, PCA, PatchSVD, LIC-ROI), and three gradient-based attacks (FGSM, PGD, APGD with DLR loss).

---

## Repository Structure

```
.
├── ProjectNotebook.ipynb          # Main experimentation notebook (CIFAR-10/100)
├── ResultsVisualisations.ipynb    # Figure generation for Chapter 5
├── SanityChecks.ipynb             # Module-level validation tests (Section 4.7)
├── requirements.txt               # Pinned dependency versions
├── README.md                      # This file
│
├── HPC/                           # ImageNet pipeline (University of Nottingham HPC)
│   ├── run_pipeline.py            # Main ImageNet evaluation script
│   ├── run_imagenet.sh            # SLURM job submission script
│   ├── setup_hpc.sh               # Environment setup for HPC cluster
│   ├── cache_compressai.py        # Pre-downloads CompressAI model checkpoints
│   ├── parse_log.py               # Extracts results from SLURM output logs
│   └── Tutorials/                 # HPC onboarding notes
│
└── Results/                       # Raw experimental output files
    ├── CIFAR-10/                   # CIFAR-10 results (.xlsx)
    ├── CIFAR-100/                  # CIFAR-100 results (.xlsx)
    ├── ImageNet/                   # ImageNet results (.xlsx)
    └── Figures/                    # Generated figures used in Chapter 5
```

---

## Framework Architecture

The framework is built around two design patterns:

**Strategy Pattern.** All image transformations — both compression methods and adversarial attacks — implement a shared abstract interface (`Perturbation`) with a single method:

```python
apply(model, images, labels, device) → Tensor
```

Concrete subclasses include `JpegPerturbation`, `Jpeg2000Perturbation`, `PcaPerturbation`, `PatchSVDPerturbation`, `LicRoiPerturbation`, `FgsmPerturbation`, `PGDPerturbation`, and `APGDPerturbation`.

**Pipeline Pattern.** `PerturbationPipeline` composes an ordered list of `Perturbation` objects and applies them sequentially. Constructing an Attack→Compression pipeline versus a Compression→Attack pipeline requires only passing the two objects in opposite order; the objects themselves are unchanged.

A **Registry Pattern** maps string keys (e.g., `"jpeg"`, `"apgd"`) to factory functions, enabling the full experimental grid to be specified declaratively.

---

## Experimental Configuration

### Datasets

| Dataset    | Classes | Resolution  | Split                     | Batch Size |
|------------|---------|-------------|---------------------------|------------|
| CIFAR-10   | 10      | 32 × 32     | Test (10,000 images)      | 128        |
| CIFAR-100  | 100     | 32 × 32     | Test (10,000 images)      | 128        |
| ImageNet   | 1,000   | 224 × 224   | Validation (50,000 images)| 32         |

### Models

| Model     | Dataset   | Weights Source                          |
|-----------|-----------|-----------------------------------------|
| ResNet-18 | CIFAR-10  | `timm` (pretrained, via `detectors`)    |
| ResNet-50 | CIFAR-10  | `timm` (pretrained, via `detectors`)    |
| ResNet-18 | CIFAR-100 | `timm` (pretrained, via `detectors`)    |
| ResNet-50 | CIFAR-100 | `timm` (pretrained, via `detectors`)    |
| ResNet-18 | ImageNet  | `torchvision` (`DEFAULT` = IMAGENET1K_V1) |
| ResNet-50 | ImageNet  | `torchvision` (`DEFAULT` = IMAGENET1K_V2) |

All models are used in inference mode with frozen BatchNorm statistics. No fine-tuning or adversarial training is performed.

### Adversarial Attacks

All attacks operate in pixel space over [0, 1] with L∞ perturbation budget ε = 8/255.

- **FGSM**: Single gradient step of size ε (cross-entropy loss).
- **PGD**: 10 iterative steps, step size α = 2/255, uniform random initialisation within the ε-ball.
- **APGD**: 20 steps, 1 random restart, DLR loss, adaptive step size initialised at 2ε with checkpoints at 22% and 75% of the iteration budget.

### Compression Methods

| Method    | Quality Parameter          | Quality Levels (Q25/Q50/Q75) |
|-----------|----------------------------|------------------------------|
| JPEG      | Quality factor             | 25, 50, 75                   |
| JPEG2000  | Compression rate           | 25, 50, 75                   |
| PCA       | % singular values retained | 25%, 50%, 75%                |
| PatchSVD  | % singular values per 8×8 patch | 25%, 50%, 75%           |
| LIC-ROI   | CompressAI quality level   | (1,3), (2,4), (4,6)         |

LIC-ROI is evaluated on ImageNet only, as the CompressAI `cheng2020_attn` architecture produces degenerate outputs at 32 × 32 resolution.

### Pipeline Types

| Pipeline                    | Description                                         |
|-----------------------------|-----------------------------------------------------|
| Clean                       | Baseline accuracy, no perturbation                  |
| Compression-only            | Standalone compression effect                       |
| Attack-only                 | Baseline adversarial vulnerability                  |
| Attack→Compression          | Compression as a post-attack purification defence   |
| Compression→Attack          | Compression as a pre-attack adversarial amplifier   |
| Compress→Attack→Compress    | CAC triple pipeline (CIFAR only)                    |

---

## Reproducibility

- **Global random seed**: 42, set via `torch.manual_seed`, `torch.cuda.manual_seed_all`, and deterministic CUDA operations (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).
- **Normalisation convention**: All perturbation operations denormalise to [0, 1] pixel space at entry and renormalise before returning. L∞ projection is enforced in pixel space.
- **Distortion metrics**: PSNR, MSE, and MAE are computed by comparing original and pipeline-output tensors after denormalisation.
- **Results format**: Each experiment writes a pandas DataFrame to `.xlsx` with columns: `model`, `pipeline`, `compression`, `attack`, `quality`, `epsilon`, `accuracy`, `correct`, `total`, `psnr_mean`, `mse_mean`, `mae_mean`.

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (experiments were run on Tesla P100 and RTX 2080 Ti)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset Access

- **CIFAR-10 / CIFAR-100**: Downloaded automatically by `torchvision.datasets` on first run.
- **ImageNet (ILSVRC 2012)**: Requires manual download of the validation set. Place the validation images in the standard ImageNet directory structure expected by `torchvision.datasets.ImageNet`.

### CompressAI Model Weights

The LIC-ROI module uses the `cheng2020_attn` model from CompressAI. On the HPC cluster, model weights should be pre-cached before job submission to avoid download timeouts:

```bash
python HPC/cache_compressai.py
```

---

## Running Experiments

### CIFAR-10 and CIFAR-100

Open `ProjectNotebook.ipynb` in a Jupyter environment with GPU access (e.g., Kaggle). The notebook contains the full pipeline grid construction, execution, and results export. CIFAR experiments complete in approximately 4–5 hours for all 88 pipelines across both ResNet-18 and ResNet-50.

### ImageNet

ImageNet experiments are designed for SLURM-based HPC clusters:

```bash
# Set up the environment
bash HPC/setup_hpc.sh

# Submit the job
sbatch HPC/run_imagenet.sh
```

`run_pipeline.py` is the main evaluation script. Results are written incrementally to allow partial recovery in the event of a job timeout. A full run over the 50,000-image validation set with 109 pipelines takes approximately 72 hours for ResNet-18 on an RTX 2080 Ti.

### Generating Figures

Open `ResultsVisualisations.ipynb` after experiments have completed. This notebook reads the `.xlsx` result files from `Results/` and generates all figures used in Chapter 5 of the dissertation.

---

## Results

Raw results are stored as Excel files in the `Results/` directory:

| File | Contents |
|------|----------|
| `Results/CIFAR-10/CIFAR10_Full_Results.xlsx` | 176 pipelines (88 per model) |
| `Results/CIFAR-10/CIFAR10_apgd_jpeg_extended.xlsx` | APGD→JPEG quality sweep Q40–Q60 |
| `Results/CIFAR-10/CIFAR10_CAC.xlsx` | Compress→Attack→Compress triple pipeline |
| `Results/CIFAR-100/CIFAR100_Full_Results.xlsx` | 176 pipelines (88 per model) |
| `Results/CIFAR-100/CIFAR100_apgd_jpeg_extended.xlsx` | APGD→JPEG quality sweep Q40–Q60 |
| `Results/CIFAR-100/CIFAR100_CAC.xlsx` | Compress→Attack→Compress triple pipeline |
| `Results/ImageNet/ImageNet_ResNet18_Results.xlsx` | 109 pipelines |
| `Results/ImageNet/ImageNet_ResNet50_Results.xlsx` | 109 pipelines |

---

## Key Findings

1. **Pipeline ordering is decisive.** The same JPEG codec at Q50 recovers accuracy from 0% to over 62% when placed after an attack, but yields 0% when placed before one.
2. **JPEG is the strongest post-attack defence on CIFAR**, with a broad optimal plateau across Q40–Q60 requiring no precise tuning.
3. **LIC-ROI provides the strongest defence on ImageNet**, recovering ResNet-18 from 0% to 45% under APGD.
4. **Pre-compression universally amplifies FGSM attacks**, with the magnitude scaling monotonically with compression aggressiveness.
5. **PSNR is not a reliable cross-method predictor of adversarial robustness**, but serves as a useful within-method indicator of adversarial signal removal.
6. **The CAC triple pipeline** recovers approximately 10 percentage points over single-pass defence on CIFAR.

---

## Compute Environments

| Environment | Hardware | Datasets |
|-------------|----------|----------|
| Kaggle Notebook | Tesla P100 GPU | CIFAR-10, CIFAR-100 |
| University of Nottingham HPC | NVIDIA RTX 2080 Ti | ImageNet |

---

## Acknowledgements

Supervised by Dr Shreyank Narayana Gowda, School of Computer Science, University of Nottingham.

---

## License

This code is provided for academic and research purposes as part of a BSc dissertation. The datasets used (CIFAR-10, CIFAR-100, ImageNet) are publicly available benchmarks used under standard academic licences for non-commercial research.
