# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dissertation research project (University of Nottingham) investigating **adversarial robustness and image compression as a defense mechanism**. The core question: can image compression pipelines defend neural networks against adversarial attacks?

## Environment

- Python 3.13 virtual environment at `.venv/`
- No `requirements.txt` — dependencies are installed inline in the notebook via `!pip install`
- Key packages: PyTorch, torchvision, timm, compressai, detectors, datasets (HuggingFace), pandas

To activate the environment: `source .venv/bin/activate`

## Running the Project

The project is entirely Jupyter notebook-based:

- **`ProjectNotebook.ipynb`** — main experimental notebook; run cells end-to-end to execute the pipeline
- **`DataAnalysisNotebook.ipynb`** — analysis and visualization of results exported to `results.xlsx`

To run on HPC (Slurm), refer to `Tutorials/` for the HPC tutorial added in the most recent commit.

## Architecture

All model, compression, and attack logic lives in `ProjectNotebook.ipynb`. The architecture follows a **registry + pipeline composition** pattern:

### Core Abstractions

- `Perturbation` (ABC) — base class with abstract `apply()` method; all attacks and compressions implement this
- `PerturbationPipeline` — chains multiple `Perturbation` instances sequentially
- `ModelSpec` (dataclass) — bundles a model loader, `num_classes`, and `input_size`
- `PipelineSpec` (dataclass) — describes a single experimental configuration

### Registries

- **Compression registry**: `JpegCompressionPIL`, `Jpeg2000CompressionPIL`, `PcaPerturbation`, `PatchSVDPerturbation`, `LearnedImageCompression`
- **Attack registry**: `FgsmPerturbation`, `PGDPerturbation`, `AutoPGDPerturbation`
- **Model registry**: ResNet18/34/50, ViT variants — keyed by name, loaded via timm/torchvision

### Execution Flow

1. `build_pipeline_grid()` — generates all attack × compression × parameter combinations
2. `evaluate_pipeline()` — runs a single pipeline, collecting accuracy + image quality metrics (PSNR, MSE, MAE)
3. `run_pipeline_grid()` — batch executor over all combinations; exports to `results.xlsx`

### Datasets

CIFAR-10, CIFAR-100, ImageNet — each has its own loader, normalization constants, and transform chain. Dataset selection is done via config variables at the top of the execution section.

### HuggingFace Authentication

Requires a HuggingFace token. In Kaggle: loaded via `UserSecretsClient`. Locally: set `HF_TOKEN` env var or use `huggingface-cli login`.
