"""
ImageNet adversarial robustness pipeline — HPC/headless runner.

Usage:
    python run_pipeline.py [--imagenet-dir ~/data/shared/imagenet]
                           [--output results.xlsx] [--max-batches N]
                           [--batch-size 32]

    --imagenet-dir  Path to the ImageNet root directory containing a 'val/'
                    subfolder (ImageFolder layout).  On the UoN HPC cluster
                    this is ~/data/shared/imagenet.

    --max-batches   Cap the number of batches processed per pipeline.  Useful
                    for smoke-testing before committing to the full run.
"""

import argparse
import datetime
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from PIL import Image
import pandas as pd

import timm
import compressai
from compressai.zoo import cheng2020_attn

# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Global normalisation tensors (initialised in setup_globals)
# ---------------------------------------------------------------------------

DATA_MEAN: tuple = None
DATA_STD: tuple = None
NUM_CLASSES: int = None
DATASET_NAME: str = None
DATA_MEAN_T: torch.Tensor = None
DATA_STD_T: torch.Tensor = None


def setup_globals(device: torch.device) -> None:
    """Populate ImageNet normalisation constants and move tensors to device."""
    global DATA_MEAN, DATA_STD, NUM_CLASSES, DATASET_NAME
    global DATA_MEAN_T, DATA_STD_T
    DATA_MEAN = (0.485, 0.456, 0.406)
    DATA_STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 1000
    DATASET_NAME = "imagenet"
    DATA_MEAN_T = torch.tensor(DATA_MEAN, device=device).view(1, 3, 1, 1)
    DATA_STD_T = torch.tensor(DATA_STD, device=device).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalised tensor → [0, 1] pixel space."""
    x = x * DATA_STD_T + DATA_MEAN_T
    return torch.clamp(x, 0.0, 1.0)


def renormalize(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] pixel space → normalised tensor."""
    return (x - DATA_MEAN_T) / DATA_STD_T


def project_linf_pixel(
    x_adv_px: torch.Tensor, x0_px: torch.Tensor, eps: float
) -> torch.Tensor:
    x_adv_px = torch.max(torch.min(x_adv_px, x0_px + eps), x0_px - eps)
    return x_adv_px.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def batch_metrics_from_normalized(
    orig_norm: torch.Tensor, pert_norm: torch.Tensor
):
    """MSE, MAE, and PSNR between two batches of normalised images."""
    x0 = denormalize(orig_norm)
    x1 = denormalize(pert_norm)
    mse = torch.mean((x0 - x1) ** 2, dim=(1, 2, 3))
    mae = torch.mean(torch.abs(x0 - x1), dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
    return mse, mae, psnr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    root: str = "./data"


class LocalImageNetDataModule:
    """Uses a locally installed ImageNet directory (ImageFolder layout)."""

    def __init__(self, config: DataConfig, imagenet_dir: str):
        self.config = config
        self.imagenet_dir = imagenet_dir

    def test_loader(self) -> DataLoader:
        val_dir = os.path.join(self.imagenet_dir, "val")
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(DATA_MEAN, DATA_STD),
        ])
        dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    loader: Callable[[], nn.Module]
    num_classes: int
    input_size: int = 224
    dataset: str = "imagenet"


def load_resnet18_imagenet() -> nn.Module:
    return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)


def load_resnet50_imagenet() -> nn.Module:
    return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "resnet18_imagenet": ModelSpec(
        name="resnet18_imagenet",
        loader=load_resnet18_imagenet,
        num_classes=1000,
        input_size=224,
        dataset="imagenet",
    ),
    "resnet50_imagenet": ModelSpec(
        name="resnet50_imagenet",
        loader=load_resnet50_imagenet,
        num_classes=1000,
        input_size=224,
        dataset="imagenet",
    ),
}


def build_model(spec_name: str, device: torch.device) -> nn.Module:
    spec = MODEL_REGISTRY[spec_name]
    model = spec.loader().to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Perturbation base classes
# ---------------------------------------------------------------------------

class Perturbation(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        pass


@dataclass
class PerturbationPipeline:
    name: str
    steps: List[Perturbation]

    def apply(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x = images
        for step in self.steps:
            x = step.apply(model, x, labels, device)
        return x


# ---------------------------------------------------------------------------
# Compression implementations
# ---------------------------------------------------------------------------

class JpegCompressionPIL:
    def __init__(self, quality: int):
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class JpegPerturbation(Perturbation):
    def __init__(self, quality: int):
        super().__init__(name=f"jpeg_q{quality}")
        self.quality = int(quality)
        self.jpeg = JpegCompressionPIL(quality)
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

    def apply(self, model, images, labels, device):
        images = images.detach().to(device)
        pixels = denormalize(images).cpu()
        out = []
        for img in pixels:
            pil_img = self.to_pil(img)
            pil_jpeg = self.jpeg(pil_img)
            out.append(self.to_tensor(pil_jpeg))
        out = torch.stack(out).to(device)
        return renormalize(out)


class Jpeg2000CompressionPIL:
    def __init__(self, quality: float, irreversible: bool = True):
        self.quality = float(quality)
        self.irreversible = bool(irreversible)

    @staticmethod
    def quality_to_rate(q: float) -> float:
        q = float(max(1.0, min(100.0, q)))
        lo, hi = 1.0, 100.0
        t = (100.0 - q) / 99.0
        return lo * ((hi / lo) ** t)

    def __call__(self, img: Image.Image) -> Image.Image:
        rate = self.quality_to_rate(self.quality)
        save_kwargs = {
            "quality_mode": "rates",
            "quality_layers": [rate],
            "irreversible": self.irreversible,
        }
        last_exc = None
        for fmt in ("JPEG2000", "JP2"):
            try:
                with BytesIO() as buffer:
                    img.save(buffer, format=fmt, **save_kwargs)
                    buffer.seek(0)
                    out = Image.open(buffer).convert("RGB")
                    out.load()
                    return out
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            "JPEG2000 support not available in this Pillow build. "
            "Install OpenJPEG + rebuild Pillow or use an alternative backend."
        ) from last_exc


class Jpeg2000Perturbation(Perturbation):
    def __init__(self, quality: float):
        super().__init__(name=f"jpeg2000_q{int(quality)}")
        self.quality = float(quality)
        self.jpeg2000 = Jpeg2000CompressionPIL(quality=float(quality), irreversible=True)
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

    def apply(self, model, images, labels, device):
        images = images.detach().to(device)
        pixels = denormalize(images).cpu()
        out = []
        for img in pixels:
            pil_img = self.to_pil(img)
            pil_jp2 = self.jpeg2000(pil_img)
            out.append(self.to_tensor(pil_jp2))
        out = torch.stack(out).to(device)
        out = torch.clamp(out, 0.0, 1.0)
        return renormalize(out)


class PcaPerturbation(Perturbation):
    def __init__(self, quality: float):
        super().__init__(name=f"pca_q{int(quality)}")
        self.quality = float(quality)

    def apply(self, model, images, labels, device):
        images = images.detach().to(device)
        x = denormalize(images)
        B, C, H, W = x.shape
        k_max = min(H, W)
        k = max(1, int(round(self.quality / 100.0 * k_max)))
        x_flat = x.view(B * C, H, W)
        x_out = torch.empty_like(x_flat)
        for i in range(x_flat.size(0)):
            A = x_flat[i]
            row_mean = A.mean(dim=1, keepdim=True)
            A_centered = A - row_mean
            U, S, Vh = torch.linalg.svd(A_centered, full_matrices=False)
            Uk = U[:, :k]
            Sk = S[:k]
            Vhk = Vh[:k, :]
            Ak = (Uk * Sk) @ Vhk
            x_out[i] = Ak + row_mean
        x_out = x_out.view(B, C, H, W)
        x_out = torch.clamp(x_out, 0.0, 1.0)
        return renormalize(x_out)


class PatchSVDPerturbation(Perturbation):
    def __init__(self, quality: float, patch_size: int = 8):
        super().__init__(name=f"patchsvd_q{int(quality)}_p{patch_size}")
        self.quality = float(quality)
        self.patch_size = int(patch_size)

    def apply(self, model, images, labels, device):
        images = images.detach().to(device)
        x = denormalize(images)
        B, C, H, W = x.shape
        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = x.shape[2], x.shape[3]
        patches = x.unfold(2, p, p).unfold(3, p, p)
        GridH = patches.shape[2]
        GridW = patches.shape[3]
        patches_flat = patches.contiguous().view(-1, p, p)
        U, S, Vh = torch.linalg.svd(patches_flat, full_matrices=False)
        k_max = p
        k = max(1, int(round(self.quality / 100.0 * k_max)))
        Uk = U[:, :, :k]
        Sk = S[:, :k]
        Vhk = Vh[:, :k, :]
        recon_flat = (Uk * Sk.unsqueeze(1)) @ Vhk
        recon_folded = recon_flat.view(B, C, GridH, GridW, p, p)
        recon_permuted = recon_folded.permute(0, 1, 2, 4, 3, 5)
        recon = recon_permuted.contiguous().view(B, C, H_pad, W_pad)
        if pad_h > 0 or pad_w > 0:
            recon = recon[:, :, :H, :W]
        recon = torch.clamp(recon, 0.0, 1.0)
        return renormalize(recon)


# CompressAI model cache
_compressai_cache: Dict[tuple, nn.Module] = {}


def _map_quality_to_lic_level(q: float) -> int:
    mapping = {20: 2, 80: 5}
    if q in mapping:
        return mapping[q]
    level = int(round(1 + (q - 1) * 5 / 99))
    return max(1, min(6, level))


def _get_compressai_model(quality_level: int, device: torch.device) -> nn.Module:
    key = (quality_level, str(device))
    if key not in _compressai_cache:
        net = cheng2020_attn(quality=quality_level, pretrained=True).to(device).eval()
        net.update()
        _compressai_cache[key] = net
    return _compressai_cache[key]


def get_saliency_map(x_norm: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Gradient magnitude w.r.t. predicted class. Returns (B, 1, H, W) in [0,1]."""
    x = x_norm.detach().requires_grad_(True)
    logits = model(x)
    preds = logits.argmax(dim=1)
    loss = logits[torch.arange(logits.size(0), device=logits.device), preds].sum()
    grad = torch.autograd.grad(loss, x, only_inputs=True)[0]
    saliency = grad.abs().mean(dim=1, keepdim=True)
    B = saliency.size(0)
    flat = saliency.view(B, -1)
    mins = flat.min(dim=1).values.view(B, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(B, 1, 1, 1)
    saliency = (saliency - mins) / (maxs - mins + 1e-8)
    return saliency.detach()


class LicRoiPerturbation(Perturbation):
    def __init__(self, quality: float, roi_weight: float = 1.0):
        super().__init__(name=f"lic_roi_q{int(quality)}")
        self.quality = quality
        self.roi_weight = roi_weight
        self.base_level = _map_quality_to_lic_level(quality)
        self.hq_level = min(6, self.base_level + 1)
        self.lq_level = max(1, self.base_level - 1)

    def _compress_decompress(
        self, x_pixel: torch.Tensor, level: int, device: torch.device
    ) -> torch.Tensor:
        net = _get_compressai_model(level, device)
        with torch.no_grad():
            out = net(x_pixel)
        return out["x_hat"].clamp(0.0, 1.0)

    def apply(self, model, images, labels, device):
        images = images.detach().to(device)
        if model is None:
            x_pixel = denormalize(images)
            base_recon = self._compress_decompress(x_pixel, self.base_level, device)
            return renormalize(base_recon)

        mask = get_saliency_map(images, model)
        x_pixel = denormalize(images)

        # Pad to multiple of 64 (CompressAI requirement)
        _, _, H, W = x_pixel.shape
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x_pixel = F.pad(x_pixel, (0, pad_w, 0, pad_h), mode="reflect")

        hq_recon = self._compress_decompress(x_pixel, self.hq_level, device)
        lq_recon = self._compress_decompress(x_pixel, self.lq_level, device)

        if pad_h > 0 or pad_w > 0:
            hq_recon = hq_recon[:, :, :H, :W]
            lq_recon = lq_recon[:, :, :H, :W]

        if mask.shape[2:] != hq_recon.shape[2:]:
            mask = F.interpolate(
                mask, size=hq_recon.shape[2:], mode="bilinear", align_corners=False
            )

        w = (mask * self.roi_weight).clamp(0.0, 1.0)
        blended = w * hq_recon + (1.0 - w) * lq_recon
        return renormalize(blended)


# ---------------------------------------------------------------------------
# Attack implementations
# ---------------------------------------------------------------------------

class FgsmPerturbation(Perturbation):
    def __init__(self, epsilon: float):
        super().__init__(name=f"fgsm_eps{epsilon}")
        self.epsilon = float(epsilon)

    def apply(self, model, images, labels, device):
        model.eval()
        x_norm = images.to(device)
        y = labels.to(device)
        x0_px = denormalize(x_norm).detach()
        x_adv_px = x0_px.clone().detach().requires_grad_(True)
        logits = model(renormalize(x_adv_px))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv_px, only_inputs=True)[0]
        x_adv_px = x_adv_px + self.epsilon * grad.sign()
        x_adv_px = project_linf_pixel(x_adv_px, x0_px, self.epsilon)
        return renormalize(x_adv_px.detach())


class PGDPerturbation(Perturbation):
    def __init__(
        self,
        epsilon: float,
        steps: int = 10,
        alpha: float = None,
        random_start: bool = True,
    ):
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.alpha = float(alpha) if alpha is not None else 2.0 / 255.0
        self.random_start = bool(random_start)
        super().__init__(
            name=f"pgd_eps{self.epsilon}_k{self.steps}_rs{int(self.random_start)}"
        )

    def apply(self, model, images, labels, device):
        model.eval()
        x_norm = images.to(device)
        y = labels.to(device)
        x0_px = denormalize(x_norm).detach()
        if self.random_start:
            noise = torch.empty_like(x0_px).uniform_(-self.epsilon, self.epsilon)
            x_adv_px = project_linf_pixel(x0_px + noise, x0_px, self.epsilon).detach()
        else:
            x_adv_px = x0_px.clone().detach()
        for _ in range(self.steps):
            x_adv_px = x_adv_px.detach().requires_grad_(True)
            logits = model(renormalize(x_adv_px))
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv_px, only_inputs=True)[0]
            with torch.no_grad():
                x_adv_px = x_adv_px + self.alpha * grad.sign()
                x_adv_px = project_linf_pixel(x_adv_px, x0_px, self.epsilon)
        return renormalize(x_adv_px.detach())


def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sorted_logits, indices = torch.sort(logits, dim=1, descending=True)
    z_y = logits[torch.arange(logits.shape[0], device=logits.device), labels]
    z_max1 = sorted_logits[:, 0]
    z_max2 = sorted_logits[:, 1]
    z_max3 = sorted_logits[:, 2]
    z_other = torch.where(indices[:, 0] == labels, z_max2, z_max1)
    return -(z_y - z_other) / (z_max1 - z_max3 + 1e-12)


class APGDPerturbation(Perturbation):
    def __init__(
        self,
        epsilon: float = 8 / 255,
        num_steps: int = 20,
        num_restarts: int = 1,
        loss_type: str = "dlr",
        step_factor: float = 2.0,
    ):
        super().__init__(name=f"apgd_{loss_type}_eps{epsilon}")
        self.epsilon = float(epsilon)
        self.num_steps = int(num_steps)
        self.num_restarts = int(num_restarts)
        self.loss_type = str(loss_type).lower()
        self.step_factor = float(step_factor)

    def apply(self, model, images, labels, device):
        model.eval()
        x_norm0 = images.detach().to(device)
        y = labels.to(device)
        x0_px = denormalize(x_norm0).detach()
        batch_size = x0_px.size(0)
        x_best_px = x0_px.clone()
        loss_best = torch.full((batch_size,), -1e10, device=device)
        c1 = max(1, int(0.22 * self.num_steps))
        c2 = max(1, int(0.75 * self.num_steps))
        checkpoints = {c1, c2}
        for _ in range(self.num_restarts):
            noise = torch.empty_like(x0_px).uniform_(-self.epsilon, self.epsilon)
            x_px = (x0_px + noise).clamp(0.0, 1.0)
            x_px = project_linf_pixel(x_px, x0_px, self.epsilon)
            eta = self.step_factor * self.epsilon
            for i in range(self.num_steps):
                x_px = x_px.detach().requires_grad_(True)
                logits = model(renormalize(x_px))
                if self.loss_type == "ce":
                    loss_indiv = F.cross_entropy(logits, y, reduction="none")
                else:
                    loss_indiv = dlr_loss(logits, y)
                loss = loss_indiv.sum()
                grad = torch.autograd.grad(loss, x_px, only_inputs=True)[0]
                with torch.no_grad():
                    improved = loss_indiv > loss_best
                    loss_best[improved] = loss_indiv[improved]
                    x_best_px[improved] = x_px[improved].detach()
                    x_px = x_px + eta * grad.sign()
                    x_px = project_linf_pixel(x_px, x0_px, self.epsilon)
                    if (i + 1) in checkpoints:
                        eta *= 0.5
        return renormalize(x_best_px.detach())


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

@dataclass
class CompressionSpec:
    name: str
    make: Callable[[float], Perturbation]


compression_registry: Dict[str, CompressionSpec] = {
    "jpeg": CompressionSpec(
        name="jpeg",
        make=lambda q: JpegPerturbation(quality=int(q)),
    ),
    "jpeg2000": CompressionSpec(
        name="jpeg2000",
        make=lambda q: Jpeg2000Perturbation(quality=float(q)),
    ),
    "pca": CompressionSpec(
        name="pca",
        make=lambda q: PcaPerturbation(quality=float(q)),
    ),
    "patchsvd": CompressionSpec(
        name="patchsvd",
        make=lambda q: PatchSVDPerturbation(quality=float(q), patch_size=8),
    ),
    "lic_roi": CompressionSpec(
        name="lic_roi",
        make=lambda q: LicRoiPerturbation(quality=float(q)),
    ),
}


@dataclass
class AttackSpec:
    name: str
    make: Callable[[float], Perturbation]


attack_registry: Dict[str, AttackSpec] = {
    "fgsm": AttackSpec(
        name="fgsm",
        make=lambda eps: FgsmPerturbation(epsilon=eps),
    ),
    "pgd": AttackSpec(
        name="pgd",
        make=lambda eps: PGDPerturbation(epsilon=eps),
    ),
    "apgd": AttackSpec(
        name="apgd",
        make=lambda eps: APGDPerturbation(
            epsilon=eps, num_steps=20, num_restarts=1, loss_type="dlr"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

@dataclass
class PipelineSpec:
    name: str
    pipeline: PerturbationPipeline
    compression: Optional[str] = None
    attack: Optional[str] = None
    quality: Optional[float] = None
    epsilon: Optional[float] = None


def build_pipeline_grid(
    compression_keys: List[str],
    attack_keys: List[str],
    qualities: List[float],
    epsilons: List[float],
) -> List[PipelineSpec]:
    specs: List[PipelineSpec] = []

    # clean
    specs.append(
        PipelineSpec(
            name="clean",
            pipeline=PerturbationPipeline(name="clean", steps=[]),
        )
    )

    # compression-only
    for ck in compression_keys:
        comp_spec = compression_registry[ck]
        for q in qualities:
            comp = comp_spec.make(q)
            name = f"{comp_spec.name}    ({int(q)}%)"
            specs.append(
                PipelineSpec(
                    name=name,
                    pipeline=PerturbationPipeline(name=name, steps=[comp]),
                    compression=comp_spec.name,
                    quality=q,
                )
            )

    # attack-only
    for ak in attack_keys:
        atk_spec = attack_registry[ak]
        for eps in epsilons:
            atk = atk_spec.make(eps)
            name = f"{atk_spec.name}    (eps={eps:.2f})"
            specs.append(
                PipelineSpec(
                    name=name,
                    pipeline=PerturbationPipeline(name=name, steps=[atk]),
                    attack=atk_spec.name,
                    epsilon=eps,
                )
            )

    # compression -> attack
    for ck in compression_keys:
        comp_spec = compression_registry[ck]
        for ak in attack_keys:
            atk_spec = attack_registry[ak]
            for q in qualities:
                for eps in epsilons:
                    comp = comp_spec.make(q)
                    atk = atk_spec.make(eps)
                    name = f"{comp_spec.name} -> {atk_spec.name}    ({int(q)}%, eps={eps:.2f})"
                    specs.append(
                        PipelineSpec(
                            name=name,
                            pipeline=PerturbationPipeline(name=name, steps=[comp, atk]),
                            compression=comp_spec.name,
                            attack=atk_spec.name,
                            quality=q,
                            epsilon=eps,
                        )
                    )

    # attack -> compression
    for ak in attack_keys:
        atk_spec = attack_registry[ak]
        for ck in compression_keys:
            comp_spec = compression_registry[ck]
            for eps in epsilons:
                for q in qualities:
                    atk = atk_spec.make(eps)
                    comp = comp_spec.make(q)
                    name = f"{atk_spec.name} -> {comp_spec.name}    (eps={eps:.2f}, {int(q)}%)"
                    specs.append(
                        PipelineSpec(
                            name=name,
                            pipeline=PerturbationPipeline(name=name, steps=[atk, comp]),
                            compression=comp_spec.name,
                            attack=atk_spec.name,
                            quality=q,
                            epsilon=eps,
                        )
                    )

    return specs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(model, images, device):
    return model(images.to(device))


def evaluate_pipeline(
    model,
    dataloader,
    pipeline: PerturbationPipeline,
    device,
    max_batches=None,
    track_metrics: bool = False,
):
    model.eval()
    total = 0
    correct = 0
    psnr_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    metric_count = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        if pipeline is not None and len(pipeline.steps) > 0:
            with torch.enable_grad():
                perturbed = pipeline.apply(model, images, labels, device)
        else:
            perturbed = images

        if track_metrics:
            batch_mse, batch_mae, batch_psnr = batch_metrics_from_normalized(
                images, perturbed
            )
            mse_sum += batch_mse.sum().item()
            mae_sum += batch_mae.sum().item()
            psnr_sum += batch_psnr.sum().item()
            metric_count += batch_mse.numel()

        with torch.no_grad():
            logits = model(perturbed)
            preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "psnr_mean": psnr_sum / metric_count if metric_count > 0 else None,
        "mse_mean": mse_sum / metric_count if metric_count > 0 else None,
        "mae_mean": mae_sum / metric_count if metric_count > 0 else None,
    }
    return result


def run_pipeline_grid(
    model_names: List[str],
    pipeline_specs: List[PipelineSpec],
    dataloader,
    device,
    max_batches=None,
) -> pd.DataFrame:
    records = []
    for model_name in model_names:
        print(f"\n----- Model: {model_name} -----", flush=True)
        model = build_model(model_name, device)
        for p in pipeline_specs:
            print(
                f"  Pipeline: {p.name} - started {datetime.datetime.now().time()}",
                flush=True,
            )
            res = evaluate_pipeline(
                model=model,
                dataloader=dataloader,
                pipeline=p.pipeline,
                device=device,
                max_batches=max_batches,
                track_metrics=True,
            )
            print(f"   - Accuracy: {res['accuracy']*100:.2f}%", flush=True)
            print(f"   - PSNR:     {res['psnr_mean']}", flush=True)
            print(f"   - MSE:      {res['mse_mean']}", flush=True)
            print(f"   - MAE:      {res['mae_mean']}", flush=True)
            print(f"   - Total:    {res['total']}", flush=True)
            print(f"   - Correct:  {res['correct']}\n", flush=True)
            records.append(
                {
                    "model": model_name,
                    "pipeline": p.name,
                    "compression": p.compression,
                    "attack": p.attack,
                    "quality": p.quality,
                    "epsilon": p.epsilon,
                    "accuracy": res["accuracy"],
                    "correct": res["correct"],
                    "total": res["total"],
                    "psnr_mean": res["psnr_mean"],
                    "mse_mean": res["mse_mean"],
                    "mae_mean": res["mae_mean"],
                }
            )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run adversarial robustness pipeline on ImageNet."
    )
    p.add_argument(
        "--imagenet-dir",
        default="/data/shared/imagenet",
        dest="imagenet_dir",
        help=(
            "Path to ImageNet root directory containing a 'val/' subfolder "
            "(ImageFolder layout).  Default: /data/shared/imagenet"
        ),
    )
    p.add_argument(
        "--output",
        default="results.xlsx",
        help="Output .xlsx file path (default: results.xlsx)",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        dest="max_batches",
        help="Cap number of batches per pipeline (default: all)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="DataLoader batch size (default: 32)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Device + seed
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 2. Normalisation globals (ImageNet only)
    setup_globals(device)

    # 3. Data loader
    print(f"Using local ImageNet at: {args.imagenet_dir}", flush=True)
    data_cfg = DataConfig(batch_size=args.batch_size, num_workers=4)
    data_module = LocalImageNetDataModule(data_cfg, args.imagenet_dir)

    base_test_loader = data_module.test_loader()

    # 4. Runner config
    compression_keys = ["jpeg", "pca", "patchsvd", "jpeg2000", "lic_roi"]
    attack_keys = ["fgsm", "pgd", "apgd"]
    qualities = [25, 50, 75]
    epsilons = [8 / 255]
    model_names = ["resnet18_imagenet", "resnet50_imagenet"]

    # 5. Build and run
    pipeline_specs = build_pipeline_grid(
        compression_keys=compression_keys,
        attack_keys=attack_keys,
        qualities=qualities,
        epsilons=epsilons,
    )
    print(f"Total pipelines: {len(pipeline_specs)}", flush=True)

    results_df = run_pipeline_grid(
        model_names=model_names,
        pipeline_specs=pipeline_specs,
        dataloader=base_test_loader,
        device=device,
        max_batches=args.max_batches,
    )

    # 6. Save
    results_df.to_excel(args.output, sheet_name="Sheet1", index=False)
    print(f"\nResults saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
