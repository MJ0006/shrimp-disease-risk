"""Backbone factory and deployment-cost measurement.

Two responsibilities, deliberately kept together because they answer one question: which
backbone, and what does it cost to run?

The registered project title promises an **edge-deployable** framework. "Edge-deployable" is a
measurable claim, not an adjective, so `deployment_profile` reports parameter count, on-disk
size and inference latency for every candidate. A model comparison that reports only accuracy
cannot support the title.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import timm
import torch
import torch.nn as nn

# Candidate backbones for the Experiment 03 comparison.
#
# Selection reasoning, not a grab-bag:
#   rexnet_150          -- the inherited notebook's choice; included so the baseline is
#                          reproduced with the same architecture rather than a substitute.
#   efficientnet_b0     -- the standard efficiency baseline; also named (loosely) in the
#                          reference deck, so it settles what that claim would have given.
#   mobilenetv3_100     -- genuinely edge-class; the honest test of the title's promise.
#   convnext_tiny       -- ShrimpXNet reports its best result with this; needed for a
#                          like-for-like comparison against published work.
#   resnet50            -- the field's default reference point; if a modern backbone cannot
#                          beat it, that is worth knowing.
CANDIDATES = {
    "rexnet_150": "inherited baseline architecture",
    "efficientnet_b0": "standard efficiency baseline",
    "mobilenetv3_large_100": "edge-class backbone",
    "convnext_tiny": "ShrimpXNet's reported best",
    "resnet50": "conventional reference point",
}


@dataclass(frozen=True)
class DeploymentProfile:
    """Measured cost of running a model -- the evidence behind 'edge-deployable'."""

    model_name: str
    total_params: int
    trainable_params: int
    size_mb: float
    latency_ms_mean: float
    latency_ms_std: float
    device: str

    def as_dict(self) -> dict:
        return asdict(self)


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Build a timm backbone with a fresh classification head.

    `pretrained=True` by default because 416 independent specimens is far too few to train a
    modern backbone from scratch -- ImageNet features are doing most of the work here, and
    saying so is more honest than implying the network learned shrimp pathology from 800
    training images.

    `num_classes` replaces timm's ImageNet head with a 4-way head automatically.
    """
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 50,
) -> tuple[float, float]:
    """Time a single forward pass, in milliseconds.

    Batch size 1 because edge inference is per-image, not batched -- reporting batched
    throughput would flatter the model relative to how it would actually be used at a pond.

    Warm-up iterations are discarded: the first CUDA calls include context setup and kernel
    autotuning, which would otherwise dominate the mean. `torch.cuda.synchronize` is required
    before reading the clock because CUDA launches are asynchronous and would otherwise be
    timed as near-instant.
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def deployment_profile(
    model_name: str,
    device: torch.device,
    num_classes: int = 4,
    img_size: int = 224,
) -> DeploymentProfile:
    """Full deployment cost for one backbone.

    `size_mb` is computed from parameter and buffer bytes rather than by saving a checkpoint,
    so it reflects the model itself and not optimiser state or serialisation overhead. This is
    the figure comparable to FeatherNetX's published 2.82 MB.
    """
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    total, trainable = count_parameters(model)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

    mean_ms, std_ms = measure_latency(model, device, img_size=img_size)

    return DeploymentProfile(
        model_name=model_name,
        total_params=total,
        trainable_params=trainable,
        size_mb=round(size_mb, 3),
        latency_ms_mean=round(mean_ms, 3),
        latency_ms_std=round(std_ms, 3),
        device=str(device),
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
