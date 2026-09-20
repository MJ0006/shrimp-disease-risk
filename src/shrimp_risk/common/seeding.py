"""Determinism controls.

A result whose seed and environment were not recorded is not a result -- it cannot be
reproduced, compared against, or defended. Every experiment in this project calls
`seed_everything` and records the value it used alongside its metrics.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch, and optionally force deterministic cuDNN.

    `deterministic=True` disables cuDNN's autotuner. That costs some speed, but the
    autotuner picks different convolution algorithms run to run, which makes two runs of
    identical code differ in the third decimal place. When the whole point of an experiment
    is comparing two conditions, that noise is not acceptable.

    Returns the seed so callers can log it without re-reading their own config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # allows non-torch stages to import this module
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def environment_fingerprint() -> dict[str, str]:
    """Capture what the run actually executed on.

    Recorded with every experiment so a number can be traced to the software and hardware
    that produced it. Library versions matter: the same code on a different torch build can
    give different results.
    """
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or ""
    except ImportError:
        info["torch"] = "not installed"
    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        pass
    return info
