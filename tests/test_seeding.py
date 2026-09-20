"""Determinism controls.

The project reports differences of 0.15 macro-F1 between conditions and 0.0004 between seeds.
Numbers that fine are only interpretable if seeding actually works, so the guarantee is tested
rather than assumed.
"""

from __future__ import annotations

import os
import random

import numpy as np

from shrimp_risk.common.seeding import environment_fingerprint, seed_everything


def test_seed_everything_returns_the_seed_it_set():
    """Callers log the return value instead of re-reading their own config, so it has to be
    the seed that was actually applied."""
    assert seed_everything(2025) == 2025


def test_same_seed_gives_the_same_draws():
    seed_everything(42)
    a = (random.random(), np.random.rand(5).tolist())
    seed_everything(42)
    b = (random.random(), np.random.rand(5).tolist())
    assert a == b


def test_different_seeds_give_different_draws():
    """Guards against a seed that is accepted and ignored -- which would silently turn the
    three-seed stability result into three copies of one run."""
    seed_everything(42)
    a = np.random.rand(10).tolist()
    seed_everything(7)
    b = np.random.rand(10).tolist()
    assert a != b


def test_pythonhashseed_is_set():
    """Set for subprocesses and for any dict/set ordering that reaches a stored artefact."""
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_environment_fingerprint_records_what_the_run_executed_on():
    env = environment_fingerprint()
    assert env["python"].count(".") == 2
    assert env["platform"]
    assert "torch" in env  # either a version or the explicit "not installed"


def test_fingerprint_reports_gpu_details_when_cuda_is_available():
    """On this machine the runs are GPU-bound and the GPU model is part of the reported
    environment. When CUDA is absent the keys are legitimately missing, so the test asserts
    the conditional rather than the hardware."""
    env = environment_fingerprint()
    if env.get("cuda_available") == "True":
        assert env.get("gpu")
        assert env.get("cuda")
