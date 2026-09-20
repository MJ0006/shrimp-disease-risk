"""Training and evaluation loop.

Departures from the inherited notebook, and why each one exists
---------------------------------------------------------------
1. **Model selection on macro-F1, not accuracy.** Classes are imbalanced 403/328/220/198.
   Accuracy lets a model coast on Healthy; macro-F1 weights every class equally, so failing
   the 198-image BG class is visible rather than absorbed.
2. **Predictions are returned, not just metrics.** Prevalence quantification downstream needs
   per-image posteriors, and a confusion matrix cannot be reconstructed from a summary number.
   Validation posteriors are returned too, scored on the *restored* best weights rather than
   collected during training, because Stage 8 fits a calibration temperature on them and that
   temperature has to belong to the model that is actually reported.
3. **Every run records its own environment and config.** A metric with no recorded seed,
   protocol and library version cannot be compared against another run honestly.
4. **Early stopping restores the best weights.** The notebook saved a checkpoint but then
   evaluated whatever state training ended in; those are different models.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .datasets import CLASSES


@dataclass
class TrainConfig:
    """One experiment's controlled conditions.

    Grouped into a single object so a run's settings can be serialised alongside its results.
    Comparing two runs whose configs were not recorded is guesswork.
    """

    model_name: str = "rexnet_150"
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    img_size: int = 224
    patience: int = 7
    augment: bool = True
    class_weighted_loss: bool = False
    label_smoothing: float = 0.0
    seed: int = 2025
    protocol: str = "specimen_level"
    fold: int = 0
    amp: bool = True

    # Where to persist the weights early stopping selected, if anywhere.
    #
    # This exists because the weights used to be discarded. `best_state` lived in memory for the
    # duration of the process and was never written, so the moment training returned, the trained
    # network was gone. Every score in `reports/tables/` was still real -- they are computed
    # before the process exits -- but nothing downstream could ever load the model that produced
    # them. `export_edge_model.py` consequently exported a randomly initialised backbone and
    # called it the deployed model; on four-class softmax it returned ~0.25 for everything.
    #
    # Left as None the behaviour is exactly as before, so the 34 stored runs reproduce unchanged.
    checkpoint_path: str | None = None

    # Three switches that exist for one reason: Experiment 02b has to reproduce the inherited
    # notebook faithfully, and the notebook does none of these three things the way this project
    # does. Every default below is the behaviour every run before 2026-09-06 used, so the 34
    # stored runs are unaffected and reproduce unchanged.
    #
    #   optimizer         the notebook uses plain Adam; this project uses AdamW
    #   scheduler         the notebook uses none; this project cosine-anneals
    #   selection_metric  the notebook selects on torchmetrics' default multiclass F1, which is
    #                     MICRO-averaged -- i.e. accuracy. This project selects on macro-F1
    #                     precisely because accuracy lets a model coast on the majority class.
    #
    # They are configuration, not alternatives on equal footing: the defaults are what the project
    # argues for, and the other settings exist to reproduce a baseline it argues against (`D-010`).
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    selection_metric: str = "macro_f1"


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_f1: float
    val_loss: float
    val_f1: float
    val_acc: float
    seconds: float


@dataclass
class RunResult:
    config: TrainConfig
    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    test_acc: float = 0.0
    test_f1_macro: float = 0.0
    test_f1_weighted: float = 0.0
    per_class: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)
    test_probs: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    val_probs: np.ndarray | None = None
    val_labels: np.ndarray | None = None
    train_seconds: float = 0.0


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over a loader. Trains if an optimizer is supplied, else evaluates.

    Sharing one function between train and eval keeps the two paths from drifting apart — a
    subtle difference in how loss is reduced between them would make train/val curves
    incomparable.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    return total_loss / max(n, 1), probs, labels


def train_and_evaluate(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    device: torch.device,
    class_names: list[str],
    loss_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> RunResult:
    """Train with early stopping on validation macro-F1, then evaluate once on test.

    The test set is touched exactly once, after training and model selection are complete.
    Selecting on test — even indirectly, by rerunning with different settings and keeping the
    best test score — turns the test set into a validation set and makes the final number
    optimistic. That failure mode is precisely what this project is about, so the code makes
    it structurally awkward to commit.
    """
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if loss_weights is not None else None,
        label_smoothing=cfg.label_smoothing,
    )
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        # Plain Adam, and weight decay is NOT passed. Adam's `weight_decay` is L2 added to the
        # gradient, which is a different regulariser from AdamW's decoupled decay -- passing the
        # project's 1e-4 here would silently make this a third thing that is neither the
        # notebook's setting nor this project's.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected 'adamw' or 'adam'")

    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}; expected 'cosine' or 'none'")

    if cfg.selection_metric not in ("macro_f1", "micro_f1"):
        raise ValueError(
            f"unknown selection_metric {cfg.selection_metric!r}; expected 'macro_f1' or 'micro_f1'"
        )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    result = RunResult(config=cfg)
    best_state = copy.deepcopy(model.state_dict())
    best_f1, stale = -1.0, 0
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_probs, tr_labels = _run_epoch(
            model, loaders["train"], loss_fn, device, optimizer, scaler
        )
        va_loss, va_probs, va_labels = _run_epoch(model, loaders["val"], loss_fn, device)
        if scheduler is not None:
            scheduler.step()

        tr_f1 = f1_score(tr_labels, tr_probs.argmax(1), average="macro", zero_division=0)
        va_f1 = f1_score(va_labels, va_probs.argmax(1), average="macro", zero_division=0)
        va_acc = accuracy_score(va_labels, va_probs.argmax(1))

        result.history.append(EpochRecord(
            epoch=epoch, train_loss=tr_loss, train_f1=tr_f1,
            val_loss=va_loss, val_f1=va_f1, val_acc=va_acc,
            seconds=time.perf_counter() - t0,
        ))

        if verbose:
            print(f"  epoch {epoch:3d}  train loss {tr_loss:.4f} f1 {tr_f1:.4f}  "
                  f"| val loss {va_loss:.4f} f1 {va_f1:.4f} acc {va_acc:.4f}")

        # Micro-averaged F1 over all classes equals accuracy, which is exactly what the
        # notebook's torchmetrics default computes. Naming it `micro_f1` keeps the config
        # readable as "what the notebook selected on" rather than requiring the reader to know
        # that identity.
        selection_score = va_f1 if cfg.selection_metric == "macro_f1" else va_acc

        if selection_score > best_f1:
            best_f1, stale = selection_score, 0
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch, result.best_val_f1 = epoch, selection_score
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (no val F1 gain for {stale})")
                break

    result.train_seconds = time.perf_counter() - t_start

    # Restore the selected model. Evaluating the final-epoch weights instead would report a
    # different model from the one selection chose.
    model.load_state_dict(best_state)

    # Persist those weights if the caller asked for them. The payload carries the config and the
    # selection score alongside the tensors, because a checkpoint whose provenance is unrecorded
    # cannot be matched back to the run that reported its score -- and a checkpoint that cannot be
    # matched to a score is not evidence of anything.
    if cfg.checkpoint_path:
        ckpt_path = Path(cfg.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "model_name": cfg.model_name,
                "num_classes": len(CLASSES),
                "img_size": cfg.img_size,
                "best_epoch": result.best_epoch,
                "best_val_f1": result.best_val_f1,
                "selection_metric": cfg.selection_metric,
                "config": asdict(cfg),
            },
            ckpt_path,
        )
        if verbose:
            print(f"  checkpoint -> {ckpt_path}")

    # Validation is scored a second time, here, on the restored weights.
    #
    # The loop above already produced validation posteriors every epoch, so re-running looks
    # wasteful. It is not the same quantity. Each epoch's posteriors came from the model as it
    # stood *during* that epoch; the model that ends up being reported is the one early stopping
    # selected, and after `load_state_dict` those two are different networks whenever the best
    # epoch was not the last. Downstream calibration fits a temperature on these posteriors and
    # then applies it to the test posteriors of *this* model -- so they have to come from this
    # model, or the temperature corrects a network nobody is using.
    #
    # The validation fold is the only set that can carry this. Fitting on test would tune a
    # parameter on the set the score is reported from, which is the failure mode this whole
    # project is about; fitting on train would see data the model has already memorised. One
    # extra forward pass over ~180 images is the cheapest part of any run here.
    _, va_probs_final, va_labels_final = _run_epoch(model, loaders["val"], loss_fn, device)
    result.val_probs = va_probs_final
    result.val_labels = va_labels_final

    _, te_probs, te_labels = _run_epoch(model, loaders["test"], loss_fn, device)
    preds = te_probs.argmax(1)

    result.test_acc = float(accuracy_score(te_labels, preds))
    result.test_f1_macro = float(f1_score(te_labels, preds, average="macro", zero_division=0))
    result.test_f1_weighted = float(
        f1_score(te_labels, preds, average="weighted", zero_division=0)
    )
    result.per_class = classification_report(
        te_labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    result.confusion = confusion_matrix(
        te_labels, preds, labels=list(range(len(class_names)))
    ).tolist()
    result.test_probs = te_probs
    result.test_labels = te_labels

    if verbose:
        print(f"  TEST  acc {result.test_acc:.4f}  macro-F1 {result.test_f1_macro:.4f}  "
              f"(best epoch {result.best_epoch}, val F1 {result.best_val_f1:.4f})")

    return result
