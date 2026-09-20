"""Dataset and image transforms for the shrimp classification track.

Design notes
------------
The dataset reads from the manifest, never by walking directories. Directory walks return
files in filesystem order, which differs between machines and silently changes which image
sits at index *i*. Since splits are stored as manifest row indices, a stable ordering is not a
convenience here -- it is what makes a split mean the same thing on two different runs.

Augmentation is applied to the training partition only. The reference notebook applied none at
all, which is one of the four hypotheses for its train-0.973 / test-0.638 collapse, so
augmentation is exposed as a switch rather than hard-wired: Experiment 02 needs to run both
with and without it to attribute the effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ImageNet statistics. Used because every backbone here is ImageNet-pretrained, so inputs must
# be normalised the way those weights expect. Computing dataset-specific statistics instead
# would mismatch the pretrained filters for no benefit at this dataset size.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Fixed, sorted class order so label integers mean the same thing across runs, folds and
# saved checkpoints. Deriving order from data would make label 2 depend on which rows a fold
# happened to contain.
CLASSES = ["BG", "BG_WSSV", "Healthy", "WSSV"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def build_transforms(img_size: int = 224, train: bool = False, augment: bool = True):
    """Compose the preprocessing pipeline.

    Evaluation path is deliberately minimal -- resize and normalise only. Any randomness at
    evaluation time would make a score depend on the draw rather than the model.

    The training augmentations are chosen for what actually varies in this dataset: the photos
    are handheld smartphone shots of specimens on varying backgrounds, so orientation and
    illumination differ shot to shot while the disease signal (white spots, gill darkening) is
    a colour/texture pattern on the body. Flips and small rotations therefore add realistic
    variation; heavy colour distortion is avoided because hue IS the signal for Black Gill.
    """
    if not train or not augment:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        # Mild jitter only: saturation/hue shifts large enough to recolour a darkened gill
        # would destroy the very feature that separates BG from Healthy.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])


class ShrimpImageDataset(Dataset):
    """Serves (image_tensor, label) pairs for a set of manifest rows.

    Holds the manifest subset rather than a list of paths so that specimen ids and class names
    stay attached to each sample. Downstream analysis -- per-specimen aggregation, prevalence
    quantification -- needs them, and re-deriving them from filenames later would duplicate
    parsing logic that already lives in the extraction script.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: np.ndarray,
        project_root: Path,
        img_size: int = 224,
        train: bool = False,
        augment: bool = True,
    ) -> None:
        self.rows = manifest.iloc[indices].reset_index(drop=True)
        self.project_root = Path(project_root)
        self.tf = build_transforms(img_size, train=train, augment=augment)

        unknown = set(self.rows["class"]) - set(CLASS_TO_IDX)
        if unknown:
            raise ValueError(f"manifest contains unknown classes: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[i]
        # convert("RGB") guards against greyscale or CMYK files, which would otherwise yield
        # a tensor with the wrong channel count and fail deep inside the backbone.
        img = Image.open(self.project_root / row["path"]).convert("RGB")
        return self.tf(img), CLASS_TO_IDX[row["class"]]

    @property
    def specimen_ids(self) -> np.ndarray:
        return self.rows["specimen_id"].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.rows["class"].map(CLASS_TO_IDX).to_numpy()


def class_weights(manifest: pd.DataFrame, indices: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for the loss, computed on the TRAINING partition only.

    Computed on training rows alone because using the full dataset would leak the test set's
    class distribution into training -- a small leak, but this project's whole argument is that
    small leaks matter.

    Imbalance here is moderate (403 Healthy vs 198 BG, ratio ~2.0), so weighting is offered as
    an option rather than applied by default; Experiment 02 tests whether it helps.
    """
    counts = manifest.iloc[indices]["class"].value_counts()
    w = torch.tensor(
        [len(indices) / (len(CLASSES) * counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float32,
    )
    return w


def make_loaders(
    manifest: pd.DataFrame,
    split,
    project_root: Path,
    batch_size: int = 32,
    img_size: int = 224,
    augment: bool = True,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build train/val/test loaders for one split.

    Validation and test use the evaluation transform and are never shuffled, so predictions
    come back in manifest order and can be joined straight back to specimen ids.
    """
    common = dict(project_root=project_root, img_size=img_size)
    loaders = {}

    for name, train_mode in (("train", True), ("val", False), ("test", False)):
        ds = ShrimpImageDataset(
            manifest,
            getattr(split, name),
            train=train_mode,
            augment=augment and train_mode,
            **common,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train_mode,
            num_workers=num_workers,
            pin_memory=True,
            # Dropping a last batch of size 1 avoids a BatchNorm crash in train mode; harmless
            # at evaluation where the batch is not normalised across samples.
            drop_last=train_mode and len(ds) % batch_size == 1,
            persistent_workers=num_workers > 0,
        )
    return loaders
