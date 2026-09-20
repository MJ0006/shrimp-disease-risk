"""Canonical project paths.

Centralised so no module hard-codes a relative path and silently breaks when run from a
different working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

IMAGE_MANIFEST = DATA_INTERIM / "image_manifest.csv"

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
MODELS = REPORTS / "models"

EXPERIMENTS = PROJECT_ROOT / "experiments"
RUNS = EXPERIMENTS / "runs"


def resolve_image_manifest(prefer_cache_size: int | None = 256):
    """Return the manifest to train from, preferring the pre-resized cache.

    Source images are 2048x2048; training uses 224x224. Decoding 4.19 MP per sample per epoch
    makes JPEG decode the bottleneck and leaves the GPU idle. `build_image_cache.py` writes a
    256px cache plus a parallel manifest whose `path` column points at it, keeping a
    `source_path` column so provenance back to `data/raw/` is never lost.

    Falls back to the full-resolution manifest if the cache has not been built, so nothing
    breaks on a fresh checkout -- it simply runs slower.
    """
    if prefer_cache_size is not None:
        cached = DATA_INTERIM / f"image_manifest_cached_{prefer_cache_size}.csv"
        if cached.exists():
            return cached
    return IMAGE_MANIFEST
