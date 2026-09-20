"""Shared fixtures.

Two rules shape this file.

1. **Tests that need the 1,149-image corpus must skip, not fail, when it is absent.** The
   archive is ~500 MB and is not redistributed with the code. A missing dataset means "not
   checkable here", which is different from "broken".
2. **Fixtures are session-scoped.** The manifest is read once; parsing it per test would
   dominate the runtime of the whole suite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "image_manifest.csv"
SUMMARY_XLSX = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "images"
    / "ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of "
    "Shrimp Diseases in Bangladesh"
    / "Root"
    / "ShrimpDiseaseBD_Summary.xlsx"
)

# Published composition (Islam et al. 2025, Data in Brief 60:111553). Hard-coded on purpose:
# these are the numbers the documentation and the PPT quote, so if the data on disk ever stops
# matching them, the tests must fail rather than quietly adopt the new values.
PUBLISHED_CLASS_COUNTS = {"Healthy": 403, "BG": 198, "WSSV": 328, "BG_WSSV": 220}
PUBLISHED_TOTAL_IMAGES = 1149
PUBLISHED_TOTAL_SPECIMENS = 416
PUBLISHED_ANNOTATED = 746


@pytest.fixture(scope="session")
def manifest() -> pd.DataFrame:
    """The full manifest, raw + annotated rows."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"image manifest not found at {MANIFEST_PATH}; run scripts/extract_images.py")
    return pd.read_csv(MANIFEST_PATH)


@pytest.fixture(scope="session")
def raw_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """The classification subset only -- what every split in the project is built from."""
    return manifest[manifest["subset"] == "raw"].reset_index(drop=True)
