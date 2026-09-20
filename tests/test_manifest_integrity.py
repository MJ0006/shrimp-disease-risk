"""Manifest integrity: does the data on disk still match what the documentation claims?

Every number in the Master Document, the decision log and the Review-2 slides -- 1,149 images,
416 specimens, 403/198/328/220, 24/51/341 group sizes -- is read off this manifest. These
tests exist so that if the extraction is ever re-run against a different archive (the raw
directory holds four candidate zips), the mismatch surfaces here rather than in a viva.

The strongest test in this file is `test_manifest_matches_official_summary_spreadsheet`. It
checks our manifest against `ShrimpDiseaseBD_Summary.xlsx` -- a file the dataset authors ship
inside the Mendeley download, which enumerates all 416 individual shrimp and the images
belonging to each. That file is the external authority for the specimen structure, and it is
why `D-011` describes the grouping as confirmed rather than inferred.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import (
    PUBLISHED_ANNOTATED,
    PUBLISHED_CLASS_COUNTS,
    PUBLISHED_TOTAL_IMAGES,
    PUBLISHED_TOTAL_SPECIMENS,
    SUMMARY_XLSX,
)

pytestmark = pytest.mark.dataset

# The co-infection class is spelled BG_WSSV in the Mendeley folder tree and WSSV_BG in the
# authors' summary spreadsheet. `canonical_class` normalises to BG_WSSV; the alias is declared
# here so the cross-check compares like with like instead of silently reporting a mismatch.
SUMMARY_CLASS_ALIASES = {"WSSV_BG": "BG_WSSV"}


def test_total_image_count(raw_manifest):
    assert len(raw_manifest) == PUBLISHED_TOTAL_IMAGES


def test_per_class_counts_match_the_publication(raw_manifest):
    counts = raw_manifest["class"].value_counts().to_dict()
    assert counts == PUBLISHED_CLASS_COUNTS


def test_specimen_count(raw_manifest):
    assert raw_manifest["specimen_id"].nunique() == PUBLISHED_TOTAL_SPECIMENS


def test_group_size_distribution(raw_manifest):
    """341 specimens photographed 3 times, 51 twice, 24 once -- mean 2.76 images per shrimp.

    This distribution is the reason image-level splitting leaks so heavily: with most
    specimens contributing three images, a 70% training fraction almost always captures at
    least one sibling of any given test image.
    """
    sizes = raw_manifest.groupby("specimen_id").size().value_counts().to_dict()
    assert sizes == {3: 341, 2: 51, 1: 24}


def test_no_specimen_spans_two_classes(raw_manifest):
    """If one specimen carried two class labels, grouping and stratification would conflict
    and the group-aware splitter would be splitting on an incoherent key."""
    per_specimen = raw_manifest.groupby("specimen_id")["class"].nunique()
    assert per_specimen.max() == 1


def test_no_byte_identical_duplicates(raw_manifest):
    """Zero exact duplicates. Recorded because it is the reason a hash-based duplicate check
    -- the standard defence -- finds nothing here. The leakage is near-duplication by
    construction, visible only through specimen identity."""
    assert int(raw_manifest["sha256"].duplicated().sum()) == 0


def test_annotated_subset_is_the_three_diseased_classes(manifest):
    annotated = manifest[manifest["subset"] == "annotated"]
    assert len(annotated) == PUBLISHED_ANNOTATED
    assert set(annotated["class"]) == {"BG", "WSSV", "BG_WSSV"}


def test_annotated_images_reuse_the_same_specimens(manifest, raw_manifest):
    """The 746 annotated images are re-published photographs of specimens already in the raw
    set -- 274 of the 416, being every diseased specimen. Training on both without grouping
    would leak; `load_raw_manifest` drops them, and this test records why that is necessary
    rather than tidy."""
    annotated = manifest[manifest["subset"] == "annotated"]
    assert set(annotated["specimen_id"]) <= set(raw_manifest["specimen_id"])
    assert annotated["specimen_id"].nunique() == 274


def test_annotated_images_are_byte_identical_republications(manifest, raw_manifest):
    """Stronger than "same specimens": every annotated file has the SAME SHA-256 as a raw file.

    The annotated directory is not a second photo shoot. It is the same 746 diseased-class
    images shipped again alongside YOLO label files. Combining the two subsets would therefore
    duplicate 746 images exactly -- doubling the weight of the three diseased classes and
    guaranteeing exact-duplicate leakage on top of the specimen leakage.

    It also scopes the project's "zero duplicates" claim precisely: zero WITHIN the 1,149-image
    classification corpus, 746 across the archive as a whole.
    """
    annotated = manifest[manifest["subset"] == "annotated"]
    assert set(annotated["sha256"]) <= set(raw_manifest["sha256"])
    assert len(annotated) == 746
    assert annotated["label_path"].notna().all(), "annotated rows must carry a YOLO label file"


def test_every_manifest_row_has_a_specimen_id(raw_manifest):
    assert raw_manifest["specimen_id"].notna().all()
    assert not raw_manifest["specimen_id"].astype(str).str.strip().eq("").any()


def test_specimen_id_is_derivable_from_the_filename(raw_manifest):
    """The manifest's specimen_id column must agree with the filename it came from. Guards
    against a manifest edited or regenerated by some other route."""
    stem = raw_manifest["path"].str.rsplit("/", n=1).str[-1]
    derived = stem.str.replace(r"-img-\d+\.\w+$", "", regex=True)
    assert (derived == raw_manifest["specimen_id"]).all()


# --------------------------------------------------------------------------------------
# Cross-check against the authors' own summary spreadsheet
# --------------------------------------------------------------------------------------

def _load_summary() -> pd.DataFrame:
    if not SUMMARY_XLSX.exists():
        pytest.skip("ShrimpDiseaseBD_Summary.xlsx not extracted")
    df = pd.read_excel(SUMMARY_XLSX)
    df["Class"] = df["Class"].replace(SUMMARY_CLASS_ALIASES)
    return df


def test_manifest_matches_official_summary_spreadsheet(raw_manifest):
    """Six independent quantities, all of which must agree. This is `D-011` as an executable
    check: the dataset ships a file enumerating its 416 specimens, and our manifest reproduces
    it exactly -- 0 files listed but missing, 0 files present but unlisted."""
    summary = _load_summary()

    assert len(summary) == PUBLISHED_TOTAL_SPECIMENS

    # Specimen counts per class.
    assert summary["Class"].value_counts().to_dict() == (
        raw_manifest.groupby("class")["specimen_id"].nunique().to_dict()
    )

    # Group sizes.
    assert summary["Image Count"].value_counts().to_dict() == (
        raw_manifest.groupby("specimen_id").size().value_counts().to_dict()
    )
    assert int(summary["Image Count"].sum()) == PUBLISHED_TOTAL_IMAGES

    # Filename-by-filename set equality, in both directions.
    listed = {name for row in summary["File_Name"] for name in str(row).split()}
    ours = set(raw_manifest["path"].str.rsplit("/", n=1).str[-1])
    assert not listed - ours, f"{len(listed - ours)} files listed by the authors are missing"
    assert not ours - listed, f"{len(ours - listed)} files on disk are unlisted by the authors"


def test_disease_present_flag_agrees_with_our_class_labels(raw_manifest):
    """`Disease_Present` is 274 Yes / 142 No, and 142 is exactly the Healthy specimen count.
    An independent confirmation that our Healthy label means what the authors meant by it."""
    summary = _load_summary()
    healthy_specimens = raw_manifest[raw_manifest["class"] == "Healthy"]["specimen_id"].nunique()

    assert (summary["Disease_Present"] == "No").sum() == healthy_specimens == 142
    assert (summary["Disease_Present"] == "Yes").sum() == PUBLISHED_TOTAL_SPECIMENS - 142
