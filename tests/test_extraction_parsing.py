"""Unit tests for the two parsing functions the whole data layer depends on.

`specimen_id` and `canonical_class` in `scripts/extract_images.py` are four lines each, which
is exactly why they deserve tests: everything downstream -- grouping, splitting, the leakage
measurement, the class balance table -- is built on their output, and a silent parse failure
would degrade specimen-level splitting into image-level splitting without raising anything.

That failure mode is the one this project exists to expose, so it gets an explicit guard.
"""

from __future__ import annotations

import pytest

from extract_images import canonical_class, specimen_id


# --------------------------------------------------------------------------------------
# specimen_id
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Healthy-1-img-1.jpg", "Healthy-1"),
        ("Healthy-1-img-3.jpg", "Healthy-1"),
        ("BG-10-img-2.jpg", "BG-10"),
        ("WSSV-118-img-1.jpg", "WSSV-118"),
        ("BG_WSSV-81-img-3.jpg", "BG_WSSV-81"),
        ("WSSV_BG-81-img-3.jpg", "WSSV_BG-81"),
    ],
)
def test_specimen_id_parses_known_filenames(filename, expected):
    assert specimen_id(filename) == expected


def test_all_three_photos_of_one_shrimp_share_an_id():
    """The property that makes grouping possible at all."""
    ids = {specimen_id(f"Healthy-7-img-{n}.jpg") for n in (1, 2, 3)}
    assert ids == {"Healthy-7"}


def test_different_specimens_of_the_same_class_do_not_collide():
    assert specimen_id("BG-1-img-1.jpg") != specimen_id("BG-11-img-1.jpg")


@pytest.mark.parametrize(
    "filename",
    ["notes.txt", "IMG_2043.jpg", "Healthy.jpg", "Healthy-img-1.jpg", ""],
)
def test_specimen_id_returns_none_for_unparseable_names(filename):
    """It must return None rather than guess. A guessed id would put unrelated images in the
    same group; None is caught by `load_raw_manifest`, which refuses to build a split from a
    manifest with missing ids."""
    assert specimen_id(filename) is None


def test_extension_is_stripped_whatever_it_is():
    assert specimen_id("BG-3-img-1.png") == "BG-3"
    assert specimen_id("BG-3-img-1.JPEG") == "BG-3"


# --------------------------------------------------------------------------------------
# canonical_class
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "folder,expected",
    [
        ("1. Healthy", "Healthy"),
        ("2. BG", "BG"),
        ("3. WSSV", "WSSV"),
        ("4. WSSV_BG", "BG_WSSV"),
        ("4. BG_WSSV", "BG_WSSV"),
        ("Healthy", "Healthy"),
    ],
)
def test_canonical_class_normalises_folder_names(folder, expected):
    assert canonical_class(folder) == expected


def test_the_two_coinfection_spellings_collapse_to_one_class():
    """The Mendeley tree names it WSSV_BG, other packagings name it BG_WSSV. Without this
    normalisation the same class would appear twice and every per-class metric would be wrong.
    """
    assert canonical_class("4. WSSV_BG") == canonical_class("4. BG_WSSV")


def test_ordinal_prefix_is_not_required():
    assert canonical_class("BG") == "BG"
