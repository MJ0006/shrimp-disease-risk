"""Extract and verify the ShrimpDiseaseImageBD image dataset.

Why a script rather than right-clicking "Extract All"
-----------------------------------------------------
The archive arrived in four variants (a Mendeley original, two same-size Kaggle copies, and a
third-party augmented derivative). Which bytes actually trained the model is an examinable
question, so extraction is scripted: one declared authoritative source, recorded counts, and a
manifest that every downstream stage reads instead of walking the directory itself.

Authoritative source
--------------------
Islam, M. M., Sarker, A., Choudhury, A., Ahmed, N., & Rasel, A. A. S. (2025).
"ShrimpDiseaseImageBD: An Image Dataset for Computer Vision-Based Detection of Shrimp
Diseases in Bangladesh." Mendeley Data, V3. doi:10.17632/jhrtdj9txm.3   Licence: CC BY 4.0

Published composition: 1,149 raw images -- Healthy 403, BG 198, WSSV 328, BG_WSSV 220.
The annotated subset covers the three diseased classes only (198+328+220 = 746).

Usage
-----
    python scripts/extract_images.py            # extract + manifest
    python scripts/extract_images.py --verify   # re-verify an existing extraction
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
DEST = PROJECT_ROOT / "data" / "interim" / "images"
MANIFEST = PROJECT_ROOT / "data" / "interim" / "image_manifest.csv"

SOURCE_ZIP = RAW / (
    "ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based "
    "Detection of Shrimp Diseases in Bangladesh.zip"
)

# Published per-class counts. Treated as an expectation to CHECK, not a value to assume:
# a mismatch means the archive is not what the paper describes and must be investigated
# before any model is trained on it.
EXPECTED_RAW = {"Healthy": 403, "BG": 198, "WSSV": 328, "BG_WSSV": 220}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def specimen_id(filename: str) -> str | None:
    """Recover the specimen a photo belongs to from its filename.

    Files are named "<Class>-<specimen>-img-<n>.jpg", e.g. Healthy-1-img-1/2/3. That means the
    1,149 images are NOT 1,149 independent observations -- they are repeat photographs of a much
    smaller set of individual shrimp.

    This matters more than anything else in the pipeline. A random image-level split scatters
    photos of the SAME shrimp across train and test, so a model can score well by recognising an
    individual it already memorised rather than by learning the disease. Every split in this
    project therefore groups on this id.
    """
    m = re.match(r"^(?P<cls>.+?)-(?P<spec>\d+)-img-\d+\.", filename, re.I)
    return f"{m.group('cls')}-{m.group('spec')}" if m else None


def canonical_class(folder: str) -> str:
    """Map an on-disk folder name to a canonical class label.

    Folders are named like "1. BG" or "4. WSSV_BG". We strip any leading "<n>. " ordinal and
    normalise the co-infection class, which appears as both BG_WSSV and WSSV_BG across the
    Mendeley and Kaggle packagings of the same data. Without this normalisation the two
    spellings would silently become two separate classes.
    """
    name = folder.split(". ", 1)[-1].strip()
    if name.upper() in {"WSSV_BG", "BG_WSSV"}:
        return "BG_WSSV"
    return name


def unzip(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dest)


def flatten_nested_zips(root: Path) -> None:
    """The Mendeley package nests 'Raw Images.zip' and 'Annotated...zip' inside the outer
    archive, so a single extract leaves zips on disk rather than images."""
    for z in sorted(root.rglob("*.zip")):
        print(f"  unpacking nested {z.name}")
        unzip(z, z.parent)
        z.unlink()


def find_split_roots(root: Path) -> dict[str, Path]:
    """Locate the 'Raw Images' and 'Annotated' trees wherever the packaging put them.

    Path depth differs between the Mendeley and Kaggle layouts (the Kaggle copy repeats the
    folder name a level deeper), so we search rather than hard-code a path.
    """
    found: dict[str, Path] = {}
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        low = d.name.lower()
        if low == "raw images" and "raw" not in found:
            found["raw"] = d
        elif low.startswith("annotated") and "annotated" not in found:
            found["annotated"] = d
    return found


def build_manifest(raw_root: Path, annotated_root: Path | None) -> list[dict]:
    """Walk the class folders and record one row per image.

    Records a content hash for every file. This is what makes duplicate detection possible in
    Stage 3b -- exact duplicates split across train and test are the leading candidate
    explanation for the inherited notebook's val 0.877 / test 0.638 gap, and we cannot test
    that hypothesis without hashes.
    """
    rows: list[dict] = []

    for split_name, root in (("raw", raw_root), ("annotated", annotated_root)):
        if root is None:
            continue
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cls = canonical_class(cls_dir.name)
            for img in sorted(cls_dir.rglob("*")):
                if img.suffix.lower() not in IMAGE_EXT or not img.is_file():
                    continue
                data = img.read_bytes()
                label = img.with_suffix(".txt")
                # Detection labels live in a sibling 'labels' dir in the annotated tree.
                if not label.exists():
                    alt = img.parent.parent / "labels" / (img.stem + ".txt")
                    label = alt if alt.exists() else None
                rows.append(
                    {
                        "path": str(img.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "subset": split_name,
                        "class": cls,
                        "specimen_id": specimen_id(img.name) or "",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "label_path": (
                            str(label.relative_to(PROJECT_ROOT)).replace("\\", "/")
                            if label else ""
                        ),
                    }
                )
    return rows


def report(rows: list[dict]) -> bool:
    raw = [r for r in rows if r["subset"] == "raw"]
    ann = [r for r in rows if r["subset"] == "annotated"]

    print(f"\nraw images      : {len(raw):,}")
    print(f"annotated images: {len(ann):,}")

    # Specimen-level structure: the single most consequential property of this dataset.
    specs = {r["specimen_id"] for r in raw if r["specimen_id"]}
    unparsed = sum(1 for r in raw if not r["specimen_id"])
    per_spec = Counter(r["specimen_id"] for r in raw if r["specimen_id"])
    if specs:
        n = len(specs)
        print(f"\nunique specimens: {n:,} across {len(raw):,} images "
              f"({len(raw)/n:.2f} images per specimen)")
        if unparsed:
            print(f"  WARNING: {unparsed} filenames did not parse to a specimen id")
        print(f"  group sizes: {dict(sorted(Counter(per_spec.values()).items()))}")
        print("  -> image-level random splitting WILL leak specimens across train/test")

    counts = Counter(r["class"] for r in raw)
    print("\nper-class (raw):")
    ok = True
    for cls, expected in EXPECTED_RAW.items():
        got = counts.get(cls, 0)
        flag = "OK " if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {cls:10s} {got:5d}  expected {expected:5d}  {flag}")

    extra = set(counts) - set(EXPECTED_RAW)
    if extra:
        ok = False
        print(f"  UNEXPECTED CLASSES: {sorted(extra)}")

    # Duplicate detection -- the reason we hashed everything.
    hashes = Counter(r["sha256"] for r in raw)
    dupes = {h: c for h, c in hashes.items() if c > 1}
    n_dupe_files = sum(dupes.values()) - len(dupes)
    print(f"\nexact duplicate images (raw): {n_dupe_files} extra copies "
          f"across {len(dupes)} hash groups")
    if dupes:
        print("  -> group-aware splitting required; see Stage 3b")

    labelled = sum(1 for r in ann if r["label_path"])
    print(f"annotated images with a label file: {labelled:,}/{len(ann):,}")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="skip extraction, re-verify only")
    args = ap.parse_args()

    if not args.verify:
        if not SOURCE_ZIP.exists():
            print(f"authoritative archive not found:\n  {SOURCE_ZIP}", file=sys.stderr)
            return 1
        if DEST.exists():
            shutil.rmtree(DEST)
        print(f"extracting {SOURCE_ZIP.name} ({SOURCE_ZIP.stat().st_size/1e6:.0f} MB) ...")
        unzip(SOURCE_ZIP, DEST)
        flatten_nested_zips(DEST)

    roots = find_split_roots(DEST)
    if "raw" not in roots:
        print("could not locate a 'Raw Images' directory in the extraction", file=sys.stderr)
        return 1
    print(f"\nraw root      : {roots['raw'].relative_to(PROJECT_ROOT)}")
    if "annotated" in roots:
        print(f"annotated root: {roots['annotated'].relative_to(PROJECT_ROOT)}")

    rows = build_manifest(roots["raw"], roots.get("annotated"))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = report(rows)
    print(f"\nmanifest -> {MANIFEST.relative_to(PROJECT_ROOT)}  ({len(rows):,} rows)")
    if not ok:
        print("\nWARNING: composition does not match the published description.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
