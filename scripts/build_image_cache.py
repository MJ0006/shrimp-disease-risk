"""Pre-resize the image corpus to a training-sized cache.

The problem
-----------
Every source image is 2048 x 2048 (4.19 megapixels, ~263 KB JPEG, 303 MB total). Training feeds
the network 224 x 224 tensors, so each epoch decodes 4.19 MP and immediately throws away 99.8%
of the pixels. Repeated across 30 epochs and 9 experimental runs, JPEG decoding -- not the GPU --
becomes the bottleneck, and an 8 GB Blackwell GPU sits idle waiting on CPU workers.

The fix
-------
Decode once, resize to 256 x 256, store as JPEG quality 95. Training then reads ~15 KB files.

Why 256 and not 224
-------------------
Caching straight to the network's input size would fix the geometry permanently and rule out
any augmentation that crops or rotates -- a rotation of a 224 image already at final size must
either pad or lose corners. A 256 cache leaves ~14% linear headroom for RandomResizedCrop and
rotation, at negligible storage cost.

Why this does not compromise the results
----------------------------------------
Downsampling 2048 -> 256 -> 224 versus 2048 -> 224 differs only in resampling detail, far below
the scale of the disease features involved (white spots and gill discolouration occupy large
image regions, not single pixels). The cache is regenerable from `data/raw/`, is written to
`data/processed/`, and is never treated as source data. Both are recorded so the shortcut is
visible rather than hidden.

Usage
-----
    python scripts/build_image_cache.py            # build cache + cached manifest
    python scripts/build_image_cache.py --size 320 # different cache resolution
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    DATA_PROCESSED,
    IMAGE_MANIFEST,
    PROJECT_ROOT,
)


def _resize_one(job: tuple[str, str, int]) -> tuple[str, int, bool]:
    """Resize a single image. Runs in a worker process, so it takes plain strings."""
    src, dst, size = job
    dst_path = Path(dst)
    if dst_path.exists():
        return dst, dst_path.stat().st_size, True
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        # LANCZOS for downscaling: it is the highest-quality resampling filter Pillow offers
        # and this runs once, so its extra cost is irrelevant.
        im = im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        im.save(dst_path, "JPEG", quality=95, optimize=True)
    return dst, dst_path.stat().st_size, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=256, help="cache edge length (default 256)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = pd.read_csv(IMAGE_MANIFEST)
    cache_root = DATA_PROCESSED / f"images_{args.size}"

    jobs, cached_paths = [], []
    for _, row in manifest.iterrows():
        src = PROJECT_ROOT / row["path"]
        # Flatten to <subset>/<class>/<filename>: the source tree is deeply nested and its
        # folder names vary between packagings, which is noise we do not need to preserve.
        dst = cache_root / row["subset"] / row["class"] / Path(row["path"]).name
        jobs.append((str(src), str(dst), args.size))
        cached_paths.append(str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    print(f"source   : {len(jobs):,} images @ 2048x2048")
    print(f"cache    : {cache_root.relative_to(PROJECT_ROOT)} @ {args.size}x{args.size}")
    print(f"workers  : {args.workers}\n")

    t0 = time.perf_counter()
    total_bytes, skipped = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (_, nbytes, was_cached) in enumerate(ex.map(_resize_one, jobs, chunksize=16), 1):
            total_bytes += nbytes
            skipped += was_cached
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,}  ({time.perf_counter()-t0:.1f}s)")

    cached = manifest.copy()
    cached["source_path"] = cached["path"]
    cached["path"] = cached_paths
    out = IMAGE_MANIFEST.parent / f"image_manifest_cached_{args.size}.csv"
    cached.to_csv(out, index=False)

    src_mb = manifest["bytes"].sum() / 1e6
    print(f"\nelapsed        : {time.perf_counter()-t0:.1f}s ({skipped:,} already cached)")
    print(f"source size    : {src_mb:,.0f} MB")
    print(f"cache size     : {total_bytes/1e6:,.0f} MB  ({src_mb/max(total_bytes/1e6,1e-9):.1f}x smaller)")
    print(f"cached manifest: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
