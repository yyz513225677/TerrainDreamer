#!/usr/bin/env python3
"""Fill NoData (NaN) corners of a DEM tile so the entire rectangular
extent has elevation data, for use as a Gazebo heightmap.

Strategy (reuse-first):
  1. Read tile with rasterio (or osgeo.gdal fallback) — handles GeoTIFF I/O.
  2. Inpaint near the data boundary with GDAL's `gdal_fillnodata.py`
     (existing tool — invoked via subprocess, NOT reimplemented).
  3. For pixels still NaN after the bounded fill (far corners), set to a
     constant fill value (default: 25th-percentile of valid data, so
     the corners read as "lowlands").

Output is a regular GeoTIFF the rest of the pipeline already understands.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _read(path: Path):
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as ds:
            arr = ds.read(1, masked=False).astype(np.float32)
            nodata = ds.nodata
            transform = ds.transform
            crs = ds.crs
        return arr, nodata, transform, crs, "rasterio"
    except Exception:
        from osgeo import gdal  # type: ignore
        ds = gdal.Open(str(path))
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        return arr, nodata, None, None, "gdal"


def _write_like(src: Path, dst: Path, arr: np.ndarray):
    """Use rasterio if available to preserve georef; else gdal_translate
    via a temporary npy + gdal_calc-style write — but easier: copy src
    then overwrite band 1."""
    shutil.copyfile(src, dst)
    from osgeo import gdal  # type: ignore
    ds = gdal.Open(str(dst), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    band.WriteArray(arr.astype(np.float32))
    band.SetNoDataValue(float("nan"))
    ds.FlushCache()
    ds = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-distance-px", type=int, default=80,
                    help="gdal_fillnodata -md (pixels of inpaint reach)")
    ap.add_argument("--smoothing-iters", type=int, default=2,
                    help="gdal_fillnodata -si")
    ap.add_argument("--corner-fill-percentile", type=float, default=25.0,
                    help="percentile of valid data used as constant fill "
                         "for any pixels still NaN after inpainting")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[fill] input not found: {args.input}", file=sys.stderr)
        return 2

    # Stage 1: bounded inpaint via existing gdal_fillnodata tool.
    with tempfile.TemporaryDirectory() as td:
        stage1 = Path(td) / "stage1.tif"
        cmd = ["gdal_fillnodata.py",
               "-md", str(args.max_distance_px),
               "-si", str(args.smoothing_iters),
               "-of", "GTiff",
               str(args.input), str(stage1)]
        print("[fill] running:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("[fill] gdal_fillnodata failed:", r.stderr, file=sys.stderr)
            return r.returncode

        arr, nodata, _, _, _ = _read(stage1)

        # Stage 2: anything still NaN → constant fill (lowlands).
        nan_mask = ~np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            nan_mask |= (arr == nodata)
        finite = arr[~nan_mask]
        fill_val = float(np.percentile(finite, args.corner_fill_percentile))
        print(f"[fill] {int(nan_mask.sum())} pixels still NaN after inpaint; "
              f"filling with constant {fill_val:.2f} m "
              f"(P{args.corner_fill_percentile} of valid data)")
        arr[nan_mask] = fill_val

        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_like(args.input, args.output, arr)
        print(f"[fill] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
