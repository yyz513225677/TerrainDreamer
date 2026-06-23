#!/usr/bin/env python3
"""Bake an elevation-colour PNG matching the heightmap, à la PGDA
visualizations: blue (low) → cyan → green → yellow → orange → red (high).

Reuse-first: rasterio + NumPy + Pillow. No custom raster decode, no
custom PNG encoder. Only the colormap LUT itself is local code (5 LoC).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _read_dem(path: Path):
    """Returns (array_meters, nodata_mask)."""
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as ds:
            arr = ds.read(1, masked=False).astype(np.float64)
            nodata = ds.nodata
    except Exception:
        from osgeo import gdal  # type: ignore
        ds = gdal.Open(str(path))
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray().astype(np.float64)
        nodata = band.GetNoDataValue()
    mask = ~np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask |= (arr == nodata)
    return arr, mask


# 6-stop colormap that mimics the PGDA visualizations
# (deep blue → cyan → green → yellow → orange → dark red).
_STOPS = np.array([
    [0.04, 0.10, 0.38],   # deep blue
    [0.10, 0.45, 0.70],   # blue
    [0.18, 0.66, 0.55],   # cyan-green
    [0.55, 0.80, 0.30],   # yellow-green
    [0.95, 0.85, 0.20],   # yellow
    [0.92, 0.50, 0.18],   # orange
    [0.68, 0.22, 0.18],   # dark red
], dtype=np.float64)


def colormap_rgb(t: np.ndarray) -> np.ndarray:
    """t in [0,1] → uint8 RGB via piecewise-linear interpolation."""
    t = np.clip(t, 0.0, 1.0)
    n_segments = _STOPS.shape[0] - 1
    seg_t = t * n_segments
    idx = np.clip(seg_t.astype(np.int32), 0, n_segments - 1)
    frac = seg_t - idx
    out = (_STOPS[idx] * (1.0 - frac[..., None])
           + _STOPS[idx + 1] * frac[..., None])
    return (out * 255.0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="elevation GeoTIFF (the cropped tile)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output RGB PNG (same shape as the heightmap PNG)")
    ap.add_argument("--clip-percentile", default="2.0,98.0",
                    help="lo,hi percentiles to scale colormap to")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[colorize] input not found: {args.input}", file=sys.stderr)
        return 2

    arr, mask = _read_dem(args.input)
    try:
        lo_s, hi_s = args.clip_percentile.split(",")
        lo_pct, hi_pct = float(lo_s), float(hi_s)
    except ValueError:
        print("[colorize] --clip-percentile must be 'lo,hi'", file=sys.stderr)
        return 1

    finite = arr[~mask]
    if finite.size == 0:
        print("[colorize] all samples invalid", file=sys.stderr)
        return 1
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    t = (arr - lo) / (hi - lo)
    rgb = colormap_rgb(t)
    # Make any NoData pixels black so the heightmap edges stay clean.
    rgb[mask] = (8, 8, 12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output, format="PNG")
    print(f"[colorize] wrote {args.output} "
          f"({rgb.shape[1]}×{rgb.shape[0]}, percentile range "
          f"[{lo:.1f}, {hi:.1f}] m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
