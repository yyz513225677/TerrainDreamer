#!/usr/bin/env python3
"""Convert a Float32 GeoTIFF DEM tile to a 16-bit grey PNG heightmap.

Reuse-first: rasterio (preferred) → osgeo.gdal (fallback). NumPy for
the arithmetic. Pillow for PNG encoding. Nothing here implements
raster decoding or PNG encoding.

Output: 16-bit grey PNG suitable for Gazebo's <heightmap><uri>...</uri></heightmap>
geometry, plus a metadata YAML the SDF generator consumes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def _read_dem_rasterio(path: Path):
    import rasterio  # type: ignore
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=False).astype(np.float64)
        # rasterio's .res returns (xres, yres) as positive magnitudes.
        res_x_m, res_y_m = ds.res
        nodata = ds.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, float(res_x_m), float(res_y_m)


def _read_dem_gdal(path: Path):
    from osgeo import gdal  # type: ignore
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"gdal.Open failed for {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    geo = ds.GetGeoTransform()
    res_x_m = abs(geo[1])
    res_y_m = abs(geo[5])
    nodata = band.GetNoDataValue()
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, res_x_m, res_y_m


def read_dem(path: Path):
    """Try rasterio first, fall back to osgeo.gdal on any rasterio
    failure (ImportError or runtime read error)."""
    try:
        return _read_dem_rasterio(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[normalize] rasterio path failed ({type(exc).__name__}: "
              f"{exc}); falling back to osgeo.gdal", file=sys.stderr)
        return _read_dem_gdal(path)


def normalize_to_u16(arr: np.ndarray, lo_pct: float, hi_pct: float
                    ) -> tuple[np.ndarray, float, float]:
    """Map elevations to 0..65535 uint16 via percentile clipping."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("All elevation samples are NaN — bad DEM?")
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        raise ValueError(f"min == max after clipping ({lo}); cannot "
                         f"normalise. Try --clip-percentile 0,100.")
    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    u16 = np.round(norm * 65535.0).astype(np.uint16)
    return u16, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="Float32 GeoTIFF DEM tile")
    ap.add_argument("--output", required=True, type=Path,
                    help="output 16-bit PNG path")
    ap.add_argument("--meta", required=True, type=Path,
                    help="output metadata YAML path")
    ap.add_argument("--clip-percentile", default="0.0,100.0",
                    help="lo,hi percentiles for elevation clipping")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[normalize] input not found: {args.input}",
              file=sys.stderr)
        return 2

    try:
        lo_pct_s, hi_pct_s = args.clip_percentile.split(",")
        lo_pct, hi_pct = float(lo_pct_s), float(hi_pct_s)
    except ValueError:
        print("[normalize] --clip-percentile must be 'lo,hi'",
              file=sys.stderr)
        return 1

    print(f"[normalize] reading {args.input}")
    arr, res_x_m, res_y_m = read_dem(args.input)
    print(f"[normalize] shape={arr.shape} dtype={arr.dtype} "
          f"res_x_m={res_x_m} res_y_m={res_y_m}")

    u16, min_elev, max_elev = normalize_to_u16(arr, lo_pct, hi_pct)
    vertical_scale_m = max_elev - min_elev
    tile_width_m = res_x_m * arr.shape[1]
    tile_height_m = res_y_m * arr.shape[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u16, mode="I;16").save(args.output, format="PNG")
    print(f"[normalize] wrote {args.output} (16-bit, "
          f"{u16.shape[1]}×{u16.shape[0]})")

    meta = {
        "source_dem": str(args.input),
        "heightmap_png": str(args.output),
        "tile_width_m": float(tile_width_m),
        "tile_height_m": float(tile_height_m),
        "samples_x": int(arr.shape[1]),
        "samples_y": int(arr.shape[0]),
        "min_elevation_m": float(min_elev),
        "max_elevation_m": float(max_elev),
        "vertical_scale_m": float(vertical_scale_m),
        "horizontal_resolution_m_per_pixel": float(res_x_m),
        "generated_time": _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"[normalize] wrote {args.meta}")
    print(f"[normalize] elevation: [{min_elev:.3f}, {max_elev:.3f}] m "
          f"→ vertical_scale_m={vertical_scale_m:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
