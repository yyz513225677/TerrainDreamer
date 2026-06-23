# QGIS Manual Crop Workflow

This is the **fallback** path when you do not yet know the projected
metre coordinates of your region of interest (e.g. Shackleton crater
rim) and need to pick them interactively. If you already know the
coordinates, use the automated path:

```bash
bash scripts/prepare_dem_tile.sh \
  --input  data/raw_dem/LDEM_80S_20MPP_ADJ.TIF \
  --output data/processed_dem/shackleton_tile.tif \
  --center-x  <PS_metres_x> \
  --center-y  <PS_metres_y> \
  --size-meters 2048 \
  --samples 1025
```

## 1. Open the DEM in QGIS

1. Install QGIS 3.x (`sudo apt install qgis`).
2. `Layer → Add Layer → Add Raster Layer…` →
   `data/raw_dem/LDEM_80S_20MPP_ADJ.TIF`.
3. Confirm the CRS reported by QGIS is **polar stereographic
   metres** (south). PGDA documents the exact PROJ string for the
   product; do not change it.

## 2. Locate the region of interest

* Use the *Identify Features* tool to read elevation at the cursor.
* Use the coordinate readout at the bottom of the QGIS window — these
  are *projected metres*, not lat/lon, because the DEM CRS is
  stereographic.
* For Shackleton crater, the centre and rim coordinates are public
  knowledge but **change** with the chosen ellipsoid / pole
  convention. Read them from the DEM itself or from PGDA's published
  notes — do **not** copy numbers from this document.

## 3. Crop to a square tile

In QGIS:

1. `Raster → Extraction → Clip Raster by Extent…`
2. Input layer: the LOLA DEM.
3. Clipping extent: `Use Map Canvas Extent` (after panning/zooming) or
   manually enter `xmin, xmax, ymin, ymax` in **projected metres**.
4. Output: `data/processed_dem/shackleton_tile.tif`.
5. Optional: tick *Use Input Data Type* to preserve `Float32`.

If you prefer the CLI after picking the bounds:

```bash
gdal_translate \
  -projwin <xmin> <ymax> <xmax> <ymin> \
  -outsize 1025 1025 \
  -of GTiff \
  data/raw_dem/LDEM_80S_20MPP_ADJ.TIF \
  data/processed_dem/shackleton_tile.tif
```

(Note GDAL's `-projwin` order: `ulx uly lrx lry`.)

## 4. Verify the tile

```bash
gdalinfo -stats data/processed_dem/shackleton_tile.tif | head -40
```

Check:
* `Size is 1025, 1025` (or whatever `--samples` you chose).
* `Pixel Size = (20, -20)` (or your `--size-meters / --samples` ratio).
* `STATISTICS_MINIMUM` / `STATISTICS_MAXIMUM` are physical metres.

## 5. Next step

```bash
python3 scripts/normalize_heightmap.py \
  --input  data/processed_dem/shackleton_tile.tif \
  --output data/heightmaps/shackleton_heightmap.png \
  --meta   data/metadata/shackleton_tile.yaml
```

## Hard rules

* **Never** invent Shackleton coordinates. If you don't know them,
  spend a minute in QGIS; do not paste numbers from a chatbot.
* **Never** reproject the DEM into lat/lon before this stage — Gazebo
  needs equal-area planar metres for the heightmap `<size>` tag.
