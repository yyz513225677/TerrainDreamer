#!/usr/bin/env bash
# Crop a tile out of the source DEM using gdal_translate.
#
# usage:
#   prepare_dem_tile.sh \
#     --input data/raw_dem/LDEM_80S_20MPP_ADJ.TIF \
#     --output data/processed_dem/shackleton_tile.tif \
#     --center-x  <PS_metres_x> \
#     --center-y  <PS_metres_y> \
#     --size-meters 2048 \
#     --samples 1025
#
# If --center-x and --center-y are not provided, the script prints the
# QGIS manual-crop workflow and exits 0 (informational), per the
# project's hard rule against inventing Shackleton coordinates.

set -euo pipefail

HERE="$(cd "$(dirname "$0")"/.. && pwd)"

INPUT="${HERE}/data/raw_dem/LDEM_80S_20MPP_ADJ.TIF"
OUTPUT="${HERE}/data/processed_dem/shackleton_tile.tif"
CENTER_X=""
CENTER_Y=""
SIZE_METERS="2048"
SAMPLES="1025"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)         INPUT="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --center-x)      CENTER_X="$2"; shift 2 ;;
    --center-y)      CENTER_Y="$2"; shift 2 ;;
    --size-meters)   SIZE_METERS="$2"; shift 2 ;;
    --samples)       SAMPLES="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *)
      echo "[prepare_dem_tile] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$INPUT" ]]; then
  echo "[prepare_dem_tile] DEM not found: $INPUT" >&2
  echo "  Place the NASA PGDA LOLA DEM at data/raw_dem/ first." >&2
  exit 2
fi

if ! command -v gdal_translate >/dev/null 2>&1; then
  echo "[prepare_dem_tile] gdal_translate not found — run scripts/install_dependencies.sh" >&2
  exit 3
fi

if [[ -z "$CENTER_X" || -z "$CENTER_Y" ]]; then
  cat <<EOF
[prepare_dem_tile] --center-x and --center-y were not provided.

The South Pole DEM is in polar stereographic METRES. This project does
not invent Shackleton coordinates; pick them yourself with QGIS.

See docs/qgis_workflow.md for the full procedure. Quick path:

  1. Open data/raw_dem/LDEM_80S_20MPP_ADJ.TIF in QGIS.
  2. Confirm CRS is polar stereographic metres (south).
  3. Hover over your region of interest and read the X / Y at the
     bottom of the window (those are projected metres, not lat/lon).
  4. Re-run this script with:
       --center-x  <X_in_metres> \\
       --center-y  <Y_in_metres>

EOF
  exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"

# Compute -projwin extent (ulx uly lrx lry) from centre + size.
HALF="$(python3 -c "print(${SIZE_METERS} / 2.0)")"
ULX="$(python3 -c "print(${CENTER_X} - ${HALF})")"
ULY="$(python3 -c "print(${CENTER_Y} + ${HALF})")"
LRX="$(python3 -c "print(${CENTER_X} + ${HALF})")"
LRY="$(python3 -c "print(${CENTER_Y} - ${HALF})")"

# Warn if the chosen --samples does not match the native LOLA pitch
# (≈20 m/px). gdal_translate happily resamples; we just want the user
# to be aware that the heightmap GSD reported in the metadata YAML
# reflects the OUTPUT pitch, not the source pitch.
NATIVE_GSD_M=20.0
EFFECTIVE_GSD_M="$(python3 -c "print(${SIZE_METERS} / ${SAMPLES})")"
if python3 -c "import sys; sys.exit(0 if abs(${EFFECTIVE_GSD_M}-${NATIVE_GSD_M})/${NATIVE_GSD_M} > 0.2 else 1)"; then
  echo "[prepare_dem_tile] NOTE: --samples=${SAMPLES} over --size-meters=${SIZE_METERS}"
  echo "                  yields ${EFFECTIVE_GSD_M} m/px output, vs ~${NATIVE_GSD_M} m/px native."
  echo "                  gdal_translate -outsize WILL resample; the metadata"
  echo "                  YAML's horizontal_resolution_m_per_pixel will be the"
  echo "                  OUTPUT pitch, not the source pitch."
fi

echo "[prepare_dem_tile] cropping with gdal_translate"
echo "  input    : $INPUT"
echo "  output   : $OUTPUT"
echo "  centre   : ($CENTER_X, $CENTER_Y)  size: ${SIZE_METERS} m"
echo "  -projwin : ulx=$ULX uly=$ULY lrx=$LRX lry=$LRY"
echo "  -outsize : $SAMPLES x $SAMPLES  (${EFFECTIVE_GSD_M} m/px output)"

gdal_translate \
  -projwin "$ULX" "$ULY" "$LRX" "$LRY" \
  -outsize "$SAMPLES" "$SAMPLES" \
  -of GTiff \
  "$INPUT" "$OUTPUT"

echo "[prepare_dem_tile] wrote $OUTPUT"
gdalinfo "$OUTPUT" | head -20
