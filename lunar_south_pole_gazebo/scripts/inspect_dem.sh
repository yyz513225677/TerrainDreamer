#!/usr/bin/env bash
# Thin wrapper around `gdalinfo`. Writes a -stats -hist dump of the
# DEM to data/metadata/dem_info.txt.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <path-to-dem.tif>" >&2
  exit 1
fi

INPUT="$1"
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
OUT="${HERE}/data/metadata/dem_info.txt"

if [[ ! -f "$INPUT" ]]; then
  echo "[inspect_dem] DEM not found: $INPUT" >&2
  echo "  Place the NASA PGDA LOLA DEM at data/raw_dem/ first." >&2
  echo "  See data/raw_dem/README.md for details." >&2
  exit 2
fi

if ! command -v gdalinfo >/dev/null 2>&1; then
  echo "[inspect_dem] gdalinfo not found — run scripts/install_dependencies.sh" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUT")"
echo "[inspect_dem] gdalinfo -stats -hist $INPUT > $OUT"
gdalinfo -stats -hist "$INPUT" > "$OUT"
echo "[inspect_dem] wrote $OUT ($(wc -l < "$OUT") lines)"
