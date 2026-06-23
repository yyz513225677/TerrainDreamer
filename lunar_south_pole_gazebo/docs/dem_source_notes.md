# DEM Source Notes

## Product

* **Name (as used here):** `LDEM_80S_20MPP_ADJ.TIF`
* **Provider:** NASA Planetary Geodynamics Data Archive (PGDA) — see
  https://pgda.gsfc.nasa.gov/ (top-level domain; navigate to the
  Lunar / LOLA section yourself; do not trust deep-link URLs from
  chatbots, they go stale).
* **Region:** Lunar South Pole, 80°S – 90°S (the "ADJ" variant is the
  block-adjusted LOLA DEM).
* **Native resolution:** 20 metres per pixel.
* **Elevation units:** metres relative to the LOLA reference radius.
* **Coordinate system:** polar stereographic, X/Y in metres, true
  scale at the pole.

## Why this product

* High enough density to capture rover-scale slope (20 m/px) without
  being so big the world becomes unloadable.
* Single-tile global coverage of the South Pole region of interest
  for in-situ-resource-utilisation (ISRU) and lighting studies.
* The "adjusted" (`_ADJ`) variant has had LOLA orbit ties applied —
  better global geometry than the un-adjusted release.

## What this project does **not** do

* It does **not** download the DEM. The user places it manually at
  `data/raw_dem/LDEM_80S_20MPP_ADJ.TIF`. We do not bake a fragile
  download URL into a script.
* It does **not** assume one specific tile of Shackleton — the tile
  centre is a runtime parameter to `scripts/prepare_dem_tile.sh`.
* It does **not** apply a custom geoid model — the world frame is
  the DEM's native polar stereographic metres.

## License / attribution

NASA / GSFC LOLA-derived products are public-domain in the US. Cite:
> Smith, D. E., et al. *The Lunar Reconnaissance Orbiter Lunar Orbiter
> Laser Altimeter Investigation.* Space Science Reviews 150 (2010).

Place the citation in any paper or media that uses Gazebo renders
backed by this DEM.

## Variants you might also see

* `LDEM_80S_20MPP.TIF` — un-adjusted, otherwise identical.
* Higher-resolution NAC-DTM tiles for very specific landing sites
  (e.g. SLDEM2015) — these are *not* what this project consumes; the
  format differs and the file sizes are enormous.

If you want to use a different LOLA product, update
`config/dem_config.yaml` (`raw_dem_path` and `vertical_unit_m`); the
rest of the pipeline reads from there.
