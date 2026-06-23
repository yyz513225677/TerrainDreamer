"""Generate a more realistic rugged 100 m × 100 m lunar heightmap PNG.

Layers (combined into one heightmap):
  1. Fractal base terrain (multi-octave value noise) for natural ridges
     and valleys instead of artificial cones.
  2. Impact craters with proper bowl + raised rim profile.
  3. Eroded river bed following a random-walk centerline (not a sine).
  4. Scattered boulders (small high-frequency bumps).
  5. Flat clearance ±CENTER_CLEAR_M at the rover spawn so the rover can
     start every episode without immediately tipping over.

Output:
  - data/heightmaps/rugged_heightmap.png   (16-bit grayscale)
  - data/heightmaps/rugged_color.png       (turbo height-based RGB)

Run from the project root:
  python3 scripts/build_rugged_terrain.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ── Tunables ───────────────────────────────────────────────────────────────
SIZE_PX = 513
EXTENT_M = 100.0
CENTER_CLEAR_M = 10.0         # flat radius at origin (rover spawn)

# Fractal base
NUM_OCTAVES = 6
BASE_AMPLITUDE = 0.55
PERSISTENCE = 0.55
LACUNARITY = 2.0

# Impact craters
NUM_CRATERS = 4
CRATER_RADIUS_M = (3.0, 14.0)
CRATER_DEPTH_FRAC = (0.20, 0.35)
CRATER_RIM_FRAC = (0.08, 0.14)

# Eroded river
RIVER_LENGTH_SEG = 60
RIVER_STEP_PX = 12
RIVER_WIDTH_PX = 10
RIVER_DEPTH_FRAC = 0.20
RIVER_SOFT_BANK = 0.10

# Boulders
NUM_BOULDERS = 70
BOULDER_RADIUS_PX = (3, 8)
BOULDER_HEIGHT_FRAC = (0.15, 0.45)

SEED = 1331
BASE_LEVEL = 0.45             # mid-gray; fractal noise centers here

# Obstacle mask: anything steeper than this is considered non-traversable
# and goal destinations are rejected if they fall inside it.
OBSTACLE_SLOPE_DEG = 30.0
OBSTACLE_DILATE_PX = 3        # ~0.6 m buffer around steep regions
DEM_SIZE_Z = 4.0              # matches the SDF <size>100 100 4</size>


def fractal_terrain(shape, octaves, persistence, lacunarity, rng):
    """Multi-octave value noise built from random fields smoothed at
    progressively smaller scales. Returns a [0..1] array centered around 0.5.
    """
    H, W = shape
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    norm = 0.0
    base_sigma = min(H, W) / 4.0      # widest feature ≈ quarter of the map
    for o in range(octaves):
        sigma = max(0.8, base_sigma / (lacunarity ** o))
        # Generate random field at full resolution, then blur to the right scale.
        field = rng.standard_normal(shape).astype(np.float32)
        smooth = gaussian_filter(field, sigma=sigma, mode="reflect")
        # Standardise so each octave contributes a comparable range.
        smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
        out += amp * smooth
        norm += amp
        amp *= persistence
    out /= norm
    # Bring into [0, 1] with a smooth squash.
    out = 0.5 + 0.5 * np.tanh(0.9 * out)
    return out


def add_crater(arr, cx, cy, radius_px, depth, rim):
    """Add a circular impact crater: parabolic bowl + Gaussian rim."""
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    d = np.hypot(xs - cx, ys - cy)
    dn = d / radius_px
    bowl = -depth * np.clip(1.0 - dn * dn, 0.0, None)
    rim_profile = rim * np.exp(-((dn - 1.0) / 0.22) ** 2)
    arr += bowl + rim_profile


def random_walk_path(start_xy, length, step, rng, H, W, momentum=0.85):
    """Return a list of (x, y) points sampled along a smooth random walk
    starting at start_xy. The walk keeps direction across steps to mimic
    a river that bends gradually rather than zig-zagging."""
    x, y = start_xy
    theta = rng.uniform(0, 2 * np.pi)
    pts = [(x, y)]
    for _ in range(length):
        theta = momentum * theta + (1 - momentum) * rng.uniform(0, 2 * np.pi)
        x += step * np.cos(theta)
        y += step * np.sin(theta)
        if not (0 <= x < W and 0 <= y < H):
            theta += np.pi   # bounce inward
            x += 2 * step * np.cos(theta)
            y += 2 * step * np.sin(theta)
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        pts.append((x, y))
    return pts


def add_eroded_river(arr, pts, width_px, depth, soft_dip):
    """Carve a channel along the supplied path. For each pixel take the
    minimum distance to the polyline, then apply a sharp core dip plus a
    softer outer dip (mimicking eroded banks)."""
    H, W = arr.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    min_d = np.full_like(arr, 1e9, dtype=np.float32)
    # Sparse stamping; check distance to each control point.
    for px, py in pts:
        d = np.hypot(xs - px, ys - py).astype(np.float32)
        np.minimum(min_d, d, out=min_d)
    core = np.exp(-(min_d / width_px) ** 2)
    soft = np.exp(-(min_d / (width_px * 3.0)) ** 2)
    arr -= depth * core + soft_dip * soft


def main():
    rng = np.random.default_rng(SEED)
    W = H = SIZE_PX
    cx, cy = W / 2.0, H / 2.0
    px_per_m = W / EXTENT_M

    # ── 1. Fractal base ─────────────────────────────────────────────────
    base = fractal_terrain((H, W), NUM_OCTAVES, PERSISTENCE, LACUNARITY, rng)
    arr = BASE_LEVEL + BASE_AMPLITUDE * (base - 0.5)

    # ── 2. Impact craters ───────────────────────────────────────────────
    for _ in range(NUM_CRATERS):
        # keep craters away from the spawn area
        while True:
            rx = rng.uniform(15, W - 15)
            ry = rng.uniform(15, H - 15)
            if np.hypot(rx - cx, ry - cy) > CENTER_CLEAR_M * px_per_m + 8:
                break
        r_m = rng.uniform(*CRATER_RADIUS_M)
        r_px = r_m * px_per_m
        depth = rng.uniform(*CRATER_DEPTH_FRAC)
        rim = rng.uniform(*CRATER_RIM_FRAC)
        add_crater(arr, rx, ry, r_px, depth, rim)

    # ── 3. Eroded river bed (random-walk path) ──────────────────────────
    # Start at one edge, walk across the map.
    start_side = rng.integers(4)
    if start_side == 0:    start = (10, rng.uniform(20, H - 20))
    elif start_side == 1:  start = (W - 10, rng.uniform(20, H - 20))
    elif start_side == 2:  start = (rng.uniform(20, W - 20), 10)
    else:                  start = (rng.uniform(20, W - 20), H - 10)
    pts = random_walk_path(start, RIVER_LENGTH_SEG, RIVER_STEP_PX, rng, H, W)
    add_eroded_river(arr, pts, RIVER_WIDTH_PX, RIVER_DEPTH_FRAC, RIVER_SOFT_BANK)

    # ── 4. Scattered boulders ───────────────────────────────────────────
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    for _ in range(NUM_BOULDERS):
        bx = rng.uniform(5, W - 5)
        by = rng.uniform(5, H - 5)
        if np.hypot(bx - cx, by - cy) < CENTER_CLEAR_M * px_per_m + 5:
            continue
        r = rng.uniform(*BOULDER_RADIUS_PX)
        h = rng.uniform(*BOULDER_HEIGHT_FRAC)
        d = np.hypot(xs - bx, ys - by)
        arr += h * np.exp(-(d / r) ** 2)

    # ── 5. Center clearance ─────────────────────────────────────────────
    dist_from_center = np.hypot(xs - cx, ys - cy)
    clear_px = CENTER_CLEAR_M * px_per_m
    inside = dist_from_center < clear_px
    edge_ring = (dist_from_center >= clear_px) & (dist_from_center < clear_px + 8)
    target_level = BASE_LEVEL
    arr[inside] = target_level
    blend = (dist_from_center[edge_ring] - clear_px) / 8.0
    arr[edge_ring] = target_level * (1 - blend) + arr[edge_ring] * blend

    # ── Save ────────────────────────────────────────────────────────────
    arr = np.clip(arr, 0.0, 1.0)
    arr_u16 = (arr * 65535).astype(np.uint16)

    out_dir = (Path(__file__).resolve().parent.parent
               / "lunar_south_pole_gazebo" / "data" / "heightmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    grey_path = out_dir / "rugged_heightmap.png"
    color_path = out_dir / "rugged_color.png"

    Image.fromarray(arr_u16, mode="I;16").save(grey_path)

    # Color: turbo colormap keyed on height (same height → same color).
    import matplotlib.cm as cm
    cmap = cm.get_cmap("turbo")
    rgb_f = cmap(arr)[..., :3].astype(np.float32)
    rgb = (rgb_f * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(color_path)

    print(f"saved {grey_path}")
    print(f"saved {color_path}")
    print(f"  resolution:  {arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
    print(f"  layers:      fractal({NUM_OCTAVES} oct) + {NUM_CRATERS} craters "
          f"+ 1 eroded river ({RIVER_LENGTH_SEG} segs) + {NUM_BOULDERS} boulders")
    print(f"  clearance:   flat ±{CENTER_CLEAR_M} m at origin")

    # ── Slope stats + obstacle mask ─────────────────────────────────────
    # Convert pixel gradient to real slope (rise/run in meters).
    px_size_m = EXTENT_M / W
    gx = np.gradient(arr.astype(np.float32), axis=1) * DEM_SIZE_Z / px_size_m
    gy = np.gradient(arr.astype(np.float32), axis=0) * DEM_SIZE_Z / px_size_m
    slope = np.sqrt(gx * gx + gy * gy)
    slope_deg = np.degrees(np.arctan(slope))
    print(f"  slope:       median={np.median(slope_deg):.1f}°  "
          f"p90={np.percentile(slope_deg, 90):.1f}°  "
          f"p99={np.percentile(slope_deg, 99):.1f}°  "
          f"max={slope_deg.max():.1f}°")

    # Obstacle = anything steeper than OBSTACLE_SLOPE_DEG, dilated to give
    # the rover physical buffer. Also keep the spawn clearance traversable.
    from scipy.ndimage import binary_dilation
    obstacle = slope_deg > OBSTACLE_SLOPE_DEG
    obstacle = binary_dilation(obstacle, iterations=OBSTACLE_DILATE_PX)
    # Ensure the spawn area is always marked traversable.
    obstacle[inside] = False
    mask_path = out_dir / "rugged_obstacle_mask.png"
    Image.fromarray((obstacle * 255).astype(np.uint8), mode="L").save(mask_path)
    print(f"saved {mask_path}")
    print(f"  obstacle %:  {100.0 * obstacle.mean():.1f}% of map area")


if __name__ == "__main__":
    main()
