"""Procedurally build an Apollo-15-Hadley-inspired lunar surface ('varied' env).

Compressed analog of the Apollo 15 landing region (Hadley-Apennine), 100×100 m:
  * **Mare base**       — flat regolith plain (Mare Imbrium analog) with
                            mm-scale regolith roughness
  * **Crater field**    — power-law size distribution (Apollo/LRO statistics),
                            proper bowl profile with raised rim (fresh) or
                            softened (eroded), depth ≈ D/5 fresh, D/8 eroded
  * **Ejecta blankets** — boulders concentrated around fresh craters (the
                            ejecta-radius cones seen at Apollo sites)
  * **Hadley Rille**    — sinuous lava-channel meander across the S half,
                            split into 3 disjoint segments so two natural
                            land bridges remain (always-drivable)
  * **Lobate scarp**    — Apennine-front-style thrust cliff in the NW
  * **Highlands patch** — Mons Hadley analog in the NE: raised base + dense
                            saturation cratering
  * **Wrinkle ridge**   — compression feature in the SW
  * **Spawn zone**      — flat for r ≤ 9 m, smooth feather to r = 15 m

Outputs:
    ros_ws/.../worlds/heightmaps/varied.png         513×513 grayscale
    ros_ws/.../worlds/heightmaps/varied_traversable.npy
                                                     boolean drivability mask
    ros_ws/.../worlds/varied.world                  Gazebo Sim SDF

Usage:
    python3 scripts/build_varied_world.py
    ./run_human.sh --env varied
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image


# ── world params ────────────────────────────────────────────────────────────
RES = 513
WORLD_M = 100.0
VERT_M = 10.0
PX = WORLD_M / (RES - 1)

SPAWN_FLAT_R = 9.0          # m, radius of guaranteed-flat zone around origin
SPAWN_FEATHER_M = 6.0       # m, smooth taper from flat → full terrain
SPAWN_NO_CRATER_R = 14.0    # m, no crater centers placed within this radius
                            # (their bowls would otherwise leak into the spawn
                            # ring and create the steep slopes the user saw).


def world_to_pix(x: float, y: float) -> tuple[int, int]:
    """World (x, y) m → image (col, row). Image origin is top-left, +y is up."""
    col = int(round((x + WORLD_M / 2) / PX))
    row = int(round((WORLD_M / 2 - y) / PX))
    return max(0, min(RES - 1, col)), max(0, min(RES - 1, row))


def add_smooth_noise(h: np.ndarray, scale_px: int, amp: float,
                      rng: np.random.Generator):
    low_res = max(2, RES // scale_px)
    coarse = rng.standard_normal((low_res, low_res)).astype(np.float32)
    img = Image.fromarray(coarse).resize((RES, RES), Image.BILINEAR)
    h += np.asarray(img, dtype=np.float32) * amp


# ── crater modeling ─────────────────────────────────────────────────────────

def add_crater(
    h: np.ndarray,
    cx_world: float, cy_world: float,
    R_m: float,                  # crater radius (rim crest), m
    depth_m: float,              # bowl depth from rim crest, m
    rim_h_m: float,              # rim height above the surrounding terrain, m
    freshness: float = 1.0,      # 1.0 sharp fresh; 0.3 eroded
):
    """Bowl-with-raised-rim profile. Lunar fresh craters: depth ≈ D/5,
    rim ≈ depth/8. Shape: parabolic bowl inside R, Gaussian-annulus rim
    centered at r ≈ 1.05 R, soft fade by 1.6 R."""
    cc, rr = world_to_pix(cx_world, cy_world)
    yy, xx = np.indices((RES, RES), dtype=np.float32)
    d_px = np.hypot(xx - cc, yy - rr)
    R_px = R_m / PX

    rho = d_px / R_px
    bowl = -depth_m * np.clip(1.0 - rho * rho, 0.0, 1.0)

    rim_center_rho = 1.05
    rim_sigma_rho  = 0.20
    rim_g = np.exp(-0.5 * ((rho - rim_center_rho) / rim_sigma_rho) ** 2)
    outer = np.clip(1.0 - (rho - 1.0) / 0.6, 0.0, 1.0)   # fade past 1.6 R
    rim = rim_h_m * rim_g * outer

    inside = (d_px <= R_px).astype(np.float32)
    delta = inside * bowl + (1.0 - inside) * rim
    far = (rho > 1.7).astype(np.float32)
    delta = delta * (1.0 - far)

    h += delta * freshness


def crater_field(h: np.ndarray, rng: np.random.Generator,
                  region_xy_radius: float, center_xy=(0.0, 0.0),
                  small_n=80, med_n=20, large_n=4,
                  exclude_r=SPAWN_FLAT_R + 1.5,
                  freshness_mix=(0.55, 0.55)):
    """Drop a power-law-ish set of craters in a disk of given radius.

    `freshness_mix` = (mean_fresh_prob, weight) — fraction of craters that get
    freshness near 1.0 (sharp); the rest are softened (eroded)."""
    cx0, cy0 = center_xy

    def _sample_pos():
        for _ in range(80):
            theta = rng.uniform(-math.pi, math.pi)
            r = math.sqrt(rng.uniform(0, region_xy_radius ** 2))
            x = cx0 + r * math.cos(theta)
            y = cy0 + r * math.sin(theta)
            if (x * x + y * y) >= exclude_r * exclude_r:
                return x, y
        return None

    fresh_p, _ = freshness_mix
    fresh_craters = []   # (x, y, R) of fresh ones — used to seed boulder ejecta

    for n_class, R_lo, R_hi, depth_factor in [
        (small_n, 1.4, 3.0, 0.20),
        (med_n,   3.0, 8.0, 0.18),
        (large_n, 10.0, 20.0, 0.15),
    ]:
        for _ in range(n_class):
            pos = _sample_pos()
            if pos is None:
                continue
            x, y = pos
            R = float(rng.uniform(R_lo, R_hi))
            depth = R * depth_factor
            rim_h = depth * 0.18
            fresh = float(rng.uniform(0.85, 1.0)) if rng.uniform() < fresh_p \
                    else float(rng.uniform(0.25, 0.55))
            add_crater(h, x, y, R_m=R, depth_m=depth,
                        rim_h_m=rim_h, freshness=fresh)
            if fresh > 0.7:
                fresh_craters.append((x, y, R))

    return fresh_craters


# ── linear features ─────────────────────────────────────────────────────────

def carve_channel(h: np.ndarray, points_xy: list[tuple[float, float]],
                   half_width_m: float, depth_m: float,
                   wall_steepness: float = 1.0):
    """Carve a soft trench along a polyline (rilles, riverbeds).

    wall_steepness > 1.0 → steeper walls (more like a rille); 1.0 = smoothstep."""
    half_w_px = half_width_m / PX
    yy, xx = np.indices((RES, RES), dtype=np.float32)
    field = np.full_like(h, np.inf)
    for (ax, ay), (bx, by) in zip(points_xy[:-1], points_xy[1:]):
        ac, ar = world_to_pix(ax, ay)
        bc, br = world_to_pix(bx, by)
        vx, vy = bc - ac, br - ar
        L2 = vx * vx + vy * vy
        if L2 < 1e-3:
            continue
        t = ((xx - ac) * vx + (yy - ar) * vy) / L2
        t = np.clip(t, 0.0, 1.0)
        px = ac + t * vx
        py = ar + t * vy
        d = np.hypot(xx - px, yy - py)
        np.minimum(field, d, out=field)
    s = np.clip(1.0 - field / half_w_px, 0.0, 1.0)
    profile = s ** wall_steepness * (3.0 - 2.0 * s) if wall_steepness == 1.0 \
              else np.clip(s, 0.0, 1.0) ** wall_steepness
    h += depth_m * profile


def add_lobate_scarp(h: np.ndarray, points_xy: list[tuple[float, float]],
                      height_m: float, ramp_m: float = 1.5):
    """Curved cliff face: terrain on the LEFT side of the polyline (looking
    along travel direction) is raised by `height_m`, the right unchanged.
    `ramp_m` = horizontal distance over which the step ramps."""
    yy, xx = np.indices((RES, RES), dtype=np.float32)
    signed = np.full_like(h, np.inf)
    sign_field = np.zeros_like(h)
    for (ax, ay), (bx, by) in zip(points_xy[:-1], points_xy[1:]):
        ac, ar = world_to_pix(ax, ay)
        bc, br = world_to_pix(bx, by)
        vx, vy = bc - ac, br - ar
        L2 = vx * vx + vy * vy
        if L2 < 1e-3:
            continue
        t = ((xx - ac) * vx + (yy - ar) * vy) / L2
        t_cl = np.clip(t, 0.0, 1.0)
        px = ac + t_cl * vx
        py = ar + t_cl * vy
        d = np.hypot(xx - px, yy - py)
        cross = (xx - ac) * vy - (yy - ar) * vx
        sign = np.sign(cross)
        mask = d < signed
        signed = np.where(mask, d, signed)
        sign_field = np.where(mask, sign, sign_field)

    ramp_px = ramp_m / PX
    s = np.clip(0.5 - (signed * sign_field) / (2 * ramp_px), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    h += height_m * s


# ── boulder placement ───────────────────────────────────────────────────────

def sample_height_m(h_m: np.ndarray, x: float, y: float) -> float:
    c, r = world_to_pix(x, y)
    return float(h_m[r, c])


def boulders_around_craters(h_m: np.ndarray, fresh_craters,
                              rng: random.Random, per_crater_avg=8):
    """Yield (x, y, z, yaw, radius) ejecta boulders near fresh craters."""
    for (cx, cy, R) in fresh_craters:
        n = max(0, int(rng.gauss(per_crater_avg, 3)))
        for _ in range(n):
            dist = R * rng.uniform(1.1, 2.4)        # ejecta within ~2 R
            theta = rng.uniform(-math.pi, math.pi)
            x = cx + dist * math.cos(theta)
            y = cy + dist * math.sin(theta)
            if (x * x + y * y) < (SPAWN_FLAT_R + 1.0) ** 2:
                continue
            if abs(x) > 48 or abs(y) > 48:
                continue
            radius = rng.choice([
                rng.uniform(0.18, 0.32),
                rng.uniform(0.32, 0.55),
                rng.uniform(0.55, 0.85),
            ])
            yaw = rng.uniform(-math.pi, math.pi)
            z = sample_height_m(h_m, x, y) + radius * 0.55
            yield (x, y, z, yaw, radius)


def background_boulders(h_m: np.ndarray, rng: random.Random, n=80):
    for _ in range(n):
        for _try in range(40):
            x = rng.uniform(-46, 46)
            y = rng.uniform(-46, 46)
            if (x * x + y * y) >= (SPAWN_FLAT_R + 2.0) ** 2:
                break
        else:
            continue
        radius = rng.choice([
            rng.uniform(0.18, 0.30),
            rng.uniform(0.30, 0.50),
        ])
        yaw = rng.uniform(-math.pi, math.pi)
        z = sample_height_m(h_m, x, y) + radius * 0.55
        yield (x, y, z, yaw, radius)


# ── master heightmap build ──────────────────────────────────────────────────

def build_heightmap(seed: int = 13) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    h = np.zeros((RES, RES), dtype=np.float32)

    # Regolith texture: very subtle long+short wavelength roll
    add_smooth_noise(h, scale_px=64, amp=0.20, rng=rng)
    add_smooth_noise(h, scale_px=12, amp=0.06, rng=rng)
    add_smooth_noise(h, scale_px=4,  amp=0.025, rng=rng)

    # Mare crater field (the dominant lunar feature). Push crater centers
    # well outside the spawn zone so their bowls don't leak into the soft
    # feather and create steep ramps near the rover's start.
    fresh_mare = crater_field(
        h, rng,
        region_xy_radius=44.0,
        small_n=70, med_n=18, large_n=3,
        exclude_r=SPAWN_NO_CRATER_R,
        freshness_mix=(0.45, 1.0),
    )

    # Highlands patch in NE: raised base + dense overlapping cratering
    cc_hl, rr_hl = world_to_pix(+25.0, +25.0)
    yy, xx = np.indices((RES, RES), dtype=np.float32)
    d_hl = np.hypot(xx - cc_hl, yy - rr_hl) * PX
    hl_mask = np.clip(1.0 - (d_hl - 13.0) / 5.0, 0.0, 1.0)
    hl_mask = hl_mask * hl_mask * (3.0 - 2.0 * hl_mask)
    h += 1.6 * hl_mask
    fresh_hl = crater_field(
        h, rng,
        region_xy_radius=12.0, center_xy=(+25.0, +25.0),
        small_n=45, med_n=10, large_n=1,
        exclude_r=0.0,
        freshness_mix=(0.55, 1.0),
    )

    # Hadley-style sinuous rille across S half (lava channel).
    # Split into 3 disjoint segments so two natural "land bridges" remain —
    # the rover can always cross the south half. Bridges are at roughly
    # x ≈ -10 and x ≈ +20 (each ~7 m wide of intact regolith).
    rille_seg_1 = [
        (-46.0, -8.0), (-32.0, -14.0), (-18.0, -6.0), (-13.0, -10.0),
    ]
    rille_seg_2 = [
        (-7.0,  -13.0), (-4.0, -16.0), (+10.0, -8.0), (+16.0, -12.0),
    ]
    rille_seg_3 = [
        (+24.0, -18.0), (+38.0, -10.0), (+46.0, -16.0),
    ]
    for seg in (rille_seg_1, rille_seg_2, rille_seg_3):
        # Outer rille walls (shallower so rover sees ramped sides at segment
        # ends rather than a vertical wall).
        carve_channel(h, seg, half_width_m=3.0, depth_m=-1.8,
                       wall_steepness=1.0)
        # Inner sub-channel for the signature steep-walled look.
        carve_channel(h, seg, half_width_m=1.3, depth_m=-0.5,
                       wall_steepness=1.0)

    # Lobate scarp (thrust-fault cliff) crossing NW quadrant.
    # Lowered from 2.4 m → 1.3 m and ramp widened so it is climbable at 0.3 m/s.
    scarp_pts = [
        (-42.0, +5.0), (-32.0, +12.0), (-22.0, +14.0),
        (-12.0, +10.0), (-3.0, +16.0),
    ]
    add_lobate_scarp(h, scarp_pts, height_m=1.3, ramp_m=2.0)

    # Wrinkle ridge: low elongated ridge across SW (compression feature)
    wridge_pts = [(-40.0, +30.0), (-22.0, +28.0), (-8.0, +32.0), (+8.0, +30.0)]
    # Build as a low ridge by carving a "negative depth" channel
    carve_channel(h, wridge_pts, half_width_m=2.5, depth_m=+0.7,
                   wall_steepness=1.0)

    # Flatten the spawn region: full-flat for r ≤ SPAWN_FLAT_R, then a soft
    # SPAWN_FEATHER_M-wide smoothstep ramp to full terrain. Wider than before
    # so the rover's first few mission radii (curriculum starts at dmax=2 m)
    # see only gentle slopes.
    cc, rr = world_to_pix(0.0, 0.0)
    d_m = np.hypot(xx - cc, yy - rr) * PX
    flat_w = np.clip(1.0 - (d_m - SPAWN_FLAT_R) / SPAWN_FEATHER_M, 0.0, 1.0)
    flat_w = flat_w * flat_w * (3.0 - 2.0 * flat_w)
    h *= (1.0 - flat_w * 0.97)

    return h, fresh_mare + fresh_hl


def heightmap_to_uint8(h_m: np.ndarray) -> np.ndarray:
    z = np.clip(h_m, -VERT_M / 2 + 0.05, +VERT_M / 2 - 0.05)
    return np.clip((z + VERT_M / 2) / VERT_M * 255.0, 0, 255).astype(np.uint8)


# ── traversability mask ─────────────────────────────────────────────────────

# Slope and obstacle thresholds tuned for the Jackal at 0.5 m/s on lunar g.
# The rover routinely flips above 60° tilt; we mark anything with terrain
# slope > 25° as no-go to keep a safety margin (the rover crests slopes,
# its instantaneous tilt can spike well above the local terrain slope).
TRAV_MAX_SLOPE_DEG = 25.0
BOULDER_SAFETY_MARGIN_M = 1.5     # rover half-width 0.4 m + clearance
NOGO_BLOB_MIN_PX = 20             # ignore tiny isolated no-go pixels


def compute_traversability(
    h_m: np.ndarray,
    boulders: list,           # [(x, y, z, yaw, r), …]
) -> np.ndarray:
    """Boolean mask: True = drivable, False = no-go.

    Combines:
      1. Slope from heightmap gradient (max slope > TRAV_MAX_SLOPE_DEG → no-go)
      2. Buffered boulder footprints (rover safety margin)
    """
    # 1) Slope from gradient. h_m is in metres, sample spacing is PX metres.
    gy, gx = np.gradient(h_m, PX, PX)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)
    drivable = slope_deg < TRAV_MAX_SLOPE_DEG

    # 2) Boulder buffers — rasterize a disk of radius (rock_r + margin) at
    #    each boulder's pixel coordinate.
    yy, xx = np.indices((RES, RES), dtype=np.float32)
    for (bx, by, _bz, _byaw, br) in boulders:
        cc, rr = world_to_pix(bx, by)
        buffer_px = (br + BOULDER_SAFETY_MARGIN_M) / PX
        d2 = (xx - cc) ** 2 + (yy - rr) ** 2
        drivable &= (d2 > buffer_px * buffer_px)

    # 3) Erode tiny no-go specks so the visualization isn't a confetti.
    #    Using a simple connected-components filter via scipy if available,
    #    otherwise fall back to morphological close.
    try:
        from scipy.ndimage import binary_dilation, binary_erosion, label
        # Smooth the mask: close 1-px gaps, then drop blobs smaller than min.
        nogo = ~drivable
        nogo = binary_erosion(binary_dilation(nogo, iterations=1), iterations=1)
        labels, n = label(nogo)
        for k in range(1, n + 1):
            blob = (labels == k)
            if blob.sum() < NOGO_BLOB_MIN_PX:
                nogo[blob] = False
        drivable = ~nogo
    except ImportError:
        pass    # scipy missing — keep raw mask, less pretty but functional

    return drivable


def write_terrain_dae(
    h_m: np.ndarray,
    out_path: Path,
    decimate: int = 4,            # take every Nth vertex (513 / 4 ≈ 128 grid)
):
    """Convert the height field to a COLLADA (.dae) mesh that RViz2 can show
    as a Marker.MESH_RESOURCE. ~32k triangles at decimate=4 — plenty for an
    ovserview view, drops to <0.5 ms render time on the RTX 8000."""
    rows, cols = h_m.shape
    vy = np.linspace(WORLD_M / 2, -WORLD_M / 2, rows)[::decimate]
    vx = np.linspace(-WORLD_M / 2, WORLD_M / 2, cols)[::decimate]
    h = h_m[::decimate, ::decimate]
    R, C = h.shape
    # Vertices
    pos = []
    for r in range(R):
        for c in range(C):
            pos.append((vx[c], vy[r], float(h[r, c])))
    # Indices (two triangles per quad)
    idx = []
    for r in range(R - 1):
        for c in range(C - 1):
            a = r * C + c
            b = r * C + c + 1
            c0 = (r + 1) * C + c
            d = (r + 1) * C + c + 1
            idx.extend([a, b, c0, b, d, c0])
    n_pos = len(pos)
    n_tri = len(idx) // 3

    pos_str = " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for (x, y, z) in pos)
    idx_str = " ".join(str(i) for i in idx)

    dae = f"""<?xml version="1.0" encoding="UTF-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="moon_effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <emission><color>0.05 0.05 0.05 1</color></emission>
            <ambient><color>0.20 0.18 0.16 1</color></ambient>
            <diffuse><color>0.55 0.50 0.45 1</color></diffuse>
            <specular><color>0.05 0.05 0.05 1</color></specular>
            <shininess><float>10</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="moon_mat" name="moon_mat">
      <instance_effect url="#moon_effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="terrain_geo" name="terrain">
      <mesh>
        <source id="positions">
          <float_array id="positions-array" count="{n_pos*3}">{pos_str}</float_array>
          <technique_common>
            <accessor source="#positions-array" count="{n_pos}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
        <triangles count="{n_tri}" material="moon_mat">
          <input semantic="VERTEX" source="#vertices" offset="0"/>
          <p>{idx_str}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="terrain_node">
        <instance_geometry url="#terrain_geo">
          <bind_material>
            <technique_common>
              <instance_material symbol="moon_mat" target="#moon_mat"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dae)
    return n_pos, n_tri


def find_nogo_centroids(drivable: np.ndarray, max_blobs: int = 60) -> list:
    """Cluster no-go pixels and return world (x, y, radius_m) of each blob.
    Used to spawn red-tinted disks in gz Sim so the user can SEE no-go zones."""
    try:
        from scipy.ndimage import label, center_of_mass
    except ImportError:
        return []
    nogo = ~drivable
    labels, n = label(nogo)
    out = []
    for k in range(1, n + 1):
        blob = (labels == k)
        size_px = int(blob.sum())
        if size_px < NOGO_BLOB_MIN_PX:
            continue
        cy, cx = center_of_mass(blob)
        # Effective radius assuming circular blob.
        radius_m = math.sqrt(size_px / math.pi) * PX
        wx = cx * PX - WORLD_M / 2
        wy = WORLD_M / 2 - cy * PX
        out.append((wx, wy, radius_m, size_px))
    out.sort(key=lambda t: -t[3])
    return [(x, y, r) for (x, y, r, _) in out[:max_blobs]]


# ── SDF emit ────────────────────────────────────────────────────────────────

WORLD_HEAD = """\
<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="moon_varied">

    <!-- Gazebo Sim "Harmonic" requires plugins to be declared explicitly,
         unlike Gazebo Classic which loaded them implicitly. -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system"
            name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <physics name="lunar_phys" type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -1.62</gravity>
    <magnetic_field>6e-6 2.3e-5 -4.2e-5</magnetic_field>

    <scene>
      <ambient>0.10 0.10 0.12 1.0</ambient>
      <background>0.01 0.01 0.02 1.0</background>
      <!-- Shadows OFF: OGRE-Next 2.3.3's Hlms PBS shadow shader references
           an undefined `detailCol0` variable (line 655, 0(655):error C1503)
           which Mesa silently accepts but NVIDIA rejects, killing gz sim
           on ranger init. With shadows off, the fragment shader takes a
           simpler code path that compiles cleanly. -->
      <shadows>false</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 80 0 0.9 0</pose>
      <diffuse>1.0 0.95 0.85 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <direction>0.3 0.2 -0.9</direction>
    </light>

    <model name="terrain">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <heightmap>
              <uri>file://{HEIGHTMAP_URI}</uri>
              <size>{WORLD_M:.2f} {WORLD_M:.2f} {VERT_M:.2f}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <heightmap>
              <uri>file://{HEIGHTMAP_URI}</uri>
              <size>{WORLD_M:.2f} {WORLD_M:.2f} {VERT_M:.2f}</size>
              <pos>0 0 0</pos>
              <use_terrain_paging>false</use_terrain_paging>
            </heightmap>
          </geometry>
          <material>
            <ambient>0.32 0.30 0.28 1</ambient>
            <diffuse>0.55 0.52 0.48 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

# Boulder template. We deliberately OMIT the `<material>` element here even
# though each rock should look basaltic gray: gz-rendering 8 + OGRE-Next
# 2.3.3 doesn't dedupe identical inline materials, so 459 boulders × 1
# unique datablock each blows the texture budget ("Texture memory budget
# exceeded. Stalling GPU.") and adds 60 + seconds to sim warm-up. With no
# explicit material, all boulders share OGRE's default Hlms PBS datablock,
# which is a single allocation regardless of count. The boulders show up
# in gz sim as light gray spheres — fine for sensor rendering and for the
# RViz2 chase view (we don't see the boulders in RViz unless we publish
# their visualization markers separately).
ROCK_TMPL = """\
    <model name="rock_{idx:03d}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>
      <link name="link">
        <collision name="c">
          <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
        </collision>
        <visual name="v">
          <geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>
        </visual>
      </link>
    </model>
"""

WORLD_TAIL = """\

    <gui>
      <camera name="user_camera">
        <pose>-25 -25 18 0 0.55 0.78</pose>
      </camera>
    </gui>

  </world>
</sdf>
"""

# Translucent red disk that visually marks a no-go region in the world.
# `radius` is in metres; the disk is 0.05 m thick, sits at z=0.05 + sampled
# heightmap z so it hugs the ground.
# No-go disks are now published ONLY as RViz2 MarkerArray by
# scripts/terrain_marker_publisher.py. We deliberately don't spawn them as
# gz models — each one would carve out its own Hlms datablock and contribute
# to the same texture-budget stall that crippled startup.
NOGO_DISK_TMPL = ""


def main():
    proj = Path(__file__).resolve().parents[1]
    hm_dir = proj / "ros_ws/src/terrain_dreamer_bringup/worlds/heightmaps"
    world_path = proj / "ros_ws/src/terrain_dreamer_bringup/worlds/varied.world"
    hm_path = hm_dir / "varied.png"
    trav_path = hm_dir / "varied_traversable.npy"

    print(f"[varied] generating heightmap → {hm_path}")
    h, fresh_craters = build_heightmap(seed=13)
    img = Image.fromarray(heightmap_to_uint8(h), mode="L")
    img.save(hm_path)
    print(f"[varied]   z range: {h.min():+.2f} .. {h.max():+.2f} m")
    print(f"[varied]   {len(fresh_craters)} fresh craters (boulder ejecta sources)")

    rng = random.Random(11)
    boulders = list(boulders_around_craters(h, fresh_craters, rng,
                                              per_crater_avg=6))
    boulders += list(background_boulders(h, rng, n=70))
    print(f"[varied] total boulders: {len(boulders)}")

    print("[varied] computing traversability mask …")
    drivable = compute_traversability(h, boulders)
    np.save(trav_path, drivable)
    pct = 100.0 * drivable.sum() / drivable.size
    print(f"[varied]   drivable: {pct:.1f}% of world  → {trav_path}")

    # Terrain mesh for RViz2 (lightweight UI replaces heavy gz GUI).
    dae_path = hm_dir / "varied_terrain.dae"
    n_pos, n_tri = write_terrain_dae(h, dae_path, decimate=4)
    print(f"[varied]   terrain mesh: {n_pos} verts, {n_tri} tris → {dae_path}")

    nogo_centroids = find_nogo_centroids(drivable, max_blobs=40)
    print(f"[varied]   no-go zones to mark: {len(nogo_centroids)}")

    # Sidecar JSON: terrain_marker_publisher.py reads this so the no-go
    # visualization can live entirely in RViz2 without bloating the gz world.
    import json
    nogo_json = hm_dir / "varied_nogo.json"
    nogo_records = []
    for (x, y, r) in nogo_centroids:
        ground_z = sample_height_m(h, x, y) + 0.04
        nogo_records.append({"x": float(x), "y": float(y),
                               "z": float(ground_z), "r": float(r)})
    nogo_json.write_text(json.dumps(nogo_records, indent=2))
    print(f"[varied]   no-go centroids → {nogo_json}")

    with open(world_path, "w") as f:
        f.write(WORLD_HEAD.format(
            HEIGHTMAP_URI=str(hm_path),
            WORLD_M=WORLD_M, VERT_M=VERT_M,
        ))
        for i, (x, y, z, yaw, r) in enumerate(boulders):
            f.write(ROCK_TMPL.format(idx=i, x=x, y=y, z=z, yaw=yaw, r=r))
        f.write(WORLD_TAIL)

    print(f"[varied] wrote {world_path}")
    print(f"[varied] launch with: ./run.sh --mode human")


if __name__ == "__main__":
    main()
