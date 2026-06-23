/**
 * Coordinate transforms used by the in-browser Jackal chase view.
 *
 * Four coordinate frames:
 *
 *   - DEM-metre frame (M):  origin at tile centre, +X east, +Y north.
 *     This is the *odometry* frame the rover publishes — the Jackal SDF
 *     was spawned at (0,0) and the diff-drive odom frame is centred on
 *     the heightmap. We treat M and odom as identical.
 *
 *   - Heightmap-pixel frame (P):  origin at top-left of the PNG,
 *     +X right (=column), +Y down (=row). 0..samples-1.
 *
 *   - Three.js world frame (W):  Y-up. We map ROS(+X east, +Y north,
 *     +Z up) onto W( x = +X_M, y = +Z_M, z = -Y_M ). This keeps the
 *     conventional Three.js camera "up" axis correct.
 *
 *   - Heightmap UV frame:  [0,1]² for shader sampling. Provided here
 *     for future shader-based terrain coloring.
 *
 * Every function is pure — importable by tests without React/Three.js.
 */

export interface TileMetadata {
  tile_width_m: number;
  tile_height_m: number;
  samples_x: number;
  samples_y: number;
  min_elevation_m: number;
  max_elevation_m: number;
  vertical_scale_m: number;
  horizontal_resolution_m_per_pixel: number;
}

// ───────────────────────────────────────────────────────────────────────────
// DEM-metre  ↔  Heightmap-pixel
// ───────────────────────────────────────────────────────────────────────────

export interface PixelXY { col: number; row: number; }
export interface MapXY { x: number; y: number; }

/** Convert (x, y) in DEM metres to floating-point pixel (col, row). */
export function mapToPixel(p: MapXY, meta: TileMetadata): PixelXY {
  const halfW = meta.tile_width_m / 2;
  const halfH = meta.tile_height_m / 2;
  const px = (p.x + halfW) / meta.tile_width_m * (meta.samples_x - 1);
  // Row 0 is top of image (north = +Y), so flip Y.
  const py = (halfH - p.y) / meta.tile_height_m * (meta.samples_y - 1);
  return { col: px, row: py };
}

/** Inverse of mapToPixel — pixel (col, row) to DEM metres. */
export function pixelToMap(p: PixelXY, meta: TileMetadata): MapXY {
  const halfW = meta.tile_width_m / 2;
  const halfH = meta.tile_height_m / 2;
  const x = p.col / (meta.samples_x - 1) * meta.tile_width_m - halfW;
  const y = halfH - p.row / (meta.samples_y - 1) * meta.tile_height_m;
  return { x, y };
}

// ───────────────────────────────────────────────────────────────────────────
// DEM-metre  ↔  Three.js world (Y-up)
// ───────────────────────────────────────────────────────────────────────────

export interface World3 { x: number; y: number; z: number; }

/**
 * Map DEM(+X east, +Y north, +Z up) → Three.js(+X east, +Y up, +Z south).
 * z_m is the terrain height in metres (already de-normalised).
 */
export function mapToWorld(p: { x: number; y: number; z: number }): World3 {
  return { x: p.x, y: p.z, z: -p.y };
}

/** Inverse of mapToWorld. */
export function worldToMap(w: World3): { x: number; y: number; z: number } {
  return { x: w.x, y: -w.z, z: w.y };
}

/**
 * Convert a yaw in the DEM frame (radians, CCW from +X) to a Three.js
 * yaw about the +Y axis (CCW from +X looking down). Because our +Y in
 * Three.js was +Z in the DEM, the sign flips.
 */
export function dem_yaw_to_three_yaw(yaw_rad: number): number {
  return -yaw_rad;
}

// ───────────────────────────────────────────────────────────────────────────
// Heightmap UV
// ───────────────────────────────────────────────────────────────────────────

export function mapToUV(p: MapXY, meta: TileMetadata): { u: number; v: number } {
  const halfW = meta.tile_width_m / 2;
  const halfH = meta.tile_height_m / 2;
  return {
    u: (p.x + halfW) / meta.tile_width_m,
    v: (halfH - p.y) / meta.tile_height_m,
  };
}
