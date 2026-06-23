import { describe, it, expect } from "vitest";
import {
  mapToPixel, pixelToMap, mapToWorld, worldToMap,
  dem_yaw_to_three_yaw, mapToUV,
  type TileMetadata,
} from "../lib/coordinateTransform";
import {
  sampleHeight, sampleIntensity, makeSampler, parseFlatYaml,
} from "../lib/terrainHeight";
import {
  chasePoseFor, smoothPose, topDownPoseFor,
  CHASE_DEFAULTS, DRIVER_DEFAULTS,
} from "../lib/chaseCamera";
import { mockJackalPose } from "../hooks/useJackalPose";
import {
  CAMERA_MODES, TERRAIN_GRID_BY_QUALITY,
} from "../hooks/useCameraMode";

const META: TileMetadata = {
  tile_width_m: 1000,
  tile_height_m: 1000,
  samples_x: 101,
  samples_y: 101,
  min_elevation_m: 0,
  max_elevation_m: 100,
  vertical_scale_m: 100,
  horizontal_resolution_m_per_pixel: 10,
};

// ───────────────────────────────────────────────────────────────────────────
// coordinateTransform
// ───────────────────────────────────────────────────────────────────────────

describe("coordinateTransform", () => {
  it("mapToPixel maps origin to centre", () => {
    const p = mapToPixel({ x: 0, y: 0 }, META);
    expect(p.col).toBeCloseTo(50, 5);
    expect(p.row).toBeCloseTo(50, 5);
  });

  it("mapToPixel handles tile corners", () => {
    const ul = mapToPixel({ x: -500, y: 500 }, META);    // upper-left
    const lr = mapToPixel({ x: 500, y: -500 }, META);    // lower-right
    expect(ul.col).toBeCloseTo(0, 5);
    expect(ul.row).toBeCloseTo(0, 5);
    expect(lr.col).toBeCloseTo(100, 5);
    expect(lr.row).toBeCloseTo(100, 5);
  });

  it("pixelToMap is the inverse of mapToPixel", () => {
    const original = { x: 73.5, y: -212.0 };
    const round = pixelToMap(mapToPixel(original, META), META);
    expect(round.x).toBeCloseTo(original.x, 4);
    expect(round.y).toBeCloseTo(original.y, 4);
  });

  it("mapToWorld swaps Z↔Y and negates Y", () => {
    const w = mapToWorld({ x: 1, y: 2, z: 3 });
    expect(w.x).toBe(1);
    expect(w.y).toBe(3);
    expect(w.z).toBe(-2);
    const back = worldToMap(w);
    expect(back.x).toBe(1);
    expect(back.y).toBe(2);
    expect(back.z).toBe(3);
  });

  it("dem_yaw_to_three_yaw flips sign", () => {
    expect(dem_yaw_to_three_yaw(1.0)).toBe(-1.0);
    expect(dem_yaw_to_three_yaw(0)).toBeCloseTo(0, 12);
  });

  it("mapToUV at centre is (0.5, 0.5)", () => {
    const uv = mapToUV({ x: 0, y: 0 }, META);
    expect(uv.u).toBeCloseTo(0.5, 5);
    expect(uv.v).toBeCloseTo(0.5, 5);
  });
});

// ───────────────────────────────────────────────────────────────────────────
// terrainHeight
// ───────────────────────────────────────────────────────────────────────────

describe("terrainHeight", () => {
  /** Build a tiny 3×3 sampler whose centre pixel intensity = 1.0,
   *  edges = 0.0. */
  function tinySampler() {
    const w = 3, h = 3;
    const rgba = new Uint8ClampedArray(w * h * 4);
    // pixel (1,1) = centre, value 255
    const idx = (1 * w + 1) * 4;
    rgba[idx] = 255;
    rgba[idx + 1] = 255;
    rgba[idx + 2] = 255;
    rgba[idx + 3] = 255;
    return makeSampler({
      width: w, height: h, rgba,
      meta: { ...META, samples_x: w, samples_y: h },
    });
  }

  it("sampleIntensity at the exact centre pixel returns 1.0", () => {
    const s = tinySampler();
    expect(sampleIntensity(s, 1, 1)).toBeCloseTo(1.0, 5);
  });

  it("sampleIntensity bilinearly interpolates between two pixels", () => {
    const s = tinySampler();
    // Halfway between centre (1.0) and edge (0.0).
    const v = sampleIntensity(s, 0.5, 1);
    expect(v).toBeCloseTo(0.5, 3);
  });

  it("sampleHeight converts intensity → elevation in metres", () => {
    const s = tinySampler();
    // Centre: intensity=1 → elev = min(0) + 100 * 1 = 100
    expect(sampleHeight(s, { x: 0, y: 0 })).toBeCloseTo(100, 3);
  });

  it("sampleHeight at tile corner returns min elevation", () => {
    const s = tinySampler();
    // Use the corner pixel which is 0
    expect(sampleHeight(s, { x: -500, y: 500 })).toBeCloseTo(0, 3);
  });

  it("parseFlatYaml reads numbers and strings", () => {
    const out = parseFlatYaml(`
      # heightmap metadata
      tile_width_m: 2048.0
      samples_x: 1025
      source_dem: data/raw_dem/foo.tif
    `);
    expect(out.tile_width_m).toBe(2048.0);
    expect(out.samples_x).toBe(1025);
    expect(out.source_dem).toBe("data/raw_dem/foo.tif");
  });
});

// ───────────────────────────────────────────────────────────────────────────
// chaseCamera
// ───────────────────────────────────────────────────────────────────────────

describe("chaseCamera", () => {
  it("chasePoseFor yaw=0 places camera behind +X and looks at robot", () => {
    const p = chasePoseFor({ x: 10, y: 0, z: 0, yaw_rad: 0 }, CHASE_DEFAULTS);
    // Behind +X means camera.x < robot.x
    expect(p.position.x).toBeLessThan(10);
    expect(p.position.x).toBeCloseTo(10 - CHASE_DEFAULTS.behind_m, 5);
    expect(p.position.y).toBeCloseTo(CHASE_DEFAULTS.height_m, 5);
    expect(p.lookAt.x).toBe(10);
  });

  it("chasePoseFor yaw=90deg places camera behind -Z (robot facing -Z)", () => {
    // yaw=π/2 → forward direction (cos, -sin) = (0, -1) in (x, z).
    // Camera is then at robot + (0, +1) * behind = +Z direction.
    const p = chasePoseFor({ x: 0, y: 0, z: 0, yaw_rad: Math.PI / 2 },
                           CHASE_DEFAULTS);
    expect(p.position.x).toBeCloseTo(0, 4);
    expect(p.position.z).toBeCloseTo(CHASE_DEFAULTS.behind_m, 4);
  });

  it("smoothPose with alpha=1 returns target exactly", () => {
    const cur  = { position: { x: 0, y: 0, z: 0 },
                   lookAt:   { x: 1, y: 1, z: 1 } };
    const tgt  = { position: { x: 9, y: 9, z: 9 },
                   lookAt:   { x: 2, y: 2, z: 2 } };
    const out = smoothPose(cur, tgt, 1.0);
    expect(out.position).toEqual(tgt.position);
    expect(out.lookAt).toEqual(tgt.lookAt);
  });

  it("smoothPose with alpha=0 returns current unchanged", () => {
    const cur  = { position: { x: 0, y: 0, z: 0 },
                   lookAt:   { x: 1, y: 1, z: 1 } };
    const tgt  = { position: { x: 9, y: 9, z: 9 },
                   lookAt:   { x: 2, y: 2, z: 2 } };
    const out = smoothPose(cur, tgt, 0);
    expect(out.position).toEqual(cur.position);
    expect(out.lookAt).toEqual(cur.lookAt);
  });

  it("topDownPoseFor places camera straight above robot", () => {
    const p = topDownPoseFor({ x: 5, y: 1, z: -3, yaw_rad: 0 }, 30);
    expect(p.position.x).toBe(5);
    expect(p.position.z).toBe(-3);
    expect(p.position.y).toBe(31);
    expect(p.lookAt).toEqual({ x: 5, y: 1, z: -3 });
  });

  it("driver-mode offsets are smaller than chase offsets", () => {
    expect(DRIVER_DEFAULTS.height_m).toBeLessThan(CHASE_DEFAULTS.height_m);
    expect(DRIVER_DEFAULTS.behind_m).toBeLessThan(CHASE_DEFAULTS.behind_m);
  });
});

// ───────────────────────────────────────────────────────────────────────────
// mock trajectory
// ───────────────────────────────────────────────────────────────────────────

describe("mockJackalPose", () => {
  it("yields t=0 at origin with heading along +X", () => {
    const p = mockJackalPose(0);
    expect(p.x).toBeCloseTo(0, 5);
    expect(p.y).toBeCloseTo(0, 5);
    // At t=0, dx=A*ω, dy=2B*ω → both >0 in first quadrant.
    expect(p.yaw).toBeGreaterThan(0);
    expect(p.yaw).toBeLessThan(Math.PI / 2);
  });

  it("stays within the configured 50m × 25m envelope", () => {
    for (let t = 0; t < 200; t += 0.1) {
      const p = mockJackalPose(t);
      expect(Math.abs(p.x)).toBeLessThanOrEqual(51);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(26);
    }
  });
});

// ───────────────────────────────────────────────────────────────────────────
// useCameraMode constants
// ───────────────────────────────────────────────────────────────────────────

describe("useCameraMode constants", () => {
  it("has all four required modes from the spec", () => {
    const ids = CAMERA_MODES.map((m) => m.id);
    expect(ids).toContain("chase");
    expect(ids).toContain("topdown");
    expect(ids).toContain("orbit");
    expect(ids).toContain("driver");
  });

  it("terrain quality grid sizes are ascending", () => {
    expect(TERRAIN_GRID_BY_QUALITY.low)
      .toBeLessThan(TERRAIN_GRID_BY_QUALITY.medium);
    expect(TERRAIN_GRID_BY_QUALITY.medium)
      .toBeLessThan(TERRAIN_GRID_BY_QUALITY.high);
  });
});
