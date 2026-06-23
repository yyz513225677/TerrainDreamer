/**
 * Bilinear sampling of a 16-bit grey heightmap PNG.
 *
 * The browser decodes the PNG into a Uint8ClampedArray of RGBA bytes;
 * for a "I;16"-style grey heightmap we read the R channel (the loader
 * collapses the 16-bit value to 8-bit). That is lossy but plenty for
 * marker placement — sub-pixel-of-relief precision is not required.
 *
 * For higher-precision elevation queries (e.g. terrain-aware planning),
 * the same JSONL metadata + Float32 GeoTIFF used by Phase 1 can be
 * sampled directly server-side.
 */
import type { TileMetadata } from "./coordinateTransform";
import { mapToPixel, type MapXY } from "./coordinateTransform";

export interface HeightSampler {
  /** Width and height in pixels. */
  width: number;
  height: number;
  /** Grey channel intensity in [0, 1]. */
  intensity: Float32Array;
  meta: TileMetadata;
}

/** Sample a HeightSampler at a (col, row) using bilinear interpolation. */
export function sampleIntensity(s: HeightSampler, col: number, row: number): number {
  const c0 = Math.floor(col);
  const r0 = Math.floor(row);
  const cf = col - c0;
  const rf = row - r0;
  const c1 = Math.min(c0 + 1, s.width - 1);
  const r1 = Math.min(r0 + 1, s.height - 1);
  const cc0 = Math.max(0, c0);
  const rr0 = Math.max(0, r0);
  const i00 = s.intensity[rr0 * s.width + cc0];
  const i10 = s.intensity[rr0 * s.width + c1];
  const i01 = s.intensity[r1 * s.width + cc0];
  const i11 = s.intensity[r1 * s.width + c1];
  const top = i00 * (1 - cf) + i10 * cf;
  const bot = i01 * (1 - cf) + i11 * cf;
  return top * (1 - rf) + bot * rf;
}

/** Sample terrain elevation in metres at a DEM-metre (x, y) point. */
export function sampleHeight(s: HeightSampler, p: MapXY): number {
  const { col, row } = mapToPixel(p, s.meta);
  const i = sampleIntensity(s, col, row);
  return s.meta.min_elevation_m + i * s.meta.vertical_scale_m;
}

/**
 * Build a HeightSampler from a raw pixel buffer. The buffer must be
 * RGBA (4 bytes/pixel); we read the R channel.
 */
export function makeSampler(opts: {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  meta: TileMetadata;
}): HeightSampler {
  const { width, height, rgba, meta } = opts;
  const intensity = new Float32Array(width * height);
  for (let i = 0, j = 0; j < intensity.length; j++, i += 4) {
    intensity[j] = rgba[i] / 255;
  }
  return { width, height, intensity, meta };
}

/**
 * Load the heightmap PNG + tile metadata YAML from the dashboard's
 * /public/dem/ folder (or any URL pair). Returns a sampler ready to
 * query.
 *
 * The metadata YAML is read with a tiny key/value parser because the
 * file is flat and we don't want to pull in `js-yaml` as a dependency
 * just for ~12 fields. If we ever need anchors / nested maps we will
 * switch.
 */
export async function loadHeightSampler(
  heightmapUrl: string,
  metadataUrl: string,
): Promise<HeightSampler> {
  const [imgBlob, yamlText] = await Promise.all([
    fetch(heightmapUrl).then((r) => r.blob()),
    fetch(metadataUrl).then((r) => r.text()),
  ]);
  const meta = parseFlatYaml(yamlText) as unknown as TileMetadata;
  const url = URL.createObjectURL(imgBlob);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const el = new Image();
      el.onload = () => res(el);
      el.onerror = rej;
      el.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    ctx.drawImage(img, 0, 0);
    const { data } = ctx.getImageData(0, 0, img.width, img.height);
    return makeSampler({
      width: img.width,
      height: img.height,
      rgba: data,
      meta,
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Minimal flat-YAML parser: handles `key: value`, ignores comments
 * and blank lines, numeric values are parsed as Number. Returns the
 * exact object shape produced by `scripts/normalize_heightmap.py`.
 */
export function parseFlatYaml(text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const raw of text.split("\n")) {
    const line = raw.replace(/#.*$/, "").trim();
    if (!line || !line.includes(":")) continue;
    const [k, ...rest] = line.split(":");
    const v = rest.join(":").trim().replace(/^['"]|['"]$/g, "");
    const num = Number(v);
    out[k.trim()] = v === "" || Number.isNaN(num) ? v : num;
  }
  return out;
}
