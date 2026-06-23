/**
 * JackalChaseView — in-browser 3D third-person view of the rover on
 * the lunar DEM. Independent of Gazebo's GUI.
 *
 * Pipeline (entirely reuse-first):
 *   1. /public/dem/heightmap.png   → Three.js TextureLoader
 *   2. /public/dem/tile.yaml       → flat-YAML parser (lib/terrainHeight)
 *   3. PlaneGeometry (configurable resolution) displaced by heightmap
 *   4. Lunar gray PBR material + low-elevation directional sun
 *   5. Rover marker placed via mapToWorld + bilinear height lookup
 *   6. Chase camera math from lib/chaseCamera
 *   7. LiDAR rays drawn from useAppStore.lidar
 *
 * No custom 3D engine, no custom shader, no custom raster decoder —
 * just glue.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useTexture } from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

import { useAppStore } from "../lib/modeFSM";
import { useCameraMode, TERRAIN_GRID_BY_QUALITY } from "../hooks/useCameraMode";
import { useJackalPose } from "../hooks/useJackalPose";
import {
  HeightSampler, loadHeightSampler, sampleHeight,
} from "../lib/terrainHeight";
import {
  mapToWorld, dem_yaw_to_three_yaw,
} from "../lib/coordinateTransform";
import {
  CHASE_DEFAULTS, DRIVER_DEFAULTS,
  chasePoseFor, smoothPose, topDownPoseFor,
  type CameraPose, type PoseW,
} from "../lib/chaseCamera";
import { ChaseViewOverlay } from "./ChaseViewOverlay";

const DEM_HEIGHTMAP_URL = "/dem/heightmap.png";
const DEM_METADATA_URL = "/dem/tile.yaml";

export function JackalChaseView({ className = "" }: { className?: string }) {
  const [sampler, setSampler] = useState<HeightSampler | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadHeightSampler(DEM_HEIGHTMAP_URL, DEM_METADATA_URL)
      .then((s) => { if (!cancelled) setSampler(s); })
      .catch((e) => { if (!cancelled) setLoadErr(String(e)); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className={"relative bg-black " + className}>
      {loadErr && (
        <div className="absolute inset-0 flex items-center justify-center
                        text-amber-300 font-mono text-sm z-20">
          DEM load failed: {loadErr}
        </div>
      )}
      <Canvas
        camera={{ fov: 75, near: 0.1, far: 10000, position: [-12, 6, 12] }}
        dpr={[1, 1.5]}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#02030a"]} />
        <hemisphereLight args={["#1a2030", "#0a0b10", 0.15]} />
        <directionalLight
          position={[300, 80, 200]}
          intensity={1.6}
          color="#fff4e0"
          castShadow
        />
        <SceneContents sampler={sampler} />
      </Canvas>
      <ChaseViewOverlay />
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Scene
// ───────────────────────────────────────────────────────────────────────────

function SceneContents({ sampler }: { sampler: HeightSampler | null }) {
  const mode = useCameraMode((s) => s.mode);
  const quality = useCameraMode((s) => s.quality);
  const showGrid = useCameraMode((s) => s.showGrid);
  const showLidar = useCameraMode((s) => s.showLidar);
  const showRoutePath = useCameraMode((s) => s.showRoutePath);

  const robotPose = useJackalPose({ sampler });

  return (
    <>
      {sampler && (
        <TerrainMesh
          sampler={sampler}
          grid={TERRAIN_GRID_BY_QUALITY[quality]}
        />
      )}
      {showGrid && sampler && <TerrainGrid sampler={sampler} />}
      <RoverMarker pose={robotPose} />
      {showLidar && <LidarRays robotPose={robotPose} />}
      {showRoutePath && <RouteTrail pose={robotPose} />}
      <GoalMarker sampler={sampler} />
      <ChaseCameraController mode={mode} robotPose={robotPose} />
    </>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Terrain mesh (PlaneGeometry displaced by heightmap)
// ───────────────────────────────────────────────────────────────────────────

function TerrainMesh({ sampler, grid }:
                     { sampler: HeightSampler; grid: number }) {
  // PGDA-style colorized texture baked by Phase 1 (blue→green→yellow→red).
  const colorMap = useTexture("/dem/color.png");

  const geometry = useMemo(() => {
    const w = sampler.meta.tile_width_m;
    const h = sampler.meta.tile_height_m;
    const segs = grid - 1;
    const g = new THREE.PlaneGeometry(w, h, segs, segs);
    g.rotateX(-Math.PI / 2);
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const dem_x = x;
      const dem_y = -z;
      const elev = sampleHeight(sampler, { x: dem_x, y: dem_y });
      pos.setY(i, elev - sampler.meta.min_elevation_m);
    }
    // After rotateX(-π/2), default PlaneGeometry UVs need a vertical
    // flip so the texture's row 0 (top of PNG = +Y north) lands on
    // -Z in world (=+Y in DEM north).
    const uv = g.attributes.uv as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
    g.computeVertexNormals();
    return g;
  }, [sampler, grid]);

  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial
        map={colorMap}
        roughness={0.92}
        metalness={0.0}
      />
    </mesh>
  );
}

function TerrainGrid({ sampler }: { sampler: HeightSampler }) {
  const w = sampler.meta.tile_width_m;
  const h = sampler.meta.tile_height_m;
  const size = Math.max(w, h);
  const divisions = 20;
  return (
    <gridHelper
      args={[size, divisions, "#33c0d8", "#1a2030"]}
      position={[0, sampler.meta.vertical_scale_m + 0.05, 0]}
    />
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Rover marker (low-poly chassis)
// ───────────────────────────────────────────────────────────────────────────

// ── Real Clearpath Jackal meshes (fetched from jackal_description) ────────
// Loaded once into a module-level cache; reused across renders.
type StlGeo = THREE.BufferGeometry;
const stlCache: { [k: string]: StlGeo | "loading" | Error } = {};
function useSTL(url: string): StlGeo | null {
  const [geom, setGeom] = useState<StlGeo | null>(() => {
    const v = stlCache[url];
    return v instanceof THREE.BufferGeometry ? v : null;
  });
  useEffect(() => {
    const c = stlCache[url];
    if (c instanceof THREE.BufferGeometry) { setGeom(c); return; }
    if (c === "loading") return;
    stlCache[url] = "loading";
    const loader = new STLLoader();
    loader.load(
      url,
      (g: THREE.BufferGeometry) => {
        g.computeVertexNormals();
        stlCache[url] = g;
        setGeom(g);
      },
      undefined,
      (err: unknown) => { stlCache[url] = err as Error; },
    );
  }, [url]);
  return geom;
}

/**
 * Clearpath Jackal third-person marker — loads the real `.stl` meshes
 * shipped in `jackal_description` (base + fenders + wheels). Geometry
 * comes in metres; in the URDF the model is oriented +X = forward,
 * which matches our convention so no extra rotation is needed.
 *
 * Iconic colour scheme: yellow chassis, black fenders, black tires.
 */
function RoverMarker({ pose }: { pose: PoseW }) {
  const group = useRef<THREE.Group>(null);
  const baseGeo = useSTL("/jackal/jackal-base.stl");
  const fenderGeo = useSTL("/jackal/jackal-fenders.stl");
  const wheelGeo = useSTL("/jackal/jackal-wheel.stl");

  useFrame(() => {
    if (!group.current) return;
    group.current.position.set(pose.x, pose.y, pose.z);
    group.current.rotation.set(0, pose.yaw_rad, 0);
  });

  // Wheel positions from the URDF: track 0.323 m, wheelbase 0.262 m.
  const wheelXY: [number, number, string][] = [
    [ 0.131,  0.1615, "front_left"],
    [ 0.131, -0.1615, "front_right"],
    [-0.131,  0.1615, "rear_left"],
    [-0.131, -0.1615, "rear_right"],
  ];

  return (
    <group ref={group}>
      {/* Chassis — Clearpath Jackal signature yellow */}
      {baseGeo && (
        <mesh geometry={baseGeo} castShadow receiveShadow>
          <meshStandardMaterial color="#facc15" metalness={0.3} roughness={0.55} />
        </mesh>
      )}
      {/* Black fenders / bumpers */}
      {fenderGeo && (
        <mesh geometry={fenderGeo} castShadow>
          <meshStandardMaterial color="#0e0f12" metalness={0.2} roughness={0.7} />
        </mesh>
      )}
      {/* Four real wheels — STL is one wheel; placed at URDF coords.
          ROS Z-up → Three.js Y-up means the STL's native cylinder axis
          needs rotating 90° about X so the wheel rolls about the
          lateral (Three.js Z) axis. */}
      {wheelGeo && wheelXY.map(([wx, wy, key]) => (
        <mesh
          key={key}
          geometry={wheelGeo}
          position={[wx, 0.098, wy]}
          rotation={[Math.PI / 2, 0, 0]}
          castShadow
        >
          <meshStandardMaterial color="#15171b" metalness={0.1} roughness={0.85} />
        </mesh>
      ))}
      {/* Sensor tower (lidar dome) — top of chassis */}
      <mesh position={[0, 0.27, 0]}>
        <cylinderGeometry args={[0.05, 0.06, 0.08, 16]} />
        <meshStandardMaterial color="#0c1018" emissive="#22d3ee" emissiveIntensity={0.45} />
      </mesh>
      {/* VectorNav VN-100 Rugged IMU — datasheet 36 × 33 × 9 mm,
          black anodized aluminum housing, mounted aft of the lidar
          tower on top of the chassis. */}
      <VN100Imu position={[-0.08, 0.230, 0.0]} />
      {/* Loading fallback — visible until STLs arrive */}
      {!baseGeo && (
        <mesh position={[0, 0.10, 0]} castShadow>
          <boxGeometry args={[0.42, 0.184, 0.31]} />
          <meshStandardMaterial color="#facc15" wireframe />
        </mesh>
      )}
    </group>
  );
}

/**
 * Procedural model of the VectorNav VN-100 Rugged IMU
 * (datasheet: 36 × 33 × 9 mm aluminum housing, 2× M3 mounting holes,
 * DB-9 connector on one short side). Reuse-first: no public mesh
 * available, so this is built from primitives and dimensions.
 */
function VN100Imu({ position }: { position: [number, number, number] }) {
  // Conversion: datasheet mm → metres
  const W = 0.036, L = 0.033, H = 0.009;
  return (
    <group position={position}>
      {/* Main aluminum housing — anodized black, slightly metallic */}
      <mesh castShadow>
        <boxGeometry args={[W, H, L]} />
        <meshStandardMaterial
          color="#1a1c1f"
          metalness={0.55}
          roughness={0.42}
        />
      </mesh>
      {/* Top label decal — VectorNav logo area (slight white tint) */}
      <mesh position={[0, H / 2 + 0.0005, 0]}>
        <planeGeometry args={[W * 0.55, L * 0.42]} />
        <meshStandardMaterial
          color="#dadada"
          roughness={0.85}
          metalness={0.0}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Two M3 mounting holes — recessed black dots near opposite corners */}
      {[[-1, -1], [1, 1]].map(([sx, sz], i) => (
        <mesh
          key={i}
          position={[sx * (W / 2 - 0.005), H / 2 + 0.0006, sz * (L / 2 - 0.005)]}
          rotation={[Math.PI / 2, 0, 0]}
        >
          <cylinderGeometry args={[0.0017, 0.0017, 0.0005, 12]} />
          <meshStandardMaterial color="#000000" metalness={0.7} roughness={0.3} />
        </mesh>
      ))}
      {/* DB-9 connector stub on the +X face */}
      <mesh position={[W / 2 + 0.004, 0, 0]}>
        <boxGeometry args={[0.008, H * 0.7, L * 0.55]} />
        <meshStandardMaterial color="#3a3a3a" metalness={0.6} roughness={0.4} />
      </mesh>
      {/* Status LED — tiny green dot on the top face */}
      <mesh position={[W / 2 - 0.005, H / 2 + 0.0006, L / 2 - 0.004]}>
        <cylinderGeometry args={[0.0009, 0.0009, 0.0005, 8]} />
        <meshStandardMaterial
          color="#22c55e"
          emissive="#22c55e"
          emissiveIntensity={1.0}
        />
      </mesh>
    </group>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// LiDAR rays from store
// ───────────────────────────────────────────────────────────────────────────

function LidarRays({ robotPose }: { robotPose: PoseW }) {
  const lidar = useAppStore((s) => s.lidar);
  const geom = useMemo(() => {
    const positions: number[] = [];
    if (!lidar || !lidar.ranges) return null;
    const angle_min = lidar.angle_min ?? -Math.PI;
    const angle_inc = lidar.angle_increment ?? (2 * Math.PI / lidar.ranges.length);
    for (let i = 0; i < lidar.ranges.length; i++) {
      const r = lidar.ranges[i];
      if (!isFinite(r) || r <= 0 || r > 60) continue;
      const theta = angle_min + i * angle_inc + robotPose.yaw_rad;
      // Convert ROS-frame (yaw CCW from +X east) into Three.js direction
      // about +Y. Our convention: +X east, -Z north.
      const dx = Math.cos(theta);
      const dz = -Math.sin(theta);
      const ex = robotPose.x + r * dx;
      const ez = robotPose.z + r * dz;
      positions.push(robotPose.x, robotPose.y + 0.40, robotPose.z);
      positions.push(ex, robotPose.y + 0.40, ez);
    }
    if (!positions.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position",
      new THREE.Float32BufferAttribute(positions, 3));
    return g;
  }, [lidar, robotPose.x, robotPose.y, robotPose.z, robotPose.yaw_rad]);

  if (!geom) return null;
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color="#22d3ee" transparent opacity={0.55} />
    </lineSegments>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Route trail (kept in a ref — appended each frame, capped at 1024 points)
// ───────────────────────────────────────────────────────────────────────────

function RouteTrail({ pose }: { pose: PoseW }) {
  const positions = useRef<Float32Array>(new Float32Array(1024 * 3));
  const count = useRef(0);
  const lineObj = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position",
      new THREE.BufferAttribute(positions.current, 3));
    g.setDrawRange(0, 0);
    const m = new THREE.LineBasicMaterial({
      color: "#f59e0b", transparent: true, opacity: 0.85,
    });
    return new THREE.Line(g, m);
  }, []);

  useFrame(() => {
    if (count.current >= 1024) {
      positions.current.copyWithin(0, 3);
      count.current = 1023;
    }
    const j = count.current * 3;
    positions.current[j] = pose.x;
    positions.current[j + 1] = pose.y + 0.05;
    positions.current[j + 2] = pose.z;
    count.current++;
    lineObj.geometry.setDrawRange(0, count.current);
    (lineObj.geometry.attributes.position as THREE.BufferAttribute)
      .needsUpdate = true;
  });

  return <primitive object={lineObj} />;
}

// ───────────────────────────────────────────────────────────────────────────
// Goal marker (subscribes to store telemetry.goal_xy)
// ───────────────────────────────────────────────────────────────────────────

function GoalMarker({ sampler }: { sampler: HeightSampler | null }) {
  const goal_xy = useAppStore((s) => s.telemetry?.goal_xy ?? null);
  if (!goal_xy) return null;
  const [gx, gy] = goal_xy;
  const elev = sampler ? sampleHeight(sampler, { x: gx, y: gy }) : 0;
  const w = mapToWorld({ x: gx, y: gy, z: elev });
  return (
    <group position={[w.x, w.y, w.z]}>
      <mesh position={[0, 6, 0]}>
        <cylinderGeometry args={[0.5, 0.5, 12, 12]} />
        <meshStandardMaterial color="#f59e0b" emissive="#f59e0b"
          emissiveIntensity={0.4} transparent opacity={0.55} />
      </mesh>
    </group>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// Camera controller — switches between chase / topdown / orbit / driver
// ───────────────────────────────────────────────────────────────────────────

function ChaseCameraController({ mode, robotPose }:
                                { mode: string; robotPose: PoseW }) {
  const { camera } = useThree();
  const current = useRef<CameraPose>({
    position: { x: -12, y: 6, z: 12 },
    lookAt: { x: 0, y: 0, z: 0 },
  });

  useFrame(() => {
    if (mode === "orbit") return; // OrbitControls owns the camera
    let target: CameraPose;
    if (mode === "topdown") {
      target = topDownPoseFor(robotPose, 45);
    } else if (mode === "driver") {
      target = chasePoseFor(robotPose, DRIVER_DEFAULTS);
    } else {
      target = chasePoseFor(robotPose, CHASE_DEFAULTS);
    }
    current.current = smoothPose(
      current.current, target,
      mode === "driver" ? DRIVER_DEFAULTS.smooth_alpha
                        : CHASE_DEFAULTS.smooth_alpha,
    );
    camera.position.set(
      current.current.position.x,
      current.current.position.y,
      current.current.position.z,
    );
    camera.lookAt(
      current.current.lookAt.x,
      current.current.lookAt.y,
      current.current.lookAt.z,
    );
  });

  // Orbit mode: hand camera over to OrbitControls.
  if (mode === "orbit") {
    return <OrbitControls makeDefault target={[robotPose.x, robotPose.y, robotPose.z]} />;
  }
  return null;
}
