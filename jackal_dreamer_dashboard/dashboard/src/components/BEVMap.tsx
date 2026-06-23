import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../lib/modeFSM";
import { Panel } from "./Panel";

interface View {
  // pixels per meter
  scale: number;
  // canvas-px offset applied to robot-centered coords
  panX: number;
  panY: number;
  autoCenter: boolean;
}

const ROBOT_LEN_M = 0.508; // jackal length
const ROBOT_WID_M = 0.43; // jackal width

export function BEVMap() {
  const t = useAppStore((s) => s.telemetry);
  const lidar = useAppStore((s) => s.lidar);
  const dreamer = useAppStore((s) => s.dreamer);
  const trail = useAppStore((s) => s.pathTrail);
  const markers = useAppStore((s) => s.collisionMarkers);
  const routes = useAppStore((s) => s.routes);
  const replay = useAppStore((s) => s.replay);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [view, setView] = useState<View>({ scale: 18, panX: 0, panY: 0, autoCenter: true });
  const draggingRef = useRef<{ x: number; y: number } | null>(null);

  // Pan + zoom
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      setView((v) => ({ ...v, scale: Math.max(4, Math.min(120, v.scale * factor)) }));
    };
    const onDown = (e: MouseEvent) => {
      draggingRef.current = { x: e.clientX, y: e.clientY };
    };
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const dx = e.clientX - draggingRef.current.x;
      const dy = e.clientY - draggingRef.current.y;
      draggingRef.current = { x: e.clientX, y: e.clientY };
      setView((v) => ({ ...v, panX: v.panX + dx, panY: v.panY + dy, autoCenter: false }));
    };
    const onUp = () => {
      draggingRef.current = null;
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    c.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      c.removeEventListener("wheel", onWheel);
      c.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  // Resize canvas to wrapper
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = wrap.clientWidth * dpr;
      canvas.height = wrap.clientHeight * dpr;
      canvas.style.width = `${wrap.clientWidth}px`;
      canvas.style.height = `${wrap.clientHeight}px`;
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let raf = 0;

    function draw() {
      const c = canvasRef.current;
      if (!c) return;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const W = c.width;
      const H = c.height;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#0b0d10";
      ctx.fillRect(0, 0, W, H);

      const cx = W / 2 + view.panX * dpr;
      const cy = H / 2 + view.panY * dpr;
      const s = view.scale * dpr; // px per meter
      const rx = t?.pose.x ?? 0;
      const ry = t?.pose.y ?? 0;
      const ryaw = t?.pose.yaw ?? 0;

      const world = (x: number, y: number): [number, number] => [
        cx + (x - rx) * s,
        cy - (y - ry) * s,
      ];

      // Range rings
      ctx.strokeStyle = "#252b33";
      ctx.lineWidth = 1;
      for (const r of [5, 10, 20]) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * s, 0, Math.PI * 2);
        ctx.stroke();
      }
      // Range labels
      ctx.fillStyle = "#7a8590";
      ctx.font = `${10 * dpr}px JetBrains Mono, monospace`;
      ctx.textAlign = "left";
      for (const r of [5, 10, 20]) {
        ctx.fillText(`${r}m`, cx + r * s + 4 * dpr, cy - 2 * dpr);
      }

      // Origin crosshairs
      ctx.strokeStyle = "#1a1f25";
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(W, cy);
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, H);
      ctx.stroke();

      // Expert routes (amber)
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = "#f59e0b";
      for (const route of routes) {
        if (!route.path) continue;
        ctx.beginPath();
        route.path.forEach(([x, y], i) => {
          const [px, py] = world(x, y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Replay route (highlighted if active)
      if (replay.active && replay.routeId) {
        const r = routes.find((rr) => rr.route_id === replay.routeId);
        if (r?.path) {
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 3 * dpr;
          ctx.setLineDash([6 * dpr, 4 * dpr]);
          ctx.beginPath();
          r.path.forEach(([x, y], i) => {
            const [px, py] = world(x, y);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Dreamer planned route (cyan/blue)
      if (dreamer?.planned_route?.length) {
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        dreamer.planned_route.forEach(([x, y], i) => {
          const [px, py] = world(x, y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Trail (current trajectory)
      if (trail.length > 1) {
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 1.5 * dpr;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        trail.forEach(([x, y], i) => {
          const [px, py] = world(x, y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // LiDAR scatter (cyan)
      if (lidar?.ranges?.length) {
        ctx.fillStyle = "#22d3ee";
        const aMin = lidar.angle_min ?? 0;
        const aInc = lidar.angle_increment ?? (Math.PI * 2) / lidar.ranges.length;
        for (let i = 0; i < lidar.ranges.length; i++) {
          const r = lidar.ranges[i];
          if (!Number.isFinite(r) || r <= 0 || r > 30) continue;
          const a = aMin + i * aInc + ryaw;
          const wx = rx + r * Math.cos(a);
          const wy = ry + r * Math.sin(a);
          const [px, py] = world(wx, wy);
          ctx.fillRect(px - 0.5 * dpr, py - 0.5 * dpr, 1.5 * dpr, 1.5 * dpr);
        }
      }

      // Collision markers (red)
      ctx.fillStyle = "#ef4444";
      for (const [x, y] of markers) {
        const [px, py] = world(x, y);
        ctx.beginPath();
        ctx.arc(px, py, 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Goal flag
      if (t?.goal_xy) {
        const [gx, gy] = world(t.goal_xy[0], t.goal_xy[1]);
        ctx.fillStyle = "#10b981";
        ctx.beginPath();
        ctx.arc(gx, gy, 5 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(gx, gy - 14 * dpr);
        ctx.lineTo(gx + 8 * dpr, gy - 11 * dpr);
        ctx.lineTo(gx, gy - 8 * dpr);
        ctx.stroke();
      }

      // Robot footprint
      const [bx, by] = world(rx, ry);
      ctx.save();
      ctx.translate(bx, by);
      ctx.rotate(-ryaw);
      const w = ROBOT_WID_M * s;
      const l = ROBOT_LEN_M * s;
      ctx.fillStyle = "rgba(34, 211, 238, 0.18)";
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.rect(-l / 2, -w / 2, l, w);
      ctx.fill();
      ctx.stroke();
      // heading arrow
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(l / 2 + 6 * dpr, 0);
      ctx.stroke();
      ctx.restore();

      raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [view, t, lidar, dreamer, trail, markers, routes, replay]);

  const recenter = () => setView((v) => ({ ...v, panX: 0, panY: 0, autoCenter: true }));

  return (
    <Panel
      title="BEV Map"
      accent="#22d3ee"
      className="h-full"
      right={
        <div className="flex items-center gap-2">
          <span className="metric-label">scale</span>
          <span className="font-mono text-label tabular-nums text-text-primary">
            {view.scale.toFixed(0)} px/m
          </span>
          <button className="btn" onClick={recenter} title="Recenter on robot">
            CENTER
          </button>
        </div>
      }
    >
      <div ref={wrapRef} className="w-full h-full relative">
        <canvas ref={canvasRef} className="absolute inset-0 cursor-grab active:cursor-grabbing" />
        <div className="absolute bottom-2 left-2 flex items-center gap-3 text-label text-text-muted pointer-events-none">
          <span><span className="inline-block w-2 h-2 mr-1 align-middle" style={{ backgroundColor: "#22d3ee" }} />LiDAR / trail</span>
          <span><span className="inline-block w-2 h-2 mr-1 align-middle" style={{ backgroundColor: "#f59e0b" }} />expert route</span>
          <span><span className="inline-block w-2 h-2 mr-1 align-middle" style={{ backgroundColor: "#3b82f6" }} />dreamer plan</span>
          <span><span className="inline-block w-2 h-2 mr-1 align-middle" style={{ backgroundColor: "#ef4444" }} />collision</span>
          <span><span className="inline-block w-2 h-2 mr-1 align-middle" style={{ backgroundColor: "#10b981" }} />goal</span>
        </div>
      </div>
    </Panel>
  );
}
