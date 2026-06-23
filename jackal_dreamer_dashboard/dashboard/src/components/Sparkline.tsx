interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  stroke?: string;
  fill?: string;
  /** If provided, clamps the y-range; otherwise auto-scales */
  yMin?: number;
  yMax?: number;
}

export function Sparkline({
  data,
  width = 120,
  height = 24,
  stroke = "#22d3ee",
  fill = "none",
  yMin,
  yMax,
}: SparklineProps) {
  if (data.length < 2) {
    return (
      <svg width={width} height={height} className="block">
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="#252b33"
          strokeDasharray="2 2"
        />
      </svg>
    );
  }
  const lo = yMin ?? Math.min(...data);
  const hi = yMax ?? Math.max(...data);
  const span = hi - lo === 0 ? 1 : hi - lo;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * (width - 2) + 1;
    const y = height - 1 - ((v - lo) / span) * (height - 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const d = `M ${pts.join(" L ")}`;
  const last = data[data.length - 1];
  const lastX = width - 1;
  const lastY = height - 1 - ((last - lo) / span) * (height - 2);
  return (
    <svg width={width} height={height} className="block">
      <path d={d} stroke={stroke} strokeWidth={1} fill={fill} />
      <circle cx={lastX} cy={lastY} r={1.6} fill={stroke} />
    </svg>
  );
}
