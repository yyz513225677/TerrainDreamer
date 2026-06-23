interface MetricRowProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  precision?: number;
  accent?: string;
  hint?: string;
  width?: "auto" | "narrow";
}

function fmt(v: string | number | null | undefined, precision: number) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(precision);
}

export function MetricRow({ label, value, unit, precision = 2, accent, hint, width = "auto" }: MetricRowProps) {
  return (
    <div
      className={`flex items-baseline justify-between gap-3 px-3 py-1 border-b border-border-subtle/60 last:border-b-0 ${width === "narrow" ? "" : ""}`}
      title={hint}
    >
      <span className="metric-label">{label}</span>
      <span className="metric-num" style={accent ? { color: accent } : undefined}>
        {fmt(value, precision)}
        {unit && (
          <span className="ml-1 text-text-muted text-label">{unit}</span>
        )}
      </span>
    </div>
  );
}
