interface StatusPillProps {
  label: string;
  state: "ok" | "warn" | "danger" | "idle" | "info";
  blink?: boolean;
  title?: string;
}

const PALETTE: Record<StatusPillProps["state"], { fg: string; border: string; dot: string }> = {
  ok: { fg: "text-accent-ok", border: "border-accent-ok/60", dot: "bg-accent-ok" },
  warn: { fg: "text-accent-warning", border: "border-accent-warning/60", dot: "bg-accent-warning" },
  danger: { fg: "text-accent-danger", border: "border-accent-danger/60", dot: "bg-accent-danger" },
  idle: { fg: "text-text-muted", border: "border-border-subtle", dot: "bg-text-muted" },
  info: { fg: "text-accent-dreamer", border: "border-accent-dreamer/60", dot: "bg-accent-dreamer" },
};

export function StatusPill({ label, state, blink, title }: StatusPillProps) {
  const c = PALETTE[state];
  return (
    <span className={`pill inline-flex items-center gap-1.5 ${c.fg} ${c.border}`} title={title}>
      <span
        className={`inline-block w-1.5 h-1.5 rounded-full ${c.dot} ${blink ? "animate-pulse" : ""}`}
        aria-hidden
      />
      <span className="uppercase tracking-wider">{label}</span>
    </span>
  );
}
