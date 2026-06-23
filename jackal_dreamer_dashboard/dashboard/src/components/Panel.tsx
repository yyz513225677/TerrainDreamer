import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  children: ReactNode;
  accent?: string;
  right?: ReactNode;
  className?: string;
}

export function Panel({ title, children, accent, right, className }: PanelProps) {
  return (
    <section className={`panel flex flex-col ${className ?? ""}`}>
      <header className="flex items-center justify-between px-3 py-1.5 border-b border-border-subtle bg-bg-raised/40">
        <div className="flex items-center gap-2">
          {accent && (
            <span
              className="inline-block w-1.5 h-3"
              style={{ backgroundColor: accent }}
              aria-hidden
            />
          )}
          <h2 className="panel-header">{title}</h2>
        </div>
        {right}
      </header>
      <div className="flex-1 min-h-0">{children}</div>
    </section>
  );
}
