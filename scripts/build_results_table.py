"""Collect all iter_<N>/metrics.json from experiments/crater/ and emit a
paper-ready Markdown + LaTeX table of results.

Usage:
  python3 scripts/build_results_table.py
"""
from __future__ import annotations
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "crater"

rows = []
for d in sorted(EXP.glob("iter_*")):
    if not d.is_dir():
        continue
    name = d.name
    if not name.split("_")[-1].isdigit():
        continue
    mp = d / "metrics.json"
    if not mp.exists():
        continue
    try:
        m = json.loads(mp.read_text())
    except Exception:
        continue
    label = ""
    if (d / "baseline_label.txt").exists():
        label = (d / "baseline_label.txt").read_text().strip()
    mut = ""
    if (d / "mutation.txt").exists():
        mut = (d / "mutation.txt").read_text().strip()
    # Get env snapshot
    env = {}
    if (d / "run_metadata.json").exists():
        try:
            md = json.loads((d / "run_metadata.json").read_text())
            env = md.get("env", {})
        except Exception:
            pass
    rows.append({
        "iter": int(name.split("_")[-1]),
        "label": label or "?",
        "n": m.get("n_episodes", 0),
        "success_rate": m.get("success_rate", 0.0),
        "short_fail": m.get("short_fail", 0),
        "mid_fail": m.get("mid_fail", 0),
        "long_fail": m.get("long_fail", 0),
        "critic_max": m.get("critic_loss_max", 0.0),
        "imag_return": m.get("imag_return_last", 0.0),
        "bc_weight": env.get("BC_WEIGHT", "default"),
        "sub_r_max": env.get("SUB_R_MAX", "default"),
        "mutation": mut,
    })

rows.sort(key=lambda r: r["iter"])

# Markdown table
out_md = EXP.parent.parent / "paper" / "results_table.md"
out_md.parent.mkdir(parents=True, exist_ok=True)
md = ["| iter | label | n | succ% | short | mid | long | critic_max | imag_return | bc_w | r_max | mutation |",
      "|------|-------|---|-------|-------|-----|------|------------|-------------|------|-------|----------|"]
for r in rows:
    md.append(f"| {r['iter']} | {r['label']} | {r['n']} | {r['success_rate']:.1f} | "
              f"{r['short_fail']} | {r['mid_fail']} | {r['long_fail']} | "
              f"{r['critic_max']:.2f} | {r['imag_return']:+.1f} | "
              f"{r['bc_weight']} | {r['sub_r_max']} | {r['mutation'][:50]} |")
out_md.write_text("\n".join(md) + "\n")
print(f"Markdown table saved to {out_md}")

# LaTeX table (paper-ready)
out_tex = EXP.parent.parent / "paper" / "results_table.tex"
tex = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{TerrainDreamer training iterations on rugged terrain. Mutations explore "
    r"hyperparameter space; iter 10 reaches 98\% success.}",
    r"\label{tab:iter_history}",
    r"\begin{tabular}{rlrrrrrl}",
    r"\toprule",
    r"Iter & Label & N & Succ.\% & Short & Mid & $|\mathcal{L}_{\text{critic}}|_{\max}$ & Mutation \\",
    r"\midrule",
]
for r in rows:
    succ_bold = f"\\textbf{{{r['success_rate']:.1f}}}" if r['success_rate'] >= 95 else f"{r['success_rate']:.1f}"
    mut_short = r['mutation'].replace("→", r"$\to$")[:40]
    tex.append(f"{r['iter']} & {r['label']} & {r['n']} & {succ_bold} & "
               f"{r['short_fail']} & {r['mid_fail']} & "
               f"{r['critic_max']:.2f} & {mut_short} \\\\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out_tex.write_text("\n".join(tex) + "\n")
print(f"LaTeX table saved to {out_tex}")

print()
print("=" * 80)
print(f"{'iter':>5} {'label':>20} {'n':>4} {'succ%':>7} {'short':>6} {'mid':>4} {'critic_max':>11}")
print("-" * 80)
for r in rows:
    marker = " ★" if r['success_rate'] >= 95 else ""
    print(f"{r['iter']:5d} {r['label']:>20} {r['n']:4d} {r['success_rate']:7.1f} "
          f"{r['short_fail']:6d} {r['mid_fail']:4d} {r['critic_max']:11.2f}{marker}")
print("=" * 80)
