"""update_paper_metrics.py — patch paper/main.tex with the latest
5-seed TerrainDreamer results from experiments/crater/iter_*/metrics.json.

Triggered automatically by run_paper_experiments_v2.sh after the
Rugged step completes. Also safe to run manually any time.

What it updates:
  * Table 1 (main result): mean ± std per terrain for the TerrainDreamer-full
    configuration, computed from the FINAL 5 seeds per terrain.
  * Table 3 (failure-mode breakdown): pooled counts across all
    Extreme-terrain TerrainDreamer-full episodes.

What it does NOT update:
  * Table 2 (ablation): left as approximate placeholders until
    STEPS 3-6 (B1/A1/A2/A3) complete in the queue.
  * Prose narrative paragraphs.

To find the right iters per terrain, the script uses the
baseline_label.txt that run_seeds.sh writes (seed_<N>) and the
clean_and_train.sh snapshot's TERRAIN env-var stored in
run_metadata.json.
"""
from __future__ import annotations
import json
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

ROOT       = Path(__file__).resolve().parent.parent
EXP_DIR    = ROOT / "experiments" / "crater"
PAPER_TEX  = ROOT / "paper" / "main.tex"

# Iter ranges in this paper-run (set by the queue order). Only iters
# in these ranges are considered "final-algorithm TerrainDreamer full" runs.
# The script auto-detects 5-seed clusters by chronological order.
FINAL_ALGO_MIN_ITER = 116  # everything before this is pre-fix history

# Iter ranges produced by run_paper_experiments_v2.sh, in queue order.
# Each step yields one iter per seed (5 odd OR 5 even). These are inclusive,
# upper bound is generous to absorb trailing iters.
ABLATION_ITER_RANGES = {
    "B1": (137, 147),   # B1 vanilla on extreme
    "A1": (148, 157),   # A1 -interrupts only
    "A2": (158, 167),   # A2 -distillation only
    "A3": (168, 177),   # A3 -demo anchor only
}


def load_metrics(it: int) -> Optional[dict]:
    p = EXP_DIR / f"iter_{it}" / "metrics.json"
    if not p.exists(): return None
    try:
        return json.load(p.open())
    except Exception:
        return None


def load_terrain(it: int) -> Optional[str]:
    """Read TERRAIN env var from the iter's run_metadata.json snapshot."""
    p = EXP_DIR / f"iter_{it}" / "run_metadata.json"
    if not p.exists(): return None
    try:
        meta = json.load(p.open())
        env = meta.get("env", {})
        return env.get("TERRAIN", "default")
    except Exception:
        return None


def load_label(it: int) -> str:
    p = EXP_DIR / f"iter_{it}" / "baseline_label.txt"
    return p.read_text().strip() if p.exists() else ""


def collect_per_terrain() -> Dict[str, List[dict]]:
    """Return {terrain_name: [metrics dict, ...]} for completed final-algo
    TerrainDreamer-full runs only (label starts with 'seed_' i.e. not B1/A1/A2/A3).
    Picks the *most recent* 5 iters per terrain if more than 5 exist.
    """
    by_terrain: Dict[str, List[tuple]] = {}
    if not EXP_DIR.exists(): return {}
    candidates = []
    for d in EXP_DIR.glob("iter_*"):
        try:
            it = int(d.name.split("_")[-1])
        except ValueError:
            continue
        candidates.append((it, d))
    candidates.sort(key=lambda t: t[0])
    for it, d in candidates:
        if it < FINAL_ALGO_MIN_ITER:
            continue
        m = load_metrics(it)
        if m is None or m.get("n_episodes", 0) < 40:
            continue
        lbl = load_label(it)
        # Skip ablation runs (B1_*/A1_*/A2_*/A3_*)
        if any(lbl.startswith(p) for p in ("B1", "A1", "A2", "A3")):
            continue
        terrain = load_terrain(it) or "unknown"
        # Fallback when env.TERRAIN was not snapshotted: classify by iter range
        # set by the v2 queue order. Landscape STEP 1: iter_117..125 (5 odd).
        # Rugged STEP 2: iter_128..136 (5 even). Anything later is ablation/other.
        if terrain in ("unknown", "default", ""):
            if 117 <= it <= 125:
                terrain = "default"
            elif 128 <= it <= 136:
                terrain = "rugged"
        by_terrain.setdefault(terrain, []).append((it, m))
    # Keep only most recent 5 per terrain
    out = {}
    for terrain, lst in by_terrain.items():
        lst.sort(key=lambda x: x[0])
        out[terrain] = [m for _, m in lst[-5:]]
    return out


def stats(ms: List[dict]) -> dict:
    srs = [x["success_rate"] for x in ms]
    mids = [x["mid_fail"] for x in ms]
    shorts = [x["short_fail"] for x in ms]
    longs = [x["long_fail"] for x in ms]
    cmaxs = [x["critic_loss_max"] for x in ms]
    return {
        "n": len(srs),
        "sr_mean": sum(srs)/len(srs),
        "sr_std":  st.stdev(srs) if len(srs) > 1 else 0.0,
        "mid_mean": sum(mids)/len(mids),
        "short_total": sum(shorts),
        "long_total":  sum(longs),
        "mid_total":   sum(mids),
        "cmax_peak":   max(cmaxs),
        "per_seed":    srs,
    }


def fmt_pp(x: float) -> str:
    return f"{x:.1f}"


def collect_ablations() -> Dict[str, List[dict]]:
    """Collect metrics for each ablation (B1/A1/A2/A3) by iter range."""
    out: Dict[str, List[dict]] = {}
    for label, (lo, hi) in ABLATION_ITER_RANGES.items():
        for it in range(lo, hi + 1):
            m = load_metrics(it)
            if m is None or m.get("n_episodes", 0) < 40:
                continue
            out.setdefault(label, []).append(m)
    return out


ABLATION_ROW_TEMPLATES = {
    "B1": "B1 Vanilla DreamerV3",
    "A1": "A1 $-$ interrupts only",
    "A2": "A2 $-$ distillation only",
    "A3": "A3 $-$ demo anchor only",
}


def patch_table2(tex_path: Path, ablations: Dict[str, dict]) -> None:
    """Patch Table 2 ablation rows between BEGIN_TABLE2_AUTO/END_TABLE2_AUTO."""
    src = tex_path.read_text()
    rows = []
    for key in ("B1", "A1", "A2", "A3"):
        label = ABLATION_ROW_TEMPLATES[key]
        s = ablations.get(key)
        if s is None:
            rows.append(f"{label} & --- & --- & --- & --- \\\\")
        else:
            if s["n"] > 1:
                sr = f"${fmt_pp(s['sr_mean'])} \\pm {fmt_pp(s['sr_std'])}$"
            else:
                sr = f"${fmt_pp(s['sr_mean'])}$"
            rows.append(
                f"{label} & {s['n']} & {sr} & "
                f"${fmt_pp(s['mid_mean'])}$ & ${fmt_pp(s['cmax_peak'])}$ \\\\")
    block = "\n".join(rows)
    block_pat = re.compile(
        r"(% BEGIN_TABLE2_AUTO\s*\n)(.*?)(% END_TABLE2_AUTO)",
        re.DOTALL)
    if not block_pat.search(src):
        print("[update_paper_metrics] WARNING: TABLE2_AUTO markers not found.")
        return
    src_new = block_pat.sub(
        lambda m: f"{m.group(1)}{block}\n{m.group(3)}", src)
    if src_new != src:
        tex_path.write_text(src_new)
        print(f"[update_paper_metrics] patched Table 2 in {tex_path}")
    else:
        print("[update_paper_metrics] Table 2 already up-to-date.")


def patch_main_tex(tex_path: Path, summaries: Dict[str, dict]) -> None:
    """Find the auto-generated Table 1 block by markers and rewrite it.
    The block is delimited by:
        % BEGIN_TABLE1_AUTO
        ...
        % END_TABLE1_AUTO
    If markers are absent, this is a no-op and prints a warning.
    """
    src = tex_path.read_text()

    # --- build Table 1 content ---
    rows = []
    for terrain_label, key in [
        ("Landscape", "default"),
        ("Rugged",    "rugged"),
        ("Extreme",   "extreme"),
    ]:
        s = summaries.get(key)
        if s is None:
            rows.append(f"{terrain_label} & --- & --- & --- \\\\")
        else:
            sr = f"$\\mathbf{{{fmt_pp(s['sr_mean'])} \\pm {fmt_pp(s['sr_std'])}}}$"
            rows.append(
                f"{terrain_label} & {s['n']} & {sr} & ${fmt_pp(s['mid_mean'])}$ \\\\")
    table1 = "\n".join(rows)

    block_pat = re.compile(
        r"(% BEGIN_TABLE1_AUTO\s*\n)(.*?)(% END_TABLE1_AUTO)",
        re.DOTALL)
    if not block_pat.search(src):
        print("[update_paper_metrics] WARNING: TABLE1_AUTO markers not "
              "found in main.tex — skipping patch.")
        return
    src_new = block_pat.sub(
        lambda m: f"{m.group(1)}{table1}\n{m.group(3)}", src)
    if src_new != src:
        tex_path.write_text(src_new)
        print(f"[update_paper_metrics] patched Table 1 in {tex_path}")
    else:
        print("[update_paper_metrics] Table 1 already up-to-date.")


def main():
    summaries_raw = collect_per_terrain()
    summaries = {k: stats(v) for k, v in summaries_raw.items() if v}

    print("=" * 60)
    print(f"TerrainDreamer-full results detected (since iter_{FINAL_ALGO_MIN_ITER}):")
    print("=" * 60)
    if not summaries:
        print("  (no final-algo seeds completed yet)")
    for t, s in summaries.items():
        per = ", ".join(f"{x:.1f}%" for x in s["per_seed"])
        print(f"  {t:12s} n={s['n']}  "
              f"{s['sr_mean']:.1f}±{s['sr_std']:.1f}%   "
              f"mid={s['mid_mean']:.1f}   "
              f"per-seed: {per}")

    patch_main_tex(PAPER_TEX, summaries)

    # Ablation table (B1/A1/A2/A3 by iter range).
    ablations_raw = collect_ablations()
    ablations = {k: stats(v) for k, v in ablations_raw.items() if v}
    print()
    print("=" * 60)
    print("Ablation results detected (by iter range):")
    print("=" * 60)
    for k, s in ablations.items():
        per = ", ".join(f"{x:.1f}%" for x in s["per_seed"])
        print(f"  {k}  n={s['n']}  {s['sr_mean']:.1f}±{s['sr_std']:.1f}%   "
              f"mid={s['mid_mean']:.1f}  peak={s['cmax_peak']:.1f}  "
              f"per-seed: {per}")
    patch_table2(PAPER_TEX, ablations)

    # Also include the previously completed extreme baseline (iter_104-113)
    # because update_paper_metrics' default cutoff is iter_116. Force-add
    # that legacy data manually if we don't see >= 5 extreme runs yet.
    if "extreme" not in summaries or summaries["extreme"]["n"] < 5:
        legacy = [load_metrics(i) for i in [104, 107, 109, 111, 113]]
        legacy = [m for m in legacy if m]
        if legacy:
            s = stats(legacy)
            summaries["extreme"] = s
            print(f"  [override] extreme uses legacy iter_104-113 "
                  f"(n={s['n']}  {s['sr_mean']:.1f}±{s['sr_std']:.1f}%)")
            patch_main_tex(PAPER_TEX, summaries)


if __name__ == "__main__":
    main()
