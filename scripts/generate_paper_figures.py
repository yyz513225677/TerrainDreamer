"""Generate paper-ready figures from existing training data.

Outputs PNG figures to paper/figures/. Each figure uses the *latest*
final-algorithm seeds per terrain plus the B1 ablation, and uses
meaningful axis labels (seeds, episodes, terrain names — not iter
indices or array positions).

Final-algorithm seed → iter map (set by the v2 queue order):
    Landscape: iter_117, 119, 121, 123, 125     (seeds 42,1337,2024,7,31415)
    Rugged:    iter_128, 130, 132, 134, 136     (same 5 seeds)
    Extreme (TerrainDreamer-full, legacy):
               iter_104, 107, 109, 111, 113
    Extreme (B1 vanilla ablation):
               iter_138, 140, 142, 144, 146
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
FIGS = PROJECT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
HM_DIR = PROJECT / "lunar_south_pole_gazebo" / "data" / "heightmaps"
EXP_DIR = PROJECT / "experiments" / "crater"

SEEDS = [42, 1337, 2024, 7, 31415]

# Iter ranges per group, in seed order matching SEEDS
GROUPS = {
    "Landscape (TD-full)": [117, 119, 121, 123, 125],
    "Rugged (TD-full)":    [128, 130, 132, 134, 136],
    "Extreme (TD-full)":   [104, 107, 109, 111, 113],
    "Extreme (B1)":        [138, 140, 142, 144, 146],
}

GROUP_COLOR = {
    "Landscape (TD-full)": "#3a7bd5",
    "Rugged (TD-full)":    "#2ca56b",
    "Extreme (TD-full)":   "#d24",
    "Extreme (B1)":        "#888",
}


def load_metrics(it):
    p = EXP_DIR / f"iter_{it}" / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── 1. Terrain comparison ─────────────────────────────────────────────────
def figure_terrains():
    """Heightmaps with axes labelled in METRES (100 m × 100 m maps).
    Larger figsize + larger fonts for two-column figure* placement."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, name, label in zip(
        axes,
        ["landscape_heightmap.png", "rugged_heightmap.png",
         "extreme_heightmap.png"],
        ["Landscape (gentle)", "Rugged (medium)", "Extreme (hard)"],
    ):
        p = HM_DIR / name
        if not p.exists():
            ax.text(0.5, 0.5, f"missing: {name}", ha="center", va="center",
                    fontsize=14)
            continue
        img = np.array(Image.open(p))
        im = ax.imshow(img, cmap="turbo",
                       extent=[-50, 50, -50, 50], origin="lower")
        ax.set_title(label, fontsize=18, pad=8)
        ax.set_xlabel("x [m]", fontsize=15, labelpad=4)
        ax.set_ylabel("y [m]", fontsize=15, labelpad=4)
        ax.tick_params(axis="both", labelsize=13)
    fig.suptitle("Procedurally-generated lunar south-pole heightmaps "
                 "(100 m $\\times$ 100 m)",
                 fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGS / "terrains_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ── 2. Obstacle masks ─────────────────────────────────────────────────────
def figure_obstacle_masks():
    """Binary traversability masks for the two non-trivial terrains."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, name, label in zip(
        axes,
        ["rugged_obstacle_mask.png", "extreme_obstacle_mask.png"],
        ["Rugged (7.3\\% blocked)", "Extreme (22.0\\% blocked)"],
    ):
        p = HM_DIR / name
        if not p.exists():
            ax.text(0.5, 0.5, f"missing: {name}", ha="center", va="center",
                    fontsize=14)
            continue
        img = np.array(Image.open(p))
        ax.imshow(img, cmap="gray_r", extent=[-50, 50, -50, 50],
                  origin="lower", vmin=0, vmax=255)
        # Use raw % (no LaTeX escape) — matplotlib renders it literally.
        ax.set_title(label.replace("\\%", "%"), fontsize=18, pad=8)
        ax.set_xlabel("x [m]", fontsize=15, labelpad=4)
        ax.set_ylabel("y [m]", fontsize=15, labelpad=4)
        ax.tick_params(axis="both", labelsize=13)
    fig.suptitle("Binary traversability (obstacle) masks",
                 fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIGS / "obstacle_masks.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ── 3. Training curves over EPISODES (parsed from log) ────────────────────
LOSS_RE = re.compile(
    r"episodes=(\d+).*?loss/world_total=([-\d.]+).*?"
    r"loss/actor_dreamer=([-\d.]+).*?loss/critic=([-\d.]+).*?"
    r"stat/imag_return_mean=([-\d.]+)"
)


def _parse_log(path):
    eps, world, actor, critic, ret = [], [], [], [], []
    if not path.exists():
        return eps, world, actor, critic, ret
    with path.open() as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                eps.append(int(m.group(1)))
                world.append(float(m.group(2)))
                actor.append(float(m.group(3)))
                critic.append(float(m.group(4)))
                ret.append(float(m.group(5)))
    return eps, world, actor, critic, ret


def figure_training_curves():
    """X axis = episodes seen, not array index. Pick the most recent
    final-algorithm extreme iter (full config) for the showcase curves."""
    candidate_iters = GROUPS["Extreme (TD-full)"][::-1]
    e, world, actor, critic, ret = [], [], [], [], []
    chosen = None
    for it in candidate_iters:
        d = EXP_DIR / f"iter_{it}"
        log = d / "dreamer_train.log"
        if not log.exists():
            continue
        e, world, actor, critic, ret = _parse_log(log)
        if e:
            chosen = it
            break
    if not e:
        # Fall back to the most recent log anywhere
        all_iters = sorted([
            int(d.name.split("_")[-1])
            for d in EXP_DIR.glob("iter_*")
            if d.name.split("_")[-1].isdigit()
        ])
        for it in reversed(all_iters):
            d = EXP_DIR / f"iter_{it}"
            log = d / "dreamer_train.log"
            if not log.exists():
                continue
            e, world, actor, critic, ret = _parse_log(log)
            if e:
                chosen = it
                break
    if not e:
        print("no training log parseable")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    e = np.array(e)
    panels = [
        (axes[0], critic, "#d24",    "Critic loss", "Critic loss (log scale)", True),
        (axes[1], actor,  "#3a7bd5", "Actor dreamer loss",
         "Actor (imagined-return) loss", False),
        (axes[2], ret,    "#2ca56b",
         r"$\mathbb{E}[\,$imagined return$\,]$",
         "Imagined return (bounded by return norm.)", False),
    ]
    for ax, ys, color, ylabel, title, logy in panels:
        ax.plot(e, ys, color=color, lw=2.0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Episodes seen", fontsize=16, labelpad=6)
        ax.set_ylabel(ylabel, fontsize=16, labelpad=6)
        ax.set_title(title, fontsize=17, pad=8)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Training curves — extreme terrain, iter\\_{chosen}",
                 fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGS / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ── 4. Success per seed (grouped by terrain / config) ─────────────────────
def figure_success_per_seed():
    """Grouped bar chart: each seed (x) has bars for each terrain group.
    X axis is the seed ID (meaningful), not iter index.
    Layout: legend below the plot (no overlap with bars), large fonts."""
    group_succ = {}
    for g, iters in GROUPS.items():
        srs = []
        for it in iters:
            m = load_metrics(it)
            srs.append(m["success_rate"] if m else None)
        group_succ[g] = srs

    n_seeds = len(SEEDS)
    width = 0.20
    x = np.arange(n_seeds)
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (g, srs) in enumerate(group_succ.items()):
        ys = [s if s is not None else 0.0 for s in srs]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, ys, width, label=g,
                      color=GROUP_COLOR[g], alpha=0.9, edgecolor="black",
                      linewidth=0.4)
        # Value labels above each bar.
        for b, raw in zip(bars, srs):
            if raw is None:
                b.set_hatch("//"); continue
            ax.text(b.get_x() + b.get_width()/2.0, raw + 1.2,
                    f"{raw:.1f}", ha="center", va="bottom",
                    fontsize=11, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Success rate [%]", fontsize=17)
    ax.set_xlabel("Seed", fontsize=17, labelpad=10)
    ax.set_ylim(0, 115)
    ax.axhline(90, ls="--", color="gray", alpha=0.5, lw=1.2)
    ax.text(n_seeds - 0.5, 91.5, "90% threshold",
            ha="right", va="bottom", fontsize=12, color="#666")
    ax.set_title("Per-seed success rate — TerrainDreamer-full vs B1 baseline",
                 fontsize=18, pad=14)

    # External legend below the x-axis.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=15, frameon=False,
              handlelength=2.0, columnspacing=2.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = FIGS / "success_per_seed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ── 5. Failure breakdown (pooled per group) ───────────────────────────────
def figure_failure_breakdown():
    """Stacked bar — one stack per group, pooled across that group's seeds.
    X axis is the group (terrain / config), not iter index.
    Layout: legend placed below the plot so it never overlaps the bars,
    all text sized for two-column IEEE journal layout."""
    groups = list(GROUPS.keys())
    n_eps, short, mid, long_, succ_total = [], [], [], [], []
    for g in groups:
        s_short = s_mid = s_long = s_succ = s_n = 0
        for it in GROUPS[g]:
            m = load_metrics(it)
            if m is None:
                continue
            s_short += m["short_fail"]
            s_mid   += m["mid_fail"]
            s_long  += m["long_fail"]
            s_n     += m["n_episodes"]
            s_succ  += int(round(m["success_rate"]/100.0 * m["n_episodes"]))
        short.append(s_short); mid.append(s_mid); long_.append(s_long)
        n_eps.append(s_n); succ_total.append(s_succ)

    x = np.arange(len(groups))
    # Bigger canvas + roomy bottom margin for the external legend.
    fig, ax = plt.subplots(figsize=(13, 7))
    short = np.array(short); mid = np.array(mid); long_ = np.array(long_)
    succ = np.array(succ_total)
    bar_w = 0.55
    ax.bar(x, succ,  bar_w, label="success",            color="#2ca56b")
    ax.bar(x, short, bar_w, bottom=succ,                label="short (early tilt)",   color="#f1a14c")
    ax.bar(x, mid,   bar_w, bottom=succ+short,          label="mid (mission tilt)",   color="#3a7bd5")
    ax.bar(x, long_, bar_w, bottom=succ+short+mid,      label="long (timeout)",       color="#d24")

    # Two-line x labels — terrain on line 1, config on line 2.
    pretty_labels = [g.replace(" (", "\n(") for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)

    ax.set_ylabel("Episode count (5 seeds pooled)", fontsize=17)
    ax.set_xlabel("Terrain / configuration", fontsize=17, labelpad=10)
    ax.set_title("Pooled failure-mode breakdown — 5 seeds per group",
                 fontsize=18, pad=14)

    # n={n} labels — placed ABOVE the stack top with extra y-padding.
    top = (succ + short + mid + long_)
    headroom = top.max() * 0.10
    for xi, n, t in zip(x, n_eps, top):
        ax.text(xi, t + headroom * 0.25, f"n={n}",
                ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Headroom so labels + legend don't crowd the top of the chart.
    ax.set_ylim(0, top.max() + headroom * 1.4)
    ax.grid(axis="y", alpha=0.3)

    # External legend below the x-axis — never overlaps the bars.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=15, frameon=False,
              handlelength=2.0, columnspacing=2.0)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = FIGS / "failure_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    figure_terrains()
    figure_obstacle_masks()
    figure_training_curves()
    figure_success_per_seed()
    figure_failure_breakdown()
    print(f"\nAll figures saved to {FIGS}")
