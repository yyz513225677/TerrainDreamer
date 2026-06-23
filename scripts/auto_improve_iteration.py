"""TerrainDreamer auto-improvement driver — snapshots current 200-ep run and
decides the next iteration's hyperparameters.

Run after each 200-episode block:

  python3 scripts/auto_improve_iteration.py

What it does:

1. Computes metrics from /tmp/dreamer_train.log (success rate, fail-mode
   breakdown, last-window losses, return scale).
2. Loads current env-var hyperparameters.
3. Snapshots both into ``experiments/crater/iter_<N>/`` together with a
   copy of the training log and per-iter config.
4. Compares with the previous iteration's metrics (if any).
5. Picks the next mutation (env-var overrides) according to a small set
   of rules that take the trend into account.
6. Writes ``experiments/crater/next_env.sh`` — a shell file the launcher
   can ``source`` to apply the mutation before restarting the trainer.

The orchestrator only changes ONE thing per iteration so we can attribute
the next regression/improvement to the specific change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "crater"
LOG_PATH = Path("/tmp/dreamer_train.log")
ENV_OUT = EXPERIMENTS_DIR / "next_env.sh"

ENV_KEYS = [
    "BASE_POLICY_MODE",       # simple | reactive | memory
    "BC_WEIGHT",              # default 2.0
    "RECOVERY_BC_WEIGHT",     # default 4.0
    "DREAMER_ACTOR_WEIGHT",   # default 0.3
    "TIMEOUT_PENALTY",        # default -5.0
    "SUB_R_MAX",              # default 10.0
    "TERRAIN",                # default | rugged
]

EP_RE = re.compile(r"\[(\d+)\.\d+\].*=== Episode (\d+):.*?dest=\(([-\d.]+), ([-\d.]+)\)")
SUCC_RE = re.compile(r"yellow→red")
LOSS_RE = re.compile(
    r"\[(\d+)\.\d+\].*?episodes=(\d+).*?"
    r"loss/world_total=([-\d.]+).*?"
    r"loss/actor_dreamer=([-\d.]+).*?"
    r"loss/critic=([-\d.]+).*?"
    r"stat/imag_return_mean=([-\d.]+)"
)


def parse_log(path: Path):
    if not path.exists():
        return [], []
    eps, losses = [], []
    last_ep, last_ts, last_dest = None, None, None
    success_set = set()
    with path.open() as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                if last_ep is not None:
                    eps.append((last_ep, last_ts, last_dest,
                                last_ep in success_set,
                                int(m.group(1)) - last_ts))
                last_ep = int(m.group(2))
                last_ts = int(m.group(1))
                last_dest = (float(m.group(3)), float(m.group(4)))
                continue
            if "yellow→red" in line and last_ep is not None:
                success_set.add(last_ep)
                continue
            lm = LOSS_RE.search(line)
            if lm:
                losses.append((
                    int(lm.group(2)), int(lm.group(1)),
                    float(lm.group(3)), float(lm.group(4)),
                    float(lm.group(5)), float(lm.group(6)),
                ))
    return eps, losses


def compute_metrics(eps, losses):
    if not eps:
        return None
    succ = [e for e in eps if e[3]]
    fail = [e for e in eps if not e[3]]
    n = len(eps)
    short = [e for e in fail if e[4] < 200]
    long_ = [e for e in fail if e[4] > 800]
    mid = [e for e in fail if 200 <= e[4] <= 800]
    last_quarter = losses[-max(1, len(losses) // 4):] if losses else []
    return {
        "n_episodes": n,
        "success_rate": 100 * len(succ) / n,
        "short_fail": len(short),
        "mid_fail": len(mid),
        "long_fail": len(long_),
        "success_mean_dist_m": mean(
            math.hypot(*e[2]) for e in succ) if succ else 0.0,
        "fail_mean_dist_m": mean(
            math.hypot(*e[2]) for e in fail) if fail else 0.0,
        "world_loss_mean":
            mean(L[2] for L in last_quarter) if last_quarter else 0.0,
        "critic_loss_mean":
            mean(L[4] for L in last_quarter) if last_quarter else 0.0,
        "critic_loss_max":
            max(L[4] for L in last_quarter) if last_quarter else 0.0,
        "imag_return_last": losses[-1][5] if losses else 0.0,
    }


def current_env():
    return {k: os.environ.get(k, "") for k in ENV_KEYS}


def next_iter_number():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name.split("_")[1]) for p in EXPERIMENTS_DIR.glob("iter_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()]
    return (max(existing) + 1) if existing else 0


def decide_mutation(prev_metrics, this_metrics, this_env):
    """Return a dict of env-var changes for the *next* iteration.

    Strategy: change exactly one thing. Pick based on the failure mode
    that dominates, or — if metrics regressed vs prev — revert the last
    change and try a different axis.
    """
    new_env = dict(this_env)
    mutation = "noop"

    # Defaults if blank.
    def f(k, default):
        v = this_env.get(k, "")
        try: return float(v) if v != "" else default
        except: return default

    bc_w = f("BC_WEIGHT", 2.0)
    daw = f("DREAMER_ACTOR_WEIGHT", 0.3)
    tp = f("TIMEOUT_PENALTY", -5.0)
    rmax = f("SUB_R_MAX", 10.0)
    base_mode = this_env.get("BASE_POLICY_MODE", "simple")

    # Regression check.
    regressed = (prev_metrics is not None
                 and this_metrics["success_rate"]
                       < prev_metrics["success_rate"] - 2.0)

    if regressed:
        # Roll back BASE_POLICY_MODE first if it was changed last; otherwise
        # cycle to a different mutation candidate.
        if base_mode == "memory":
            new_env["BASE_POLICY_MODE"] = "reactive"
            mutation = "revert BASE_POLICY_MODE memory → reactive"
        elif base_mode == "reactive":
            new_env["BASE_POLICY_MODE"] = "simple"
            mutation = "revert BASE_POLICY_MODE reactive → simple"
        else:
            new_env["BC_WEIGHT"] = str(max(1.0, bc_w - 0.5))
            mutation = f"reduce BC_WEIGHT {bc_w} → {new_env['BC_WEIGHT']}"
        return new_env, mutation

    # No regression — pick an improvement based on dominant failure mode.
    sr = this_metrics["success_rate"]
    short_f = this_metrics["short_fail"]
    mid_f = this_metrics["mid_fail"]
    long_f = this_metrics["long_fail"]
    crit_max = this_metrics["critic_loss_max"]
    ret_last = this_metrics["imag_return_last"]

    if crit_max > 5.0 or ret_last < -100:
        new_env["DREAMER_ACTOR_WEIGHT"] = f"{max(0.1, daw - 0.1):.2f}"
        mutation = (f"critic unstable (max={crit_max:.1f}, ret={ret_last:.0f}) "
                    f"→ lower DREAMER_ACTOR_WEIGHT {daw} → {new_env['DREAMER_ACTOR_WEIGHT']}")
    elif short_f >= max(2, mid_f + long_f):
        # Tilt / stuck dominates → try better obstacle awareness.
        if base_mode == "simple":
            new_env["BASE_POLICY_MODE"] = "reactive"
            mutation = "short failures dominate → BASE_POLICY_MODE simple → reactive"
        elif base_mode == "reactive":
            new_env["BASE_POLICY_MODE"] = "memory"
            mutation = "short failures dominate → BASE_POLICY_MODE reactive → memory"
        else:
            new_env["BC_WEIGHT"] = f"{bc_w + 0.5:.1f}"
            mutation = f"short failures persist → bump BC_WEIGHT {bc_w} → {new_env['BC_WEIGHT']}"
    elif (mid_f + long_f) >= max(2, short_f):
        # Timeouts / mid-fails dominate → actor can't reach goal.
        new_env["BC_WEIGHT"] = f"{bc_w + 0.5:.1f}"
        mutation = f"timeouts dominate → bump BC_WEIGHT {bc_w} → {new_env['BC_WEIGHT']}"
    elif sr > 95:
        # Already very good; widen sub-goal horizon to try harder distances.
        new_env["SUB_R_MAX"] = f"{min(15.0, rmax + 1.0):.1f}"
        mutation = f"high success → widen SUB_R_MAX {rmax} → {new_env['SUB_R_MAX']}"
    else:
        # Default: cycle BASE_POLICY_MODE.
        next_mode = {"simple": "reactive", "reactive": "memory",
                     "memory": "simple"}[base_mode]
        new_env["BASE_POLICY_MODE"] = next_mode
        mutation = f"cycle BASE_POLICY_MODE {base_mode} → {next_mode}"

    return new_env, mutation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(LOG_PATH))
    args = ap.parse_args()

    log_path = Path(args.log)
    eps, losses = parse_log(log_path)
    metrics = compute_metrics(eps, losses)
    if metrics is None:
        print("No episodes — nothing to do.")
        return

    iter_n = next_iter_number()
    iter_dir = EXPERIMENTS_DIR / f"iter_{iter_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    env_now = current_env()
    (iter_dir / "config.json").write_text(json.dumps(env_now, indent=2))
    (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        shutil.copy(log_path, iter_dir / "dreamer_train.log")
    except Exception:
        pass

    # Find previous iter's metrics for comparison.
    prev_metrics = None
    if iter_n > 0:
        prev_mp = EXPERIMENTS_DIR / f"iter_{iter_n - 1}" / "metrics.json"
        if prev_mp.exists():
            prev_metrics = json.loads(prev_mp.read_text())

    new_env, mutation = decide_mutation(prev_metrics, metrics, env_now)
    (iter_dir / "mutation.txt").write_text(mutation + "\n")

    # Write next_env.sh
    lines = ["#!/usr/bin/env bash", "# Generated by auto_improve_iteration.py"]
    for k in ENV_KEYS:
        v = new_env.get(k, "")
        if v != "":
            lines.append(f'export {k}="{v}"')
    ENV_OUT.write_text("\n".join(lines) + "\n")
    ENV_OUT.chmod(0o755)

    # Report
    print(f"=== Iteration {iter_n} complete ===")
    print(f"Episodes        : {metrics['n_episodes']}")
    print(f"Success rate    : {metrics['success_rate']:.1f}%")
    print(f"Failures        : short={metrics['short_fail']}, "
          f"mid={metrics['mid_fail']}, long={metrics['long_fail']}")
    print(f"Critic loss     : mean={metrics['critic_loss_mean']:.3f}, "
          f"max={metrics['critic_loss_max']:.3f}")
    print(f"Imag return     : last={metrics['imag_return_last']:.1f}")
    if prev_metrics is not None:
        d = metrics["success_rate"] - prev_metrics["success_rate"]
        print(f"Δ vs iter {iter_n-1}     : {d:+.1f} pp success rate")
    print(f"This config     : {env_now}")
    print(f"Next mutation   : {mutation}")
    print(f"Next env file   : {ENV_OUT}")


if __name__ == "__main__":
    main()
