"""Auto-analysis script for /tmp/dreamer_train.log.

Reports a compact training-health summary suitable for end-of-block
checkpoints (e.g. every 100 episodes):

  - Episode success rate + failure mode breakdown
  - Goal-distance distribution (mean / median / max)
  - Average time-to-goal vs distance
  - World-model / actor-critic / BC loss trends + last-window stats
  - Actor-critic divergence signal (imag_return_mean magnitude)
  - Per-tunable improvement suggestions

Run:
  python3 scripts/analyze_training.py                    # all episodes
  python3 scripts/analyze_training.py --tail 100         # last 100 only
  python3 scripts/analyze_training.py --log /tmp/x.log
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


EP_RE = re.compile(r"\[(\d+)\.\d+\].*=== Episode (\d+):.*?dest=\(([-\d.]+), ([-\d.]+)\)")
SUCC_RE = re.compile(r"\[(\d+)\.\d+\].*yellow→red")
LOSS_RE = re.compile(
    r"\[(\d+)\.\d+\].*?episodes=(\d+).*?"
    r"loss/world_total=([-\d.]+).*?"
    r"loss/actor_dreamer=([-\d.]+).*?"
    r"loss/critic=([-\d.]+).*?"
    r"stat/imag_return_mean=([-\d.]+).*?"
    r"loss/bc=([-\d.]+)"
)


def parse_log(path: Path, last_n: int | None):
    eps = []              # list of (ep_idx, ts_start, dest_xy, success_bool, duration_s)
    losses = []           # list of (ep_idx, ts, world, actor, critic, return, bc)
    last_ep = None
    last_ts = None
    last_dest = None
    success_set = set()

    with path.open() as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                # close previous
                if last_ep is not None:
                    eps.append((last_ep, last_ts, last_dest, last_ep in success_set,
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
                    int(lm.group(2)),       # episode
                    int(lm.group(1)),       # ts
                    float(lm.group(3)),
                    float(lm.group(4)),
                    float(lm.group(5)),
                    float(lm.group(6)),
                    float(lm.group(7)),
                ))

    # close last running ep (only if we have a "next" anchor; otherwise drop it).
    # The currently running episode (no terminal event yet) is excluded.
    # We do NOT add (last_ep, ...) here.

    if last_n is not None and len(eps) > last_n:
        eps = eps[-last_n:]

    return eps, losses


def fmt(x, n=2):
    return f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/dreamer_train.log")
    ap.add_argument("--tail", type=int, default=None,
                    help="Only analyse the last N completed episodes")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        sys.exit(1)

    eps, losses = parse_log(path, args.tail)
    n = len(eps)
    if n == 0:
        print("No completed episodes in log yet.")
        return

    succ = [e for e in eps if e[3]]
    fail = [e for e in eps if not e[3]]
    rate = 100 * len(succ) / n

    # Failure mode classification by duration:
    #   short (< 200 s)  → tilt / collision / stuck
    #   long  (> 800 s)  → timeout (couldn't reach goal)
    #   mid              → other (e.g. partial fail)
    short = [e for e in fail if e[4] < 200]
    long = [e for e in fail if e[4] > 800]
    mid = [e for e in fail if 200 <= e[4] <= 800]

    dists_succ = [math.hypot(*e[2]) for e in succ]
    dists_fail = [math.hypot(*e[2]) for e in fail]

    times_succ = [e[4] for e in succ]

    print("─" * 78)
    print(f"  Training analysis  ·  {n} completed episodes  ·  log = {path}")
    print("─" * 78)
    print(f"  Success rate     : {fmt(rate, 1)}%   ({len(succ)} / {n})")
    print(f"  Failure modes    : {len(short)} short (tilt/stuck), "
          f"{len(long)} long (timeout), {len(mid)} mid")
    if dists_succ:
        print(f"  Success distance : mean={fmt(mean(dists_succ))}m  "
              f"median={fmt(median(dists_succ))}m  "
              f"max={fmt(max(dists_succ))}m")
    if dists_fail:
        print(f"  Fail distance    : mean={fmt(mean(dists_fail))}m  "
              f"max={fmt(max(dists_fail))}m")
    if times_succ:
        print(f"  Success time     : mean={fmt(mean(times_succ),1)}s  "
              f"median={fmt(median(times_succ),1)}s  "
              f"max={fmt(max(times_succ),1)}s")
    if fail:
        print(f"  Failed episodes  : {sorted(e[0] for e in fail)[-10:]}"
              f"{' ...' if len(fail) > 10 else ''}")

    # Loss summary — last quarter window
    if losses:
        last_window = losses[-max(1, len(losses)//4):]
        wm = mean(L[2] for L in last_window)
        am = mean(L[3] for L in last_window)
        cm = mean(L[4] for L in last_window)
        rm = mean(L[5] for L in last_window)
        bm = mean(L[6] for L in last_window)
        print()
        print(f"  loss/world_total  last-window mean : {fmt(wm,3)}")
        print(f"  loss/actor_dreamer mean / |max|    : {fmt(am,2)} / "
              f"{fmt(max(abs(L[3]) for L in last_window),2)}")
        print(f"  loss/critic       mean / max       : {fmt(cm,3)} / "
              f"{fmt(max(L[4] for L in last_window),3)}")
        print(f"  imag_return_mean  last value       : {fmt(losses[-1][5],1)}")
        print(f"  loss/bc           last-window mean : {fmt(bm,3)}")

    # ── Suggestions ────────────────────────────────────────────────────
    sugg = []
    if rate < 70:
        sugg.append(
            "Success rate < 70 % — recent failure modes suggest the policy "
            "is not making it to the goal. Consider:")
        if len(short) > len(long):
            sugg.append("    • Lower BasePolicy v_max (currently 0.6) "
                        "to reduce tip-overs on rough terrain.")
            sugg.append("    • Reduce DEST_MAX (sample goals closer) until "
                        "success rate stabilises.")
        else:
            sugg.append("    • Most failures are timeouts — increase BC "
                        "weight in trainer so actor stays close to base "
                        "policy.")
            sugg.append("    • Lower lidar_penalty_weight further if "
                        "imag_return_mean ≪ 0.")

    if losses:
        last_return = losses[-1][5]
        last_critic_max = max(L[4] for L in losses[-50:])
        if last_return < -100:
            sugg.append(
                f"Critic is locked in a large negative regime "
                f"(imag_return_mean={last_return:.1f}). "
                f"Reward shaping is probably too heavy:")
            sugg.append("    • Halve lidar_penalty_weight / "
                        "tilt_penalty_weight.")
            sugg.append("    • Or raise lidar_penalty_threshold (less "
                        "frequent firing).")
        if last_critic_max > 5.0:
            sugg.append(
                "loss/critic spiking > 5 — consider critic LR ↓ "
                "(or enable return normalisation in trainer.py).")

    if not sugg:
        sugg.append("No major flags. Training appears healthy.")

    print()
    print("  Suggestions:")
    for s in sugg:
        print(f"    {s}")
    print("─" * 78)


if __name__ == "__main__":
    main()
