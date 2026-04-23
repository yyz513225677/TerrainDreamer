"""
PerformanceMonitor — classify the training state from training_log.csv.

Reads the CSV written by scripts/train_dreamer_auto.py and returns a
Diagnosis describing the current state of the run plus the raw rolling
statistics it was built from. The AutoTuner consumes the diagnosis to
decide whether to adjust LR, reload a checkpoint, shrink the curriculum,
etc.

State machine (see README):
    healthy | warming_up | plateau | regression | flip_storm |
    critic_collapse | wm_diverging

Design notes:
  * Stateful only via the CSV — so it's trivial to inspect / unit-test and
    survives process restarts.
  * Never raises on short logs; missing fields just degrade to `healthy`.
  * Numeric helpers are pure functions — no torch dep, no RL dep.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STATES = [
    "warming_up",
    "healthy",
    "plateau",
    "regression",
    "flip_storm",
    "critic_collapse",
    "wm_diverging",
]


@dataclass
class Diagnosis:
    state: str
    window: int                  # how many GOING rows this used
    success_rate: float
    flip_rate: float
    best_success_rate: float     # highest rolling succ we've ever seen
    best_mission: int            # mission where that peak occurred
    wm_trend: float              # +ve = loss rising
    imagined_return_abs: float   # |avg imagined return|
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"[diag] state={self.state}  "
                f"succ={self.success_rate:.2f} (best {self.best_success_rate:.2f} "
                f"@m{self.best_mission})  "
                f"flip={self.flip_rate:.2f}  "
                f"wmΔ={self.wm_trend:+.3f}  "
                f"|ret̂|={self.imagined_return_abs:.2f}  "
                f"— {'; '.join(self.reasons) if self.reasons else 'ok'}")


# ---------------------------------------------------------------------------
# CSV loader — tolerant to missing columns
# ---------------------------------------------------------------------------
def _read_going_rows(log_path: Path) -> List[Dict[str, str]]:
    """Return only the 'going' rows from the CSV (return rows are diagnostic-only)."""
    if not log_path.exists():
        return []
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("phase") == "going"]


def _maybe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core diagnose()
# ---------------------------------------------------------------------------
def diagnose(
    log_path: Path,
    *,
    window: int = 20,
    warmup_missions: int = 10,
    plateau_delta: float = 0.05,
    plateau_windows: int = 3,
    regression_drop: float = 0.20,
    flip_rate_thresh: float = 0.5,
    critic_collapse_thresh: float = 0.1,
    critic_collapse_windows: int = 3,
) -> Diagnosis:
    """Read the log and classify. Always returns a Diagnosis (never raises)."""
    rows = _read_going_rows(log_path)
    diag = Diagnosis(
        state="warming_up", window=0,
        success_rate=0.0, flip_rate=0.0,
        best_success_rate=0.0, best_mission=0,
        wm_trend=0.0, imagined_return_abs=0.0,
    )

    if len(rows) < warmup_missions:
        diag.reasons.append(f"only {len(rows)} GOING rows < warmup {warmup_missions}")
        return diag

    # --- rolling windows ---
    tail = rows[-window:]
    prev = rows[-2 * window:-window] if len(rows) >= 2 * window else []
    n = len(tail)
    succ = [int(r.get("reached", "0") or 0) for r in tail]
    flips = [int(r.get("flipped", "0") or 0) for r in tail]
    diag.window = n
    diag.success_rate = sum(succ) / n
    diag.flip_rate    = sum(flips) / n

    # --- best-success tracking (over the *whole* log) ---
    # Compute rolling succ @ window over the entire log to find the peak.
    reach_seq = [int(r.get("reached", "0") or 0) for r in rows]
    best_sr, best_mission = 0.0, 0
    for i in range(window - 1, len(reach_seq)):
        sr_i = sum(reach_seq[i - window + 1:i + 1]) / window
        if sr_i > best_sr:
            best_sr = sr_i
            # row mission column (int) — fallback to i+1 if parseable fails
            try:
                best_mission = int(rows[i].get("mission", i + 1))
            except (TypeError, ValueError):
                best_mission = i + 1
    diag.best_success_rate = best_sr
    diag.best_mission = best_mission

    # --- world-model loss trend ---
    wm_losses = [_maybe_float(r.get("wm_total")) for r in rows[-3 * window:]]
    wm_losses = [v for v in wm_losses if v is not None]
    if len(wm_losses) >= 2 * window:
        a = sum(wm_losses[:window]) / window
        b = sum(wm_losses[-window:]) / window
        diag.wm_trend = b - a

    # --- imagined return magnitude ---
    imag = [_maybe_float(r.get("imagined_return")) for r in tail]
    imag = [abs(v) for v in imag if v is not None]
    diag.imagined_return_abs = sum(imag) / len(imag) if imag else 0.0

    # ======================================================================
    # Classification — order matters (first match wins on severity)
    # ======================================================================

    # 1. flip storm — physical issue, deal with it first.
    if diag.flip_rate > flip_rate_thresh:
        diag.state = "flip_storm"
        diag.reasons.append(f"flip_rate={diag.flip_rate:.2f} > {flip_rate_thresh}")
        return diag

    # 2. regression — succ clearly dropped from its peak
    if best_sr > 0.3 and (best_sr - diag.success_rate) > regression_drop:
        diag.state = "regression"
        diag.reasons.append(
            f"succ dropped {best_sr:.2f} (m{best_mission}) → {diag.success_rate:.2f}"
        )
        return diag

    # 3. wm diverging — loss trending up strongly
    if diag.wm_trend > 0.2 and len(wm_losses) >= 3 * window:
        diag.state = "wm_diverging"
        diag.reasons.append(f"wm_total Δ={diag.wm_trend:+.3f} > 0.2")
        return diag

    # 4. critic collapse — |imagined_return| stays tiny
    if imag and diag.imagined_return_abs < critic_collapse_thresh:
        diag.state = "critic_collapse"
        diag.reasons.append(
            f"|imag_ret|={diag.imagined_return_abs:.3f} < {critic_collapse_thresh}"
        )
        return diag

    # 5. plateau — success_rate flat for several windows
    if len(rows) >= (plateau_windows + 1) * window:
        window_srs = []
        for w in range(plateau_windows + 1):
            lo = len(reach_seq) - (w + 1) * window
            hi = lo + window
            if lo < 0:
                break
            window_srs.append(sum(reach_seq[lo:hi]) / window)
        if len(window_srs) >= plateau_windows + 1:
            spread = max(window_srs) - min(window_srs)
            if spread < plateau_delta and diag.success_rate < 0.9:
                diag.state = "plateau"
                diag.reasons.append(
                    f"succ flat: {['%.2f' % s for s in window_srs]} spread={spread:.2f}"
                )
                return diag

    # 6. healthy — default
    diag.state = "healthy"
    return diag


# ---------------------------------------------------------------------------
# CLI: inspect a log manually
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="path to training_log.csv")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    d = diagnose(args.log, window=args.window)
    print(d)
