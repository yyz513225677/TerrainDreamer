"""Snapshot the current training run's metadata for reproducibility.

Called at the start of each training run by ``clean_and_train.sh`` (and
optionally by ``auto_loop.sh``). Writes to
``experiments/crater/iter_<N>/run_metadata.json`` containing:

* timestamp (UTC + local)
* git commit hash + dirty flag
* python + pip env capture
* env vars relevant to training (HEADLESS, TERRAIN, BASE_POLICY_MODE,
  BC_WEIGHT, DREAMER_ACTOR_WEIGHT, TIMEOUT_PENALTY, SUB_R_MAX)
* CUDA / GPU info
* Config snapshot via Config().asdict()
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT)] + list(args),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return "no nvidia-smi"


def _cfg_to_dict(cfg):
    """Recursive dataclass → dict for Config."""
    if is_dataclass(cfg):
        return {k: _cfg_to_dict(v) for k, v in asdict(cfg).items()}
    if isinstance(cfg, (list, tuple)):
        return [_cfg_to_dict(v) for v in cfg]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True,
                    help="Where to write run_metadata.json")
    args = ap.parse_args()

    out_dir = Path(args.iter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "hostname": os.uname().nodename,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "is_dirty": bool(_git("status", "--porcelain")),
            "n_changed_files":
                len(_git("status", "--porcelain").splitlines()),
        },
        "python": sys.version,
        "gpu": _gpu_info(),
        "env": {
            k: os.environ.get(k, "")
            for k in [
                "HEADLESS", "TERRAIN", "BASE_POLICY_MODE",
                "BC_WEIGHT", "DREAMER_ACTOR_WEIGHT", "RECOVERY_BC_WEIGHT",
                "TIMEOUT_PENALTY", "SUB_R_MAX", "GZ_HEADLESS", "CUDA_VISIBLE_DEVICES",
            ]
        },
    }

    # Snapshot full Config() — BUT apply the same env-var overrides the
    # driver would, so the recorded config reflects what will actually run.
    try:
        from crater import Config
        import math
        cfg = Config()
        # Mirror the env-var overrides from scripts/train_crater_ros.py.
        if "BC_WEIGHT" in os.environ:
            cfg.bc.behavior_cloning_weight = float(os.environ["BC_WEIGHT"])
        if "RECOVERY_BC_WEIGHT" in os.environ:
            cfg.bc.recovery_behavior_cloning_weight = float(os.environ["RECOVERY_BC_WEIGHT"])
        if "DREAMER_ACTOR_WEIGHT" in os.environ:
            cfg.bc.dreamer_actor_weight = float(os.environ["DREAMER_ACTOR_WEIGHT"])
        if "TIMEOUT_PENALTY" in os.environ:
            cfg.reward.timeout_penalty = float(os.environ["TIMEOUT_PENALTY"])
        if "SUB_R_MAX" in os.environ:
            cfg.model.action_high = (math.pi, float(os.environ["SUB_R_MAX"]))
        if os.environ.get("USE_HIERARCHICAL_ACTION", "1") == "0":
            cfg.model.use_hierarchical_action = False
            cfg.model.action_low = (0.0, -1.0)
            cfg.model.action_high = (0.8, 1.0)
        metadata["config"] = _cfg_to_dict(cfg)
    except Exception as exc:
        metadata["config_load_error"] = str(exc)

    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"snapshot → {out_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
