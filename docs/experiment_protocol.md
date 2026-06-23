# TerrainDreamer Experiment Protocol

For paper-grade results, every experiment must follow this protocol.
The shorthand `clean_and_train.sh` script handles steps 1-4 automatically.

## Per-run protocol

1. **Cold start the sim** — `train_lunar.sh --restart-sim` clears all stale
   gz processes and avoids transport-layer degradation (>4-hour bug).
2. **Clear checkpoints + logs** — `clean_and_train.sh` removes
   `checkpoints_auto/crater/*.pt` and `/tmp/*.log`.
3. **Snapshot run metadata** — `scripts/snapshot_run.py` writes
   `experiments/crater/iter_<N>/run_metadata.json` with git sha, env
   vars, GPU info, full Config dump.
4. **Train for the configured number of episodes** (default 100).
5. **At episode N** stop trainer, copy `/tmp/dreamer_train.log` to the
   iter dir, and run `scripts/analyze_training.py` → `metrics.json`.

## Required runs for paper

For each (method, terrain, seed) cell — 5 seeds:

| Method     | Configurations                                           |
|------------|----------------------------------------------------------|
| TerrainDreamer     | full system (default)                                    |
| – C1       | drop demo-anchored regime: BC_WEIGHT=0, DREAMER_ACTOR_WEIGHT=1.0 |
| – C2       | drop hierarchical: use_hierarchical_action=false         |
| – C3       | drop memory channel: BASE_POLICY_MODE=reactive (no memory) |
| B1 (vanilla Dreamer) | trav off, BC off, hierarchical off            |
| B2 (PPO)   | drop world model entirely (separate script needed)       |
| B3 (A*+DWA)| classical, no learning                                   |
| B4 (BC only)| pure BC, no RL                                          |

For each terrain: `landscape`, `rugged`, `crater-field` (TODO), `visibility-stress` (TODO).

Total runs: 8 methods × 4 terrains × 5 seeds × 100 episodes = 16 000 episodes.
At ~3 min / episode this is ~800 GPU-hours. Plan for compute.

## Seeds

Use seeds `[42, 1337, 2024, 7, 31415]`. Pass via env var `SEED` to the
trainer.

## Metrics to record

* `success_rate` (%)
* `success_distance_mean` (m)
* `time_to_goal` (s)
* `path_efficiency` = `straight_line_distance / actual_distance`
* `tilt_episodes` (count)
* `timeout_episodes` (count)
* `critic_loss_mean`, `critic_loss_max`
* `imag_return_mean_last`

## Reporting

Use bootstrap 95% CI over seeds; report mean ± CI in tables; for
significance tests use Welch's t-test for unequal variances or
Mann–Whitney U for non-parametric distributions.
