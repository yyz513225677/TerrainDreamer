# TerrainDreamer (Cost-Reasoning Adaptive TERrain navigator)

A pure-PyTorch DreamerV3-style world-model RL algorithm for ROS 2 +
Gazebo Sim lunar Jackal autonomous navigation, with built-in support for
human takeover, behavior cloning, failed-mission memory, and failure-
recovery behavior cloning.

## Scope

This module is **algorithm-only**. It contains:

* the world model (RSSM, encoders, reward/continuation heads),
* the actor / critic,
* the mission FSM and outbound-trajectory memory,
* a safety shield,
* human-takeover routing,
* a replay buffer that carries control-mode and demo-type metadata,
* a failed-mission buffer,
* a trainer that combines Dreamer + BC + recovery-BC losses.

It does **not** contain:

* ROS 2 nodes, launch files, or `rclpy` calls,
* Gazebo Sim worlds, models, or sensor SDFs,
* joystick / keyboard / teleop code,
* Gazebo Classic APIs (gz-sim Harmonic / Jazzy bridges only, on the ROS side).

A separate ROS 2 Gymnasium wrapper is expected to provide already-processed
observations, deliver human actions during takeover, and decide when to
record a mission as "failed". The wrapper publishes the final action onto
`/cmd_vel` (or `/j100_0001/cmd_vel`) — typically via `ros_gz_bridge` when
the underlying simulator is Gazebo Sim.

## Observation format

```python
obs = {
    "lidar_bev"     : Tensor [B, 4, 128, 128],   # occupancy / max_h / elev / rough
    "imu"           : Tensor [B, 12],            # lin_acc(3), ang_vel(3), rpy(3), body_vel(3)
    "goal_vector"   : Tensor [B, 4],             # dx, dy, distance, bearing
    "mission_phase" : Tensor [B, 2],             # [1,0] outbound | [0,1] return
    "prev_action"   : Tensor [B, 2],             # previous (linear_x, angular_z)
}
```

## Action format

Continuous, per-dim ranges:

```
linear_x  in [0.0, 0.8]   m/s
angular_z in [-1.0, 1.0]  rad/s
```

## Instantiating

```python
from crater import Config, TerrainDreamer (Cost-Reasoning Adaptive TERrain navigator)Model, Trainer, ReplayBuffer

cfg = Config()                # all defaults
cfg.device = "cuda"           # or "cpu"
model = TerrainDreamer (Cost-Reasoning Adaptive TERrain navigator)Model(cfg)
trainer = Trainer(model, cfg)
buffer = ReplayBuffer(capacity=cfg.train.replay_capacity)
```

## Acting (autonomous)

```python
action, rssm_state = model.act(obs)
# action shape: [B, 2]  -> publish twist on /cmd_vel in the ROS wrapper
```

`act()` runs encoder → RSSM posterior step → actor → SafetyShield in one
call. State is returned so the next step can resume the same posterior.

## Human takeover

```python
# Wrapper detects gamepad activity, switches mode + supplies human_action:
final, rssm_state = model.select_action(
    obs,
    state=rssm_state,
    human_action=torch.tensor([[0.4, -0.3]]),
    control_mode="human",
)
```

The `HumanTakeover` instance inside the model auto-falls-back to
autonomous if no human action arrives within `manual_control_timeout`
seconds.

## Recording transitions

```python
# Normal autonomous transition:
buffer.add(Transition(obs, action, reward, done, next_obs,
                      info={"control_mode": "autonomous", "demo_type": "normal"}))

# Human demonstration (e.g. user takes over to show the robot how):
buffer.add(Transition(obs, human_action, reward, done, next_obs,
                      info={"control_mode": "human", "demo_type": "normal"}))
```

## Saving a failed mission

```python
model.record_failed_mission({
    "initial_pose":             (0.0, 0.0, 0.0),
    "destination":              (5.0, 2.5),
    "mission_phase_at_failure": "OUTBOUND",
    "outbound_trajectory":      memory.get_outbound_path(),
    "recent_observation_sequence": [obs_t0, obs_t1, ...],
    "recent_action_sequence":      [act_t0, act_t1, ...],
    "recent_reward_sequence":      [-0.01, -0.02, -10.0],
    "failure_reason":           "collision",
    "failure_pose":             (2.7, 1.4, 0.3),
    "current_target":           (5.0, 2.5),
    "episode_index":            42,
})
```

## Recording a human recovery demo

Once a human drives the rover out of a previously-failed state, record the
transitions as recovery demos so the trainer can prioritise them via the
recovery-BC loss:

```python
model.record_recovery_demo(
    buffer,
    obs=obs_t,
    human_action=human_action_t,
    reward=reward_t,
    next_obs=obs_t1,
    done=done_t,
    info={"source_failed_mission_id": mission_id},
)
```

The recovery transition is automatically tagged
`control_mode='human', demo_type='failure_recovery'`.

## Training step

```python
metrics = trainer.train_step(buffer)
# Returns None if buffer hasn't reached cfg.train.min_replay yet.
# Once eligible, performs:
#   1. world-model update on a normal sequence batch,
#   2. actor / critic on imagined rollouts,
#   3. behavior_cloning loss if any human demos exist,
#   4. recovery_bc loss if any failure-recovery demos exist.
# Actor total = dreamer_actor + bc_w * bc + recovery_bc_w * recovery_bc.
```

## Connecting to ROS 2 / Gazebo Sim

The ROS 2 wrapper (out-of-scope here) should:

1. Subscribe to the relevant Clearpath J100 sensor topics:
   * `/j100_0001/sensors/lidar3d_0/points` (`sensor_msgs/PointCloud2`)
   * `/j100_0001/sensors/imu_0/data` (`sensor_msgs/Imu`)
   * `/j100_0001/platform/odom` (`nav_msgs/Odometry`)
   * `/clock` (`rosgraph_msgs/Clock`, bridged from `/world/<world>/clock`
     via `ros_gz_bridge` — required so `use_sim_time` works).
2. Convert the point cloud into the 4-channel BEV expected by
   `BEVTerrainEncoder` and pack the obs dict as shown above.
3. Call `model.select_action(obs, state, human_action, control_mode)` at
   ~10 Hz.
4. Publish the returned `[linear_x, angular_z]` as `geometry_msgs/TwistStamped`
   on `/j100_0001/cmd_vel` (the Clearpath-side topic that the diff-drive
   controller consumes).
5. Track when a mission has failed (collision, timeout, tilt, out-of-map)
   and call `model.record_failed_mission(...)`.

Gazebo Sim is the only supported simulator. Use `ros_gz_bridge` for any
clock/IMU/odom bridging. Gazebo Classic is **not** supported.

## Smoke test

The module ships a runnable check that doesn't need ROS or Gazebo. From
the project root:

```bash
python3 -m crater.README   # not runnable as-is; see below
```

Easier: paste this into a Python shell or save as `smoke.py`:

```python
import torch
from crater import (
    Config, TerrainDreamer (Cost-Reasoning Adaptive TERrain navigator)Model, Trainer, ReplayBuffer, Transition,
    TrajectoryMemory, MissionManager, FailedMissionBuffer,
    CONTROL_MODE_HUMAN, DEMO_TYPE_RECOVERY,
)

cfg = Config()
cfg.device = "cpu"
cfg.train.batch_size = 2
cfg.train.seq_len = 4
cfg.train.imagine_horizon = 3
cfg.train.free_nats = 0.0
cfg.train.min_replay = 0

model = TerrainDreamer (Cost-Reasoning Adaptive TERrain navigator)Model(cfg)
print("params:", sum(p.numel() for p in model.parameters()))

# Dummy obs (B=1).
def dummy_obs(B=1):
    return {
        "lidar_bev":     torch.zeros(B, 4, 128, 128),
        "imu":           torch.zeros(B, 12),
        "goal_vector":   torch.zeros(B, 4),
        "mission_phase": torch.tensor([[1.0, 0.0]] * B),
        "prev_action":   torch.zeros(B, 2),
    }

obs = dummy_obs()
embed = model.encode_obs(obs)
print("encoder embed:", tuple(embed.shape))

action, state = model.act(obs)
print("act() autonomous:", action.tolist())

action_h, state = model.select_action(
    obs, state=state, control_mode="autonomous")
print("select_action autonomous:", action_h.tolist())

action_h, state = model.select_action(
    obs, state=state,
    human_action=torch.tensor([[0.4, -0.2]]),
    control_mode="human")
print("select_action human:", action_h.tolist())

# Fill a tiny episode in the replay buffer.
buffer = ReplayBuffer(capacity=1000)
obs_t = {k: v.squeeze(0) for k, v in obs.items()}
for t in range(8):
    a = torch.tensor([0.4, 0.0])
    info = {"control_mode": "autonomous", "demo_type": "normal"}
    buffer.add(Transition(obs_t, a, 0.1, t == 7, obs_t, info))
print("buffer steps:", len(buffer), "episodes:", buffer.num_episodes())

# Add a fake failed mission + a recovery demo episode.
mid = model.record_failed_mission({
    "initial_pose": (0.0, 0.0, 0.0),
    "destination": (5.0, 2.0),
    "mission_phase_at_failure": "OUTBOUND",
    "outbound_trajectory": [(0.0, 0.0, 0.0), (1.0, 0.5, 0.1)],
    "recent_observation_sequence": [obs_t],
    "recent_action_sequence": [torch.tensor([0.4, 0.0])],
    "recent_reward_sequence": [-10.0],
    "failure_reason": "collision",
    "failure_pose": (1.2, 0.6, 0.4),
    "current_target": (5.0, 2.0),
    "episode_index": 1,
})
print("failed mission id:", mid)

for t in range(8):
    model.record_recovery_demo(
        buffer, obs_t,
        torch.tensor([0.2, 0.1]), -0.01, obs_t, t == 7,
        info={"source_failed_mission_id": mid})

trainer = Trainer(model, cfg)
metrics = trainer.train_step(buffer)
print("trainer step metrics:")
for k, v in (metrics or {}).items():
    print(f"  {k}: {v:.4f}")

print("OK")
```

Expected: prints embedding shapes, two action vectors, buffer counts, a
failed mission ID, and a dict of loss values including `loss/world_total`,
`loss/actor_dreamer`, `loss/critic`, and (when demos are sampled)
`loss/bc` and `loss/recovery_bc`.
