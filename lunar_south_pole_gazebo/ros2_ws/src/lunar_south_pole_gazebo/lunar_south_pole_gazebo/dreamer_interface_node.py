"""dreamer_interface_node — Stage 2 implementation.

Loads the parent repo's trained TerrainDreamerModel + DreamerActor and
drives the Jackal on the lunar DEM. The actor expects:

  state_feat : [B, deter+stoch_total] = [B, 4352]   (deter=256, stoch=64x64)
  goal_obs   : [B, 4]                                (dx_norm, dy_norm,
                                                       dist_norm, heading_err_norm)

Each tick (10 Hz):
  1. Process the latest /lunar_jackal/points into [N_pad, 8] features
     via PointCloudProcessor.
  2. Encode → embed [B, 256].
  3. Step the RSSM with (prev_state, prev_action, embed) → posterior state.
  4. Compute goal_obs from /lunar_jackal/odom and the active goal.
  5. actor.act(state_feat, goal_obs, explore=False) → action ∈ [-1,1]²
  6. Map to Twist (linear scale ≤ MAX_LIN, angular scale ≤ MAX_ANG) and
     publish on /lunar_jackal/cmd_vel.

Run in the project venv so torch is available:
  scripts/run_dreamer_on_jackal.sh
"""
from __future__ import annotations

import math
import os
import sys
import threading
from pathlib import Path
from typing import Optional


# ---- Make the parent-repo source importable from this node ---------------
_REPO_ROOT = Path(os.environ.get(
    "TERRAIN_DREAMER_REPO_ROOT",
    "/home/rickslab3/Documents/Leo/terrain_dreamer"))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Pure-function helpers (importable by tests without rclpy/torch)
def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _goal_obs_vector(rx: float, ry: float, ryaw: float,
                     gx: float, gy: float,
                     dist_norm_scale: float = 50.0):
    """Match the training-time goal_obs convention:
       [dx_norm, dy_norm, dist_norm, heading_err_norm]
    All four are clipped to [-1, 1]."""
    dx = gx - rx
    dy = gy - ry
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    heading_err = math.atan2(math.sin(bearing - ryaw),
                             math.cos(bearing - ryaw))
    return [
        max(-1.0, min(1.0, dx / dist_norm_scale)),
        max(-1.0, min(1.0, dy / dist_norm_scale)),
        max(-1.0, min(1.0, dist / dist_norm_scale)),
        max(-1.0, min(1.0, heading_err / math.pi)),
    ]


def main(args: Optional[list] = None) -> int:  # pragma: no cover
    # rclpy
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import Imu, PointCloud2
        from sensor_msgs_py import point_cloud2 as pc2
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String
        import numpy as np
        import yaml
    except ImportError as exc:
        print(f"[dreamer_interface_node] missing ROS/numpy dep: {exc}",
              file=sys.stderr)
        return 1

    # torch + terrain_dreamer
    try:
        import torch
        import numpy as _np  # alias to avoid name clash with later numpy import
        from terrain_dreamer.world_model.terrain_dreamer_model import (
            TerrainDreamerModel)
        from terrain_dreamer.world_model.dreamer_policy import (
            DreamerActor, DreamerCritic, imagine_train)
        from terrain_dreamer.preprocessing.point_cloud_processor import (
            PointCloudProcessor, PointCloud)
        from terrain_dreamer.training.dreamer_buffer import DreamerReplayBuffer
    except ImportError as exc:
        print(f"[dreamer_interface_node] missing torch/terrain_dreamer dep: {exc}",
              file=sys.stderr)
        print("  Hint: run via scripts/run_dreamer_on_jackal.sh which "
              "uses the project venv.", file=sys.stderr)
        return 1

    rclpy.init(args=args)
    node = Node("dreamer_interface_node")

    # ---- params -----------------------------------------------------------
    checkpoint = node.declare_parameter(
        "checkpoint_path",
        str(_REPO_ROOT / "checkpoints_auto" / "best_model.pt")).value
    goals_yaml = node.declare_parameter(
        "goals_yaml",
        "/opt/ros/jazzy/share/lunar_south_pole_gazebo/config/lunar_goals.yaml"
    ).value
    if not Path(goals_yaml).is_file():
        try:
            from ament_index_python.packages import get_package_share_directory
            goals_yaml = str(Path(get_package_share_directory(
                "lunar_south_pole_gazebo")) / "config" / "lunar_goals.yaml")
        except Exception:
            pass

    rate_hz = float(node.declare_parameter("rate_hz", 10.0).value)
    max_lin = float(node.declare_parameter("max_linear_mps", 0.4).value)
    max_ang = float(node.declare_parameter("max_angular_radps", 0.5).value)
    use_gpu = bool(node.declare_parameter("use_gpu", True).value)
    fallback_goal_x = float(node.declare_parameter(
        "fallback_goal_x", 5.0).value)
    fallback_goal_y = float(node.declare_parameter(
        "fallback_goal_y", 0.0).value)
    pad_to = int(node.declare_parameter("max_points", 1024).value)
    # ---- exploration + reset params (P0 from publication-readiness fix) -
    # When train=True, sample noisy actions from the actor (instead of
    # the deterministic mean) so the policy actually sees varied
    # transitions. The std-head's output is gated by `min_std`=0.12 so
    # this is bounded Gaussian noise.
    explore_during_train = bool(node.declare_parameter(
        "explore_during_train", True).value)
    # Periodic teleport back to spawn so the rover keeps seeing near-
    # goal states. Set teleport_every_s ≤ 0 to disable.
    teleport_every_s = float(node.declare_parameter(
        "teleport_every_s", 90.0).value)
    teleport_x = float(node.declare_parameter("teleport_x", 0.0).value)
    teleport_y = float(node.declare_parameter("teleport_y", 0.0).value)
    teleport_z = float(node.declare_parameter("teleport_z", 400.0).value)
    teleport_world = str(node.declare_parameter(
        "teleport_world", "lunar_south_pole").value)
    teleport_model = str(node.declare_parameter(
        "teleport_model", "lunar_jackal").value)
    # ---- reward shaping params (P1 fix) --------------------------------
    reward_shaping_w = float(node.declare_parameter(
        "reward_shaping_w", 40.0).value)        # was 4.0
    reward_step_penalty = float(node.declare_parameter(
        "reward_step_penalty", 0.003).value)    # was 0.03
    reward_alive_bonus = float(node.declare_parameter(
        "reward_alive_bonus", 0.05).value)      # NEW
    reward_reach = float(node.declare_parameter(
        "reward_reach", 15.0).value)
    # ---- stop-and-turn controller params --------------------------------
    # When |normalized angular action| exceeds `turn_enter_thresh`, the
    # rover stops forward motion (lin=0) and turns in place. Once the
    # |angular action| falls below `turn_exit_thresh` AND it has been
    # turning for at least `min_turn_steps` ticks, forward motion
    # resumes. Hysteresis prevents oscillation at the boundary. Set
    # `turn_enter_thresh` ≥ 1.0 to disable.
    turn_enter_thresh = float(node.declare_parameter(
        "turn_enter_thresh", 0.35).value)
    turn_exit_thresh = float(node.declare_parameter(
        "turn_exit_thresh", 0.15).value)
    min_turn_steps = int(node.declare_parameter(
        "min_turn_steps", 3).value)
    # ---- training params (when train:=true) ------------------------------
    train_mode = bool(node.declare_parameter("train", False).value)
    seq_len = int(node.declare_parameter("seq_len", 32).value)
    episode_max_steps = int(node.declare_parameter(
        "episode_max_steps", 200).value)
    lr_model = float(node.declare_parameter("lr_model", 1e-5).value)
    lr_actor = float(node.declare_parameter("lr_actor", 8e-6).value)
    lr_critic = float(node.declare_parameter("lr_critic", 8e-6).value)
    train_batch = int(node.declare_parameter("train_batch", 4).value)
    imagine_horizon = int(node.declare_parameter(
        "imagine_horizon", 15).value)
    save_every = int(node.declare_parameter("save_every_steps", 500).value)
    save_dir = node.declare_parameter(
        "save_dir",
        str(_REPO_ROOT / "lunar_south_pole_gazebo" / "outputs"
            / "live_checkpoints")).value

    device = (torch.device("cuda")
              if (use_gpu and torch.cuda.is_available())
              else torch.device("cpu"))
    node.get_logger().info(f"device={device}")

    # ---- model -----------------------------------------------------------
    # Dims derived from checkpoint inspection — see
    # docs/jackal_sensor_goal_camera_runbook.md §15.
    model = TerrainDreamerModel(
        input_channels=8, embed_dim=256, action_dim=2,
        deter_dim=256, stoch_dim=64, stoch_classes=64, hidden_dim=256,
    ).to(device)
    state_dim = model.rssm.deter_dim + model.rssm.stoch_total
    actor = DreamerActor(state_dim=state_dim, goal_dim=4,
                         action_dim=2).to(device)
    critic = DreamerCritic(state_dim=state_dim, goal_dim=4).to(device)

    if not Path(checkpoint).is_file():
        node.get_logger().error(f"checkpoint not found: {checkpoint}")
        rclpy.shutdown()
        return 2
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    actor.load_state_dict(ck["actor"])
    if "critic" in ck:
        critic.load_state_dict(ck["critic"])
    node.get_logger().info(
        f"loaded checkpoint {checkpoint} "
        f"(model={len(ck['model'])} tensors, actor={len(ck['actor'])} tensors)")

    # Eval-only by default. Training mode flips train() and wires
    # optimizers + replay buffer.
    if train_mode:
        model.train(); actor.train(); critic.train()
        model_opt = torch.optim.Adam(model.parameters(), lr=lr_model)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr_critic)
        buffer = DreamerReplayBuffer(
            seq_len=seq_len, max_points=pad_to, feat_dim=8,
            action_dim=2, max_episodes=50, min_episodes=2)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        node.get_logger().info(
            f"TRAINING MODE — lr_model={lr_model}, lr_actor={lr_actor}, "
            f"lr_critic={lr_critic}; checkpoints → {save_dir}")
    else:
        model.eval(); actor.eval(); critic.eval()
        model_opt = actor_opt = critic_opt = buffer = None

    processor = PointCloudProcessor(voxel_size=0.15, max_points=pad_to)

    # ---- goals -----------------------------------------------------------
    goal_map = {}
    if Path(goals_yaml).is_file():
        gdata = yaml.safe_load(Path(goals_yaml).read_text()) or {}
        for g in gdata.get("goals", []):
            goal_map[g["goal_id"]] = (float(g["x"]), float(g["y"]))
        node.get_logger().info(
            f"loaded {len(goal_map)} goals from {goals_yaml}")
    else:
        node.get_logger().warn(
            f"goals_yaml {goals_yaml} not found — falling back to "
            f"({fallback_goal_x},{fallback_goal_y})")

    state = {
        "cloud_xyzi": None,    # np.ndarray [N,4]
        "imu": None,           # Imu
        "odom": None,          # Odometry
        "current_goal_id": "",
        "goal_distance": None, # float
        "goal_reached": False, # bool
        "rssm_state": None,    # RSSMState
        "prev_action": torch.zeros(1, 2, device=device),
        "lock": threading.Lock(),
        "steps": 0,
        # ---- stop-and-turn FSM state ------------------------------------
        "turning": False,
        "turn_step_counter": 0,
        # ---- training-mode bookkeeping (used only when train_mode=True) ----
        "ep_features": [],
        "ep_actions": [],
        "ep_rewards": [],
        "ep_continues": [],
        "ep_goal_obs": [],
        "prev_dist": None,
        "episodes_pushed": 0,
        "train_calls": 0,
        "last_loss": None,
    }

    # ---- subscriptions ---------------------------------------------------
    qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    def on_cloud(msg):
        try:
            structured = pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            arr = np.asarray(structured)
            if arr.size == 0:
                xyzi = np.zeros((0, 4), dtype=np.float32)
            else:
                xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1
                               ).astype(np.float32)
                intensity = np.ones((xyz.shape[0], 1), dtype=np.float32)
                xyzi = np.concatenate([xyz, intensity], axis=-1)
        except Exception as e:
            node.get_logger().warn(f"cloud parse error: {e}")
            xyzi = np.zeros((0, 4), dtype=np.float32)
        with state["lock"]:
            state["cloud_xyzi"] = xyzi

    def on_imu(msg):
        with state["lock"]:
            state["imu"] = msg

    def on_odom(msg):
        with state["lock"]:
            state["odom"] = msg

    def on_goal(msg):
        with state["lock"]:
            state["current_goal_id"] = (msg.data or "").strip()

    def on_goal_distance(msg):
        with state["lock"]:
            state["goal_distance"] = float(msg.data)

    def on_goal_reached(msg):
        with state["lock"]:
            state["goal_reached"] = bool(msg.data)

    from std_msgs.msg import Float32, Bool
    node.create_subscription(PointCloud2, "/lunar_jackal/points",
                             on_cloud, qos_sensor)
    node.create_subscription(Imu, "/lunar_jackal/imu", on_imu, qos_sensor)
    node.create_subscription(Odometry, "/lunar_jackal/odom", on_odom, 10)
    node.create_subscription(String, "/lunar_jackal/current_goal",
                             on_goal, 10)
    node.create_subscription(Float32, "/lunar_jackal/goal_distance",
                             on_goal_distance, 10)
    node.create_subscription(Bool, "/lunar_jackal/goal_reached",
                             on_goal_reached, 10)

    cmd_pub = node.create_publisher(
        Twist, "/lunar_jackal/cmd_vel",
        QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    # ---- tick ------------------------------------------------------------
    def tick():
        with state["lock"]:
            cloud = state["cloud_xyzi"]
            imu = state["imu"]
            odom = state["odom"]
            goal_id = state["current_goal_id"]
            prev_action = state["prev_action"].clone()
            prev_state = state["rssm_state"]

        # Until we have at least odom + cloud, sit still.
        if odom is None or cloud is None:
            cmd_pub.publish(Twist())
            return

        # ---- features --------------------------------------------------
        if cloud.shape[0] == 0:
            feats = np.zeros((pad_to, 8), dtype=np.float32)
        else:
            try:
                pc = PointCloud(timestamp=0.0, points=cloud)
                proc = processor.process(pc)
                feats = np.zeros((pad_to, 8), dtype=np.float32)
                m = min(proc.features.shape[0], pad_to)
                feats[:m] = proc.features[:m]
            except Exception as e:
                node.get_logger().warn(f"processor error: {e}")
                feats = np.zeros((pad_to, 8), dtype=np.float32)

        feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)  # [1,N,8]

        # ---- RSSM step -------------------------------------------------
        with torch.no_grad():
            embed = model.encoder(feats_t)                          # [1, embed]
            if prev_state is None:
                prev_state = model.rssm.initial_state(
                    batch_size=1, device=device)
            _prior, post = model.rssm.observe_step(
                prev_state, prev_action, embed)
            state_feat = post.feature                               # property → tensor

        # ---- goal obs --------------------------------------------------
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        gx, gy = goal_map.get(goal_id, (fallback_goal_x, fallback_goal_y))
        goal_vec = _goal_obs_vector(p.x, p.y, yaw, gx, gy)
        goal_t = torch.tensor([goal_vec], dtype=torch.float32, device=device)

        # ---- action ----------------------------------------------------
        with torch.no_grad():
            # P0 fix: actually explore during data collection. The
            # frozen-actor pre-training case keeps explore=False.
            explore = train_mode and explore_during_train
            action = actor.act(state_feat, goal_t, explore=explore)
        a = action.squeeze(0).cpu().numpy()
        raw_lin = float(a[0]) * max_lin
        raw_ang = float(a[1]) * max_ang
        ang_mag = abs(float(a[1]))

        # ---- Stop-and-turn FSM ----
        # State transitions (with hysteresis):
        #   DRIVING  → TURNING  : |ang| > turn_enter_thresh
        #   TURNING  → DRIVING  : |ang| < turn_exit_thresh AND
        #                          turn_step_counter >= min_turn_steps
        if state["turning"]:
            state["turn_step_counter"] += 1
            if (ang_mag < turn_exit_thresh
                    and state["turn_step_counter"] >= min_turn_steps):
                state["turning"] = False
                state["turn_step_counter"] = 0
                node.get_logger().info(
                    "[stop-and-turn] turn complete → resuming forward")
        else:
            if ang_mag > turn_enter_thresh:
                state["turning"] = True
                state["turn_step_counter"] = 0
                node.get_logger().info(
                    f"[stop-and-turn] stopping for turn (|ang|={ang_mag:.2f})")

        if state["turning"]:
            lin = 0.0           # freeze forward motion mid-turn
            ang = raw_ang
        else:
            lin = raw_lin
            ang = raw_ang

        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        cmd_pub.publish(tw)

        # ---- persist for next tick -------------------------------------
        with state["lock"]:
            state["rssm_state"] = post
            state["prev_action"] = action.detach()
            state["steps"] += 1

        # ---- training mode --------------------------------------------
        if train_mode:
            with state["lock"]:
                dist = state["goal_distance"]
                reached = state["goal_reached"]
            # shaping reward = distance decrease scaled (P1 reward fix)
            if dist is None or state["prev_dist"] is None:
                shaping = 0.0
            else:
                shaping = (state["prev_dist"] - dist) * reward_shaping_w
            # alive bonus = small reward for actually moving — counters
            # the freeze equilibrium that emerged in the first 5 247 steps.
            alive_bonus = reward_alive_bonus * abs(lin)
            reward = (shaping
                      + alive_bonus
                      - reward_step_penalty
                      + (reward_reach if reached else 0.0))
            state["prev_dist"] = dist if dist is not None else state["prev_dist"]

            # Append step to the in-progress episode
            state["ep_features"].append(feats.copy())
            state["ep_actions"].append(a.copy())
            state["ep_rewards"].append(float(reward))
            state["ep_continues"].append(0.0 if reached else 1.0)
            state["ep_goal_obs"].append(_np.array(goal_vec, dtype=_np.float32))

            ep_len = len(state["ep_actions"])
            should_end = reached or ep_len >= episode_max_steps
            if should_end and ep_len >= seq_len:
                # Push to replay buffer
                ok = buffer.add_episode(
                    features=_np.stack(state["ep_features"], axis=0),
                    actions=_np.stack(state["ep_actions"], axis=0),
                    rewards=_np.asarray(state["ep_rewards"], dtype=_np.float32),
                    continues=_np.asarray(state["ep_continues"], dtype=_np.float32),
                    goal_obs=_np.stack(state["ep_goal_obs"], axis=0),
                )
                if ok:
                    state["episodes_pushed"] += 1
                    node.get_logger().info(
                        f"[train] episode pushed (len={ep_len}, "
                        f"reached={reached}); buffer eps="
                        f"{buffer.num_episodes()}")
                # Reset episode buffers
                state["ep_features"].clear()
                state["ep_actions"].clear()
                state["ep_rewards"].clear()
                state["ep_continues"].clear()
                state["ep_goal_obs"].clear()
                state["prev_dist"] = None

                # Run one gradient update if buffer is ready
                if buffer.ready():
                    try:
                        batch = buffer.sample(train_batch)
                        feats_t = torch.from_numpy(batch["features"]).to(device)
                        acts_t = torch.from_numpy(batch["actions"]).to(device)
                        rews_t = torch.from_numpy(batch["rewards"]).to(device)
                        conts_t = torch.from_numpy(batch["continues"]).to(device)
                        goal_t = torch.from_numpy(batch["goal_obs"]).to(device)

                        # World model loss
                        wm_losses = model.training_loss(
                            feats_t, acts_t, rews_t, conts_t)
                        model_opt.zero_grad()
                        wm_losses["total"].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        model_opt.step()

                        # Actor/critic via imagination
                        with torch.no_grad():
                            B, T = acts_t.shape[:2]
                            BT = B * T
                            N, C = feats_t.shape[2], feats_t.shape[3]
                            embeds = model.encoder(
                                feats_t.reshape(BT, N, C)).reshape(B, T, -1)
                            _, post_seq = model.rssm.observe_sequence(
                                embeds, acts_t)
                        ac_losses = imagine_train(
                            model, actor, critic,
                            actor_opt, critic_opt,
                            start_states=post_seq,
                            start_goals=goal_t,
                            device=device,
                            H=imagine_horizon,
                            entropy_scale=1e-3,
                        )
                        state["train_calls"] += 1
                        state["last_loss"] = float(wm_losses["total"].item())
                        node.get_logger().info(
                            f"[train] step {state['train_calls']}: "
                            f"wm/total={state['last_loss']:.3f} "
                            f"ac/actor={ac_losses.get('actor',0):.3f} "
                            f"ac/critic={ac_losses.get('critic',0):.3f}")
                    except Exception as e:
                        node.get_logger().warn(f"train step failed: {e}")

            # Save checkpoint periodically
            if (state["steps"] % save_every == 0
                    and state["train_calls"] > 0):
                ckpt_path = Path(save_dir) / f"live_ckpt_{state['steps']:07d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "step": state["steps"],
                    "train_calls": state["train_calls"],
                }, ckpt_path)
                node.get_logger().info(f"[train] saved {ckpt_path}")

        if state["steps"] % 50 == 1:
            extra = ""
            if train_mode:
                extra = (f" buf={buffer.num_episodes()}eps "
                         f"trains={state['train_calls']} "
                         f"last_wm={state['last_loss']}")
            node.get_logger().info(
                f"step={state['steps']} goal={goal_id!r} "
                f"action=({lin:+.2f} m/s, {ang:+.2f} rad/s) "
                f"goal_obs={[f'{v:+.2f}' for v in goal_vec]}{extra}")

    node.create_timer(1.0 / rate_hz, tick)

    # ---- periodic teleport (P0 fix) ------------------------------------
    # Use `gz service` to set the rover pose back to spawn. Runs in a
    # subprocess to keep rclpy free; called from a low-rate ROS timer.
    if train_mode and teleport_every_s > 0:
        import subprocess
        def teleport_back():
            try:
                # Publish a zero Twist first so the diff-drive plugin
                # doesn't continue applying velocity through the warp.
                cmd_pub.publish(Twist())
                req = (f'name: "{teleport_model}" '
                       f'position {{x: {teleport_x} y: {teleport_y} '
                       f'z: {teleport_z}}} orientation {{w: 1}}')
                subprocess.Popen(
                    ["gz", "service",
                     "-s", f"/world/{teleport_world}/set_pose",
                     "--reqtype", "gz.msgs.Pose",
                     "--reptype", "gz.msgs.Boolean",
                     "--timeout", "1500",
                     "--req", req],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Reset cached RSSM state — the post-teleport observation
                # is essentially a new episode start.
                with state["lock"]:
                    state["rssm_state"] = None
                    state["prev_dist"] = None
                node.get_logger().info(
                    f"[reset] teleported {teleport_model} → "
                    f"({teleport_x}, {teleport_y}, {teleport_z})")
            except Exception as e:
                node.get_logger().warn(f"teleport failed: {e}")
        node.create_timer(teleport_every_s, teleport_back)

    node.get_logger().info(
        "dreamer_interface_node running — publishing /lunar_jackal/cmd_vel "
        f"(explore={explore_during_train and train_mode}, "
        f"teleport_every={teleport_every_s}s)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_pub.publish(Twist())   # final zero on shutdown
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
