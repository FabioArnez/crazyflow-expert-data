"""Evaluate a checkpoint on fresh tracks: success, saturation, speed tracking, label consistency.

Example: ``expert-eval --ckpt outputs/mixed_tracks/<run>/checkpoint.pkl --track-type circle``.

Every world runs one episode from a fresh reset; the outcome of that first episode is recorded
(the environment auto-resets afterwards, which is ignored). Besides the outcome rates (with Wilson
95 % intervals), the summary reports what the data-collection QC needs (spec §3.9) on the expert's
own commands: the fraction of saturated command components per axis, how well the realised speed
follows the episode's target speed, and the commanded-vs-realised consistency of spec eq. (3).
``--save-traj`` dumps the first ``--traj-worlds`` worlds' trajectories.

Ported from crazyflow-gate-ppo (``gate_ppo/evaluate.py``); the QC metrics, the target speed and the
tanh-squashed policy are new. The summary stores the checkpoint path relative to the working
directory and its content hash, never an absolute path.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from expert_data.checkpoint import checkpoint_hash, load_checkpoint  # noqa: E402
from expert_data.config import EnvConfig  # noqa: E402
from expert_data.env import OBS_DIM, EnvParams, GateVelocityEnv  # noqa: E402
from expert_data.geometry import yaw_from_quat  # noqa: E402
from expert_data.networks import ActorCritic  # noqa: E402
from expert_data.ppo import SATURATION_THRESHOLD, RunningStats, policy_action  # noqa: E402
from expert_data.tracking import AXES, masked_label_consistency  # noqa: E402
from expert_data.velocity import realised_command  # noqa: E402

OUTCOMES = ("success", "crashed", "missed", "timeout", "gates", "length", "return")


def wilson_interval(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion ``k / n``."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="checkpoint.pkl written by expert-train")
    p.add_argument("--n-gates", type=int, default=None, help="gates per track (default: training)")
    p.add_argument("--n-gates-min", type=int, default=None, help="default: same as --n-gates")
    p.add_argument("--n-episodes", type=int, default=1024, help="worlds = episodes")
    p.add_argument("--yaw-range", type=float, default=None, help="default: training env.yaw_range")
    p.add_argument("--track-type", choices=["chain", "circle", "mixed"], default=None)
    p.add_argument("--gates-per-turn", type=float, default=None, help="circle: turn = 2*pi/this")
    p.add_argument("--circle-prob", type=float, default=None, help="mixed: share of circles")
    p.add_argument("--gate-yaw-noise", type=float, default=None, help="chain: normal offset (rad)")
    p.add_argument("--target-speed", type=float, default=None, help="fix v_target (m/s)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--no-dr", action="store_true", help="disable domain randomisation")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-traj", type=str, default=None, help="write trajectories to this .npz")
    p.add_argument("--traj-worlds", type=int, default=8)
    p.add_argument("--json", type=str, default=None, help="write the summary to this .json")
    return p.parse_args(argv)


def _relative(path: str) -> str:
    try:
        return os.path.relpath(Path(path).resolve(), Path.cwd())
    except ValueError:  # other drive (Windows); keep the file name only
        return Path(path).name


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt["config"]
    if np.asarray(ckpt["obs_stats"]["mean"]).shape[-1] != OBS_DIM:
        raise SystemExit(f"checkpoint observation is not {OBS_DIM}-D; it predates this env version")
    n_gates = args.n_gates or cfg["env"]["n_gates"]
    n_gates_min = args.n_gates_min or n_gates
    overrides = {"n_envs": args.n_episodes, "n_gates": n_gates, "n_gates_min": n_gates_min}
    if args.track_type:
        overrides["track_type"] = args.track_type
    if args.gate_yaw_noise is not None:
        overrides["gate_yaw_noise"] = args.gate_yaw_noise
    known = {f.name for f in dataclasses.fields(EnvConfig)}
    env_cfg = EnvConfig(**{k: v for k, v in {**cfg["env"], **overrides}.items() if k in known})
    env = GateVelocityEnv(env_cfg, device=args.device)
    squash = bool(cfg["net"].get("squash", False))
    network = ActorCritic(env.action_dim, tuple(cfg["net"]["hidden"]), cfg["net"]["log_std_init"])
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    stats = RunningStats(
        mean=jnp.asarray(ckpt["obs_stats"]["mean"]),
        var=jnp.asarray(ckpt["obs_stats"]["var"]),
        count=jnp.float32(ckpt["obs_stats"]["count"]),
    )
    normalize_obs = cfg["ppo"]["normalize_obs"]
    env_params = EnvParams.from_config(
        env_cfg,
        yaw_range=args.yaw_range,
        domain_randomization=not args.no_dr,
        circle_prob=args.circle_prob,
        gates_per_turn=args.gates_per_turn,
        target_speed=args.target_speed,
    )
    slowest = float(env_params.target_speed_min)
    max_steps = int(math.ceil(float(env.max_steps(jnp.array(n_gates), jnp.array(slowest)))))
    k_env, k_run = jax.random.split(jax.random.key(args.seed))
    state = env.init_state(k_env)
    state, obs = env.reset(state, env_params)
    n_traj = min(args.traj_worlds, args.n_episodes)
    track = {
        "gates_pos": np.asarray(state.gates_pos[:n_traj]),
        "gates_yaw": np.asarray(state.gates_yaw[:n_traj]),
        "n_active": np.asarray(state.n_active[:n_traj]),
    }
    target_speed = np.asarray(state.target_speed)

    def policy(obs, key):
        mean, log_std, _ = network.apply(params, stats.normalize(obs) if normalize_obs else obs)
        if args.stochastic:
            mean = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return policy_action(mean, squash)

    @jax.jit
    def rollout(state, obs, key):
        n = env.n_envs
        outcome0 = {k: jnp.zeros((n,)) for k in OUTCOMES}

        def body(carry, _):
            state, obs, key, first_done, outcome = carry
            key, k = jax.random.split(key)
            action = policy(obs, k)
            pos0 = state.sim.states.pos[:, 0]
            yaw0 = yaw_from_quat(state.sim.states.quat[:, 0])
            state, obs, reward, terminated, truncated, final_obs, info = env.step(
                state, action, env_params
            )
            done = terminated | truncated
            new = done & ~first_done
            values = {
                "success": info["success"],
                "crashed": info["crashed"],
                "missed": info["missed"],
                "timeout": truncated.astype(jnp.float32),
                "gates": info["gates_passed"],
                "length": info["episode_length"],
                "return": info["episode_return"],
            }
            outcome = {k: jnp.where(new, values[k], outcome[k]) for k in outcome}
            realised = realised_command(
                pos0, info["pos"], yaw0, yaw_from_quat(info["quat"]), env.grid.dt
            )
            per_step = {
                "alive": ~first_done,
                "action": action,
                "cmd": info["cmd"],
                "realised": realised,
                "speed": info["speed"],
            }
            traj = {
                "pos": info["pos"][:n_traj],
                "quat": info["quat"][:n_traj],
                "vel": info["vel"][:n_traj],
                "cmd": info["cmd"][:n_traj],
                "gates_passed": info["gates_passed"][:n_traj],
                "done": done[:n_traj],
            }
            return (state, obs, key, first_done | done, outcome), (per_step, traj)

        carry, (per_step, traj) = jax.lax.scan(
            body, (state, obs, key, jnp.zeros((n,), bool), outcome0), None, max_steps
        )
        return carry[3], carry[4], per_step, traj

    finished, outcome, per_step, traj = jax.device_get(rollout(state, obs, k_run))
    finished = np.asarray(finished)
    outcome = {k: np.asarray(v) for k, v in outcome.items()}
    # Worlds still flying at max_steps count as timeouts.
    outcome["timeout"] = np.where(finished, outcome["timeout"], 1.0)
    n_active = np.asarray(state.n_active)
    n = args.n_episodes
    alive = np.asarray(per_step["alive"])  # (T, N)
    n_alive = max(int(alive.sum()), 1)
    abs_action = np.abs(np.asarray(per_step["action"]))
    sat = (abs_action >= SATURATION_THRESHOLD) & alive[..., None]
    sat95 = (abs_action >= 0.95) & alive[..., None]
    ep_steps = np.maximum(alive.sum(0), 1)
    ep_speed = (np.asarray(per_step["speed"]) * alive).sum(0) / ep_steps
    speed_err = ep_speed - target_speed
    r0, lag, r_best = masked_label_consistency(
        np.asarray(per_step["cmd"]), np.asarray(per_step["realised"]), alive
    )
    k_success = float(outcome["success"].sum())
    v_lo, v_hi = float(env_params.target_speed_min), float(env_params.target_speed_max)
    speed_bins = np.linspace(v_lo, v_hi, 4) if v_hi > v_lo else np.array([v_lo, v_hi])
    per_speed = {}
    for lo, hi in zip(speed_bins[:-1], speed_bins[1:], strict=True):
        sel = (target_speed >= lo) & (target_speed <= hi)
        if sel.any():
            per_speed[f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(sel.sum()),
                "success_rate": float(outcome["success"][sel].mean()),
                "speed_minus_target": float(speed_err[sel].mean()),
            }
    summary = {
        "checkpoint": _relative(args.ckpt),
        "checkpoint_sha256_16": checkpoint_hash(args.ckpt),
        "policy": "tanh" if squash else "clip",
        "n_gates": n_gates,
        "n_gates_min": n_gates_min,
        "n_episodes": n,
        "max_steps": max_steps,
        "yaw_range": float(env_params.yaw_range),
        "track_type": env_cfg.track_type,
        "circle_prob": float(env_params.circle_prob),
        "gate_yaw_noise": env_cfg.gate_yaw_noise,
        "gates_per_turn": [
            float(env_params.circle_gates_per_turn_min),
            float(env_params.circle_gates_per_turn_max),
        ],
        "target_speed": [v_lo, v_hi],
        "deterministic": not args.stochastic,
        "domain_randomization": not args.no_dr,
        "success_rate": k_success / n,
        "success_wilson95": list(wilson_interval(k_success, n)),
        "crash_rate": float(outcome["crashed"].mean()),
        "miss_rate": float(outcome["missed"].mean()),
        "timeout_rate": float(outcome["timeout"].mean()),
        "gates_passed_mean": float(outcome["gates"].mean()),
        "episode_length_s": float(outcome["length"].mean() / env_cfg.decision_freq),
        "return_mean": float(outcome["return"].mean()),
        "per_n_active": {
            int(k): float(outcome["success"][n_active == k].mean()) for k in np.unique(n_active)
        },
        "per_target_speed": per_speed,
        "saturation_ge_0.99": dict(zip(AXES, (sat.sum((0, 1)) / n_alive).tolist(), strict=True)),
        "saturation_ge_0.95": dict(zip(AXES, (sat95.sum((0, 1)) / n_alive).tolist(), strict=True)),
        "speed_tracking": {
            "corr_episode_speed_target": float(np.corrcoef(ep_speed, target_speed)[0, 1])
            if target_speed.std() > 1e-6
            else None,
            "mean_speed_minus_target": float(speed_err.mean()),
            "mean_abs_speed_minus_target": float(np.abs(speed_err).mean()),
        },
        "label_consistency_eq3": {
            "r_lag0": dict(zip(AXES, r0.round(4).tolist(), strict=True)),
            "best_lag_steps": dict(zip(AXES, lag.tolist(), strict=True)),
            "r_best_lag": dict(zip(AXES, r_best.round(4).tolist(), strict=True)),
        },
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_traj:
        done = np.asarray(traj["done"])  # (T, K)
        ep_len = np.where(done.any(0), done.argmax(0) + 1, done.shape[0])
        np.savez(
            args.save_traj,
            pos=np.asarray(traj["pos"]),
            quat=np.asarray(traj["quat"]),
            vel=np.asarray(traj["vel"]),
            cmd=np.asarray(traj["cmd"]),
            gates_passed=np.asarray(traj["gates_passed"]),
            ep_len=ep_len,
            success=outcome["success"][:n_traj],
            target_speed=target_speed[:n_traj],
            decision_freq=env_cfg.decision_freq,
            sim_freq=env_cfg.sim_freq,
            drone=env_cfg.drone,
            half_opening=env_cfg.half_opening,
            frame_thickness=env_cfg.frame_thickness,
            **track,
        )
        print(f"trajectories of {n_traj} worlds -> {args.save_traj}")
    return summary


def cli() -> None:
    """Console entry point ``expert-eval``.

    ``main`` returns the summary for programmatic use; a console script passes the return value
    to ``sys.exit``, which would turn the dict into exit code 1.
    """
    main()


if __name__ == "__main__":
    cli()
