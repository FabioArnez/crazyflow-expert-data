"""Split the spec eq. (3) check on an expert's own rollouts into its two links.

Usage: ``uv run python scripts/analyse_label_consistency.py --ckpt <ckpt.pkl> [--out x.json]``.

The velocity layer turns the command ``a`` into a rate-limited reference velocity ``v_ref``
(``velocity.CommandConfig.acc_max_*``), which the Mellinger controller then tracks. This script logs
both and reports, per axis (heading frame, deterministic policy, first episode per world):

- ``cmd -> realised``: the eq. (3) check as ``expert-eval`` computes it;
- ``cmd -> v_ref``: what the rate limit does to the command (mean ``v_ref`` over the decision);
- ``v_ref -> realised``: how well the controller follows the feasible reference;
- the fraction of decisions in which the rate limit is active (``v_ref`` has not reached the
  command at the end of the decision), overall and after the first second;
- ``cmd -> realised`` again, restricted to decisions after the first second, and to decisions in
  which the rate limit is inactive;
- the share of the command variance that lies within episodes (the rest is between episodes);
- command chatter: the mean step ``|a_t - a_{t-1}|``, the rate at which consecutive steps change
  sign (≈ 1 for a period-2 oscillation, small for a smooth command) and the share of the
  within-episode command power above 5 Hz (the decision rate is 30 Hz).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from expert_data.checkpoint import checkpoint_hash, load_checkpoint  # noqa: E402
from expert_data.config import EnvConfig  # noqa: E402
from expert_data.env import EnvParams, GateVelocityEnv  # noqa: E402
from expert_data.evaluate import _relative  # noqa: E402
from expert_data.geometry import world_to_heading, yaw_from_quat  # noqa: E402
from expert_data.networks import ActorCritic  # noqa: E402
from expert_data.ppo import RunningStats, policy_action  # noqa: E402
from expert_data.tracking import AXES, masked_label_consistency  # noqa: E402
from expert_data.velocity import realised_command  # noqa: E402

LIMIT_TOL = 1e-3  # m/s: v_ref further than this from the command = rate limit active


def _consistency(a, b, mask, max_lag):
    r0, lag, rb = masked_label_consistency(a, b, mask, max_lag)
    return {
        "r_lag0": dict(zip(AXES[:3], r0[:3].round(3).tolist(), strict=True)),
        "best_lag_steps": dict(zip(AXES[:3], lag[:3].tolist(), strict=True)),
        "r_best_lag": dict(zip(AXES[:3], rb[:3].round(3).tolist(), strict=True)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-episodes", type=int, default=512)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")

    ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt["config"]
    known = {f.name for f in dataclasses.fields(EnvConfig)}
    env_cfg = EnvConfig(
        **{k: v for k, v in {**cfg["env"], "n_envs": args.n_episodes}.items() if k in known}
    )
    env = GateVelocityEnv(env_cfg, device=args.device)
    squash = bool(cfg["net"].get("squash", False))
    network = ActorCritic(env.action_dim, tuple(cfg["net"]["hidden"]), cfg["net"]["log_std_init"])
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    stats = RunningStats(
        mean=jnp.asarray(ckpt["obs_stats"]["mean"]),
        var=jnp.asarray(ckpt["obs_stats"]["var"]),
        count=jnp.float32(ckpt["obs_stats"]["count"]),
    )
    env_params = EnvParams.from_config(env_cfg)
    max_steps = int(
        math.ceil(float(env.max_steps(jnp.array(env_cfg.n_gates), env_params.target_speed_min)))
    )
    state = env.init_state(jax.random.key(args.seed))
    state, obs = env.reset(state, env_params)

    @jax.jit
    def rollout(state, obs):
        def body(carry, _):
            state, obs, first_done = carry
            mean, _, _ = network.apply(params, stats.normalize(obs))
            pos0 = state.sim.states.pos[:, 0]
            yaw0 = yaw_from_quat(state.sim.states.quat[:, 0])
            v_ref0 = state.ref.vel
            state_next, obs, _, terminated, truncated, _, info = env.step(
                state, policy_action(mean, squash), env_params
            )
            # The pre-reset reference is not in info; recompute it from the step's own state.
            done = terminated | truncated
            out = {
                "alive": ~first_done,
                "t": state.t,
                "cmd": info["cmd"][:, :3],
                "yaw0": yaw0,
                "v_ref0": v_ref0,
                "realised": realised_command(
                    pos0, info["pos"], yaw0, yaw_from_quat(info["quat"]), env.grid.dt
                )[:, :3],
            }
            return (state_next, obs, first_done | done), out

        _, out = jax.lax.scan(body, (state, obs, jnp.zeros((env.n_envs,), bool)), None, max_steps)
        return out

    out = jax.device_get(rollout(state, obs))
    alive = np.asarray(out["alive"])  # (T, N)
    cmd = np.asarray(out["cmd"])  # heading frame, physical units
    realised = np.asarray(out["realised"])
    yaw0 = np.asarray(out["yaw0"])
    t = np.asarray(out["t"])
    # v_ref at the start of decision k+1 is the end-of-decision reference of decision k (same
    # episode); the last alive decision of each episode has no successor and is dropped.
    v_ref_start = np.asarray(out["v_ref0"])
    v_ref_end = np.concatenate([v_ref_start[1:], v_ref_start[-1:]], axis=0)
    valid = alive & np.concatenate([alive[1:], np.zeros_like(alive[:1])], axis=0)
    v_ref_mean_h = np.asarray(
        world_to_heading(jnp.asarray(0.5 * (v_ref_start + v_ref_end)), jnp.asarray(yaw0))
    )
    v_ref_end_h = np.asarray(world_to_heading(jnp.asarray(v_ref_end), jnp.asarray(yaw0)))
    gap = np.abs(cmd - v_ref_end_h)  # (T, N, 3)
    limited = gap > LIMIT_TOL
    after_1s = valid & (t >= env_cfg.decision_freq)
    n_valid, n_after = max(int(valid.sum()), 1), max(int(after_1s.sum()), 1)

    def frac(mask_axis, base, n):
        return dict(
            zip(
                AXES[:3],
                ((mask_axis & base[..., None]).sum((0, 1)) / n).round(3).tolist(),
                strict=True,
            )
        )

    # Variance of the command within episodes vs in total (pooled over valid decisions).
    m = valid[..., None]
    total_var = ((cmd - (cmd * m).sum((0, 1)) / n_valid) ** 2 * m).sum((0, 1)) / n_valid
    ep_mean = (cmd * m).sum(0) / np.maximum(m.sum(0), 1)
    within_var = ((cmd - ep_mean[None]) ** 2 * m).sum((0, 1)) / n_valid
    unlimited = valid[..., None] & ~limited
    # Chatter: steps d_t = a_t - a_{t-1} inside the episode, and sign flips of consecutive steps.
    d = cmd[1:] - cmd[:-1]
    d_ok = (valid[1:] & valid[:-1])[..., None]
    flip_ok = d_ok[1:] & d_ok[:-1] & (np.abs(d[1:]) > 1e-3) & (np.abs(d[:-1]) > 1e-3)
    flips = (np.sign(d[1:]) != np.sign(d[:-1])) & flip_ok
    # Power above 5 Hz, per episode over its valid decisions (mean removed), pooled.
    f_hi = []
    for i in range(cmd.shape[1]):
        x = cmd[valid[:, i], i]
        if x.shape[0] >= 30:
            spec = np.abs(np.fft.rfft(x - x.mean(0), axis=0)) ** 2
            freq = np.fft.rfftfreq(x.shape[0], d=1.0 / env_cfg.decision_freq)
            f_hi.append(spec[freq > 5.0].sum(0) / np.maximum(spec[1:].sum(0), 1e-12))
    result = {
        "checkpoint": _relative(args.ckpt),
        "checkpoint_sha256_16": checkpoint_hash(args.ckpt),
        "n_episodes": args.n_episodes,
        "valid_decisions": int(valid.sum()),
        "cmd_to_realised": _consistency(cmd, realised, valid, args.max_lag),
        "cmd_to_vref_mean": _consistency(cmd, v_ref_mean_h, valid, args.max_lag),
        "vref_mean_to_realised": _consistency(v_ref_mean_h, realised, valid, args.max_lag),
        "rate_limit_active": {
            "all_decisions": frac(limited, valid, n_valid),
            "after_first_second": frac(limited, after_1s, n_after),
        },
        "cmd_to_realised_after_first_second": _consistency(cmd, realised, after_1s, args.max_lag),
        "cmd_to_realised_limit_inactive_per_axis": {
            ax: _consistency(cmd, realised, unlimited[..., i], args.max_lag)["r_best_lag"][ax]
            for i, ax in enumerate(AXES[:3])
        },
        "cmd_variance_within_episode_share": dict(
            zip(
                AXES[:3], (within_var / np.maximum(total_var, 1e-12)).round(3).tolist(), strict=True
            )
        ),
        "cmd_std": dict(zip(AXES[:3], np.sqrt(total_var).round(3).tolist(), strict=True)),
        "chatter": {
            "mean_abs_step_m_s": dict(
                zip(
                    AXES[:3],
                    ((np.abs(d) * d_ok).sum((0, 1)) / max(int(d_ok.sum()), 1)).round(3).tolist(),
                    strict=True,
                )
            ),
            "step_sign_flip_rate": dict(
                zip(
                    AXES[:3],
                    (flips.sum((0, 1)) / np.maximum(flip_ok.sum((0, 1)), 1)).round(3).tolist(),
                    strict=True,
                )
            ),
            "power_share_above_5hz": dict(
                zip(AXES[:3], np.mean(f_hi, axis=0).round(3).tolist(), strict=True)
            )
            if f_hi
            else None,
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
