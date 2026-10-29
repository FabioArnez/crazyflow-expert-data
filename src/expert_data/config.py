"""Typed configuration schema (registered with Hydra's ConfigStore in ``train.py``).

Values here are the defaults; ``configs/config.yaml`` and ``configs/experiment/*.yaml`` override
them. Anything that changes *during* training (curriculum, domain randomisation ranges) is mirrored
into ``env.EnvParams`` so it can be traced by JAX without recompiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # --- simulator and rate grid (physics : state controller : decision, spec §1.5) -------------
    drone: str = "cf2x_L250"
    sim_freq: int = 480
    state_freq: int = 120
    decision_freq: int = 30
    n_envs: int = 1024  # crazyflow worlds stepped in one fused kernel
    # --- action: a in [-1, 1]^4 -> (v^h_x, v^h_y, v^h_z, omega_z) = a * max (spec eq. 1) ---------
    v_max_x: float = 1.5  # m/s, heading-frame forward / backward
    v_max_y: float = 1.0  # m/s, heading-frame left / right
    v_max_z: float = 0.75  # m/s
    yaw_rate_max: float = 1.5  # rad/s
    # --- velocity command -> Control.state setpoints (velocity.CommandConfig) ---------------------
    cmd_anchor: str = "integrated"
    cmd_acc_max_xy: float = 2.0  # m/s^2
    cmd_acc_max_z: float = 1.0  # m/s^2
    cmd_leash: float = 0.3  # m
    cmd_yaw_leash: float = 0.3  # rad
    # --- track sampling (a chain of n_gates + 1 upright square gates) ---------------------------
    n_gates: int = 5  # max active gates per episode; sizes the gate arrays (static)
    n_gates_min: int = 3  # active gates per world ~ U{n_gates_min, ..., n_gates}
    spacing_min: float = 1.2  # m between consecutive gates
    spacing_max: float = 2.2
    yaw_range: float = 0.6  # max |heading change| between consecutive gates (rad)
    gate_yaw_noise: float = 0.0  # chain: gate normal = segment heading + U(+-this) (rad)
    dz_max: float = 0.3  # max altitude change between consecutive gates (m)
    z_min: float = 0.7  # gate altitude clipped to [z_min, z_max]
    z_max: float = 2.2
    start_z_min: float = 0.8
    start_z_max: float = 1.4
    # --- track family: "chain" (random-walk polyline), "circle" (imperfect circle), "mixed" ------
    track_type: str = "mixed"
    circle_prob: float = 0.5  # share of circle tracks in "mixed"
    circle_gates_per_turn_min: float = 8.0  # heading change per gate = 2*pi / gates_per_turn
    circle_gates_per_turn_max: float = 12.0
    circle_radius_noise: float = 0.1
    circle_angle_noise: float = 0.1
    circle_yaw_noise: float = 0.1
    start_xy_jitter: float = 0.2
    start_yaw_noise: float = 0.3
    start_vel_max: float = 0.3
    start_tilt_max: float = 0.1
    start_ang_vel_max: float = 0.2  # rad/s
    # --- gate geometry (drone-data-collect gate_14.urdf; cf2x_L250 collision sphere) -----------
    half_opening: float = 0.245  # inner opening is 0.49 m square
    frame_thickness: float = 0.06
    drone_radius: float = 0.07
    # --- termination bounds --------------------------------------------------------------------
    z_floor: float = 0.05
    z_ceiling: float = 3.5
    max_gate_dist: float = 6.0
    max_tilt: float = 1.2  # rad; beyond this the state controller has lost the drone
    # --- target speed: one per episode, observed and tracked (speed-conditioned expert) -------
    # Without it the progress reward pays for speed and the policy pins v^h_x at its bound.
    # The range stays below v_max_x so that the forward command keeps headroom.
    target_speed_min: float = 0.4  # m/s
    target_speed_max: float = 1.2  # m/s
    # --- episode length: max_steps = (time_base + n_active * time_slack * spacing_max / v_target)
    #     * decision_freq, so that slow episodes get proportionally more time ------------------
    time_base: float = 2.0  # s
    time_slack: float = 1.5
    # The "gates remaining" observation is clipped here so that chains longer than the training
    # tracks stay in-distribution (Phase-1 lesson). Keep it at the largest training n_gates.
    obs_remaining_cap: int = 5
    # --- reward (Song et al., 2021 style, as the RPM expert, plus the target-speed term) ---------
    k_progress: float = 1.0  # per metre of progress towards the current gate
    gate_bonus: float = 5.0
    crash_penalty: float = 10.0  # floor / ceiling / frame hit / out of bounds / tilt
    miss_penalty: float = 10.0  # crossed the gate plane outside the frame
    k_ang_vel: float = 2e-3  # * |omega|^2
    k_action_rate: float = 0.1  # * |a_t - a_{t-1}|^2 (normalised actions; RPM expert: 0.01)
    k_yaw: float = 1e-2  # * (1 - cos(heading error to current gate)); keeps the camera on the gate
    # * (|v| - v_target)^2; with k_progress = 1 the optimum is v_target + 1/(2 k f) ~ +0.03 m/s.
    k_speed: float = 0.5
    # * |R_psi v^h - v_ref|^2 (m/s) at the end of the decision: commanding more than the
    # rate-limited reference can reach in one decision costs reward. Without it the expert
    # dithered its command at 15 Hz around the reference (docs/plan.md, "Command chatter").
    k_cmd_gap: float = 1.0
    # --- domain randomisation (+- fraction, uniform, per world, resampled at every reset) ------
    dr_mass: float = 0.1
    dr_inertia: float = 0.1
    dr_motor_gain: float = 0.05  # thrust-curve scale (spec §3.5 "motor constant")


@dataclass
class NetConfig:
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    log_std_init: float = -0.5  # of the pre-squash Gaussian when squash is on
    # Policy distribution: tanh-squashed Gaussian (True) or the RPM expert's clipped Gaussian
    # (False). Checkpoints without this key (the first smoke run) were trained with False.
    squash: bool = True


@dataclass
class PPOConfig:
    n_steps: int = 32  # rollout length per world; batch = n_envs * n_steps
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    anneal_lr: bool = True  # linear decay to 0 over the run (mitigates entropy collapse)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    max_grad_norm: float = 0.5
    normalize_obs: bool = True  # running Welford statistics carried in the train state
    normalize_adv: bool = True
    total_env_steps: int = 30_000_000


@dataclass
class CurriculumConfig:
    """Grow the gate-yaw range once the recent success rate clears a threshold."""

    enabled: bool = True
    yaw_range_init: float = 0.2
    yaw_range_step: float = 0.1
    success_threshold: float = 0.8
    window: int = 10  # iterations averaged for the decision


@dataclass
class RunConfig:
    name: str = "velocity_expert"
    seed: int = 0
    device: str = "gpu"  # "gpu" | "cpu"
    log_interval: int = 10  # iterations between console lines (metrics.jsonl gets every iteration)
    ckpt_interval: int = 100
    init_from: str | None = None  # checkpoint .pkl to warm-start params + obs stats from
    wandb: bool = False
    wandb_project: str = "uavla-expert-data"
    jax_cache: bool = True  # persistent XLA compilation cache (crazyflow.utils.enable_cache)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    run: RunConfig = field(default_factory=RunConfig)
