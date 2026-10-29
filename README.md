# crazyflow-expert-data

Velocity-setpoint PPO expert and (later) the demonstration pipeline for Crazyflie gate flight in
[crazyflow](https://github.com/learnsyslab/crazyflow) 0.4.0. This is the data-collection expert of
UAVLA, a vision-language-action policy for UAV gate navigation. The expert acts in the same space as
the VLA, so the demonstration label is exactly the command the expert issued ("Option A"):

    a_t = (v^h_x, v^h_y, v^h_z, ω_z)    heading-frame velocity (x forward, y left, z up) + yaw rate, 30 Hz

The data it produces is meant for VLA fine-tuning (LeRobot v3.0) and for world-model training.

**Provenance.** The track samplers, the geometric gate test, the reward, the PPO core (PureJaxRL
pattern with truncation bootstrapping and running observation normalisation), the networks and the
checkpointing are copied and adapted from the motor-RPM gate expert
[crazyflow-gate-ppo](https://github.com/FabioArnez/crazyflow-gate-ppo) (crazyflow 0.3.2). There is
no import across the two repositories; each ported module says what changed in its docstring.

## Status (2026-10-29)

| Part | State |
|---|---|
| Velocity-command layer on `Control.state` (`velocity.py`) | done, tested |
| Batched gate environment with velocity actions (`env.py`) | done, tested |
| PPO, networks, checkpointing, Hydra training (`ppo.py`, `train.py`) | ported, tested |
| Tanh-squashed policy and per-episode target speed (defaults) | done, tested; no saturated commands |
| Single-gate smoke runs (clip, tanh, tanh + target speed) | learn: ≥ 510/512 deterministic successes after 10 M steps, ~1.2 min |
| Evaluation (`expert-eval`): success + Wilson CI, saturation, speed tracking, eq. (3) | done, tested |
| Command-gap penalty (removes 15 Hz command chatter) | done, tested |
| Mixed-track expert, 5 seeds | done: 99.5–100 % on 5-gate chains and circles (`results/mixed_tracks/`) |
| Scenarios, world loader, two-pass collection, QC, LeRobot export | planned in `docs/plan.md` |

All numbers in this README are tier-1 (measured on the development laptop, RTX 2080 / CPU).

## Results so far

**Velocity tracking** (`scripts/measure_tracking.py`, CPU, `results/tracking_*.json`; details and
the reasons for the design in `docs/crazyflow040_state_control.md`):

| | vx | vy | vz | ω_z |
|---|---|---|---|---|
| Step response, rise time 10–90 % | 0.33 s | 0.33 s | 0.33 s | 0.10 s |
| Steady-state error | +0.3 % | +0.3 % | +0.3 % | 0.0 % |
| Overshoot | 21 % | 21 % | 10 % | 39 % |
| Smooth random commands: lag of best correlation (decisions) | 4 | 4 | 2 | 1 |
| Correlation commanded vs realised (spec eq. 3) at that lag | 0.967 | 0.972 | 0.974 | 0.878 |

Zero command holds hover (3.3 cm steady sag from the controller's mass model). The spec's QC
criterion "r ≥ 0.95 at a lag of at most one decision" is **not** met in xy: the Mellinger cascade
lags by ≈ 133 ms on this airframe; `tests/test_velocity.py` keeps it as a strict `xfail`.

**Mixed-track expert** (`experiment=mixed_tracks`, 5 seeds, `results/mixed_tracks/README.md`):
trained from scratch in 3.0–3.3 min per seed (50 M steps, ≈ 320 k decisions/s at 1,024 worlds).
Deterministic success on 1,024 episodes per condition: 99.7–100 % on 5-gate chains and 99.5–100 % on
5-gate circles (30–72° per gate), at target speeds of 0.4–1.2 m/s. Label quality: no saturated
commands; consecutive command steps flip sign in ≤ 10.7 % of decisions and ≤ 4.9 % of the command
power is above 5 Hz; commanded vs realised velocity (spec eq. 3) r = 0.975–0.989 in xy and
0.949–0.956 in z at lags of 1–3 decisions (the z loop itself reaches 0.952–0.958).

**Single-gate smoke runs** (`results/smoke_single_gate.md`) traced the label problems that the
defaults now fix: the clipped Gaussian put 98.8 % of forward commands on the bound; the tanh policy
still crowded it; the target speed removed saturation but the expert then dithered its command at
15 Hz; the command-gap penalty removed the dither.

## Install and run

```bash
uv sync                                    # Python 3.12, crazyflow[gpu]==0.4.0 pinned
uv run pytest                              # CPU, ~90 s; tests/conftest.py forces JAX_PLATFORMS=cpu
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run expert-train experiment=single_gate # smoke run, ~1.3 min on an RTX 2080
uv run expert-train -m experiment=mixed_tracks run.seed=0,1,2,3,4   # the expert, ~3 min per seed
scripts/eval_seeds.sh results/mixed_tracks/eval <run_dir> [<run_dir> ...]  # chains, circles, labels
uv run python scripts/summarise_seeds.py results/mixed_tracks/eval
uv run expert-eval --ckpt outputs/<run>/<timestamp>/checkpoint.pkl --track-type circle \
    --n-episodes 1024 --json results/eval_<name>.json   # --target-speed, --stochastic, --save-traj
uv run python scripts/measure_tracking.py --grid 480 120 30 --out results/tracking.json
```

Any config key is a Hydra override (`env.n_envs=256 run.device=cpu ppo.total_env_steps=1e7`).
Outputs land in `outputs/<run.name>/<timestamp>/`; `train.py` sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so the GPU can be shared.

## Design

**Rates.** Physics 480 Hz, state controller 120 Hz, decision 30 Hz (16 physics and 4 controller
ticks per decision); attitude and force-torque stages at 480 Hz. `RateGrid` rejects non-integer
nestings.

**From command to setpoint** (`velocity.py`). crazyflow 0.4.0 has no velocity mode; `Control.state`
takes a 16-D setpoint `[p, v, a, quat, body rates]` for its Mellinger controller. The command is
rotated to the world frame with the measured yaw and becomes a carried reference that is advanced at
every controller tick: the reference velocity moves towards the command at ≤ 2 m/s² (xy) / 1 m/s² (z)
with acceleration feed-forward, the position integrates it, the yaw integrates `ω_z` with a body-rate
feed-forward, and the reference is leashed to 0.3 m / 0.3 rad around the drone. The spec's eq. (2)
(re-anchoring at the measured position every decision) sinks 0.45 m in 5 s at zero command; it is
kept as `anchor="measured"` for comparison.

**Environment** (`env.py`). Functional and batched (`EnvState` pytree around crazyflow's `SimData`,
same-step autoreset, `lax.scan` rollouts). Tracks: chains, imperfect circles and the `mixed` family
(default). Gate passage is the geometric plane-crossing test (spec eq. 13), not an MJX contact query.
Observation (45-D): the RPM expert's 37-D privileged observation (body-frame state, previous action,
current and next gate, gates remaining capped at `obs_remaining_cap`), the 7-D command reference
(`p_ref − p`, `v_ref`, `ψ_ref − ψ`) and the episode's target speed. Action bounds:
`±1.5 / ±1.0 / ±0.75` m/s and `±1.5` rad/s. Reward as in the RPM expert (progress, gate bonus, crash
and miss penalties, rate, action-rate and heading terms) plus `−k_speed (|v| − v_target)²` and
`−k_cmd_gap |R_ψ v^h − v_ref|²`, which makes the expert command only what the rate-limited reference
can reach (without it the expert dithered its command at 15 Hz; `results/smoke_single_gate.md`).
Domain randomisation per world: mass ±10 %, inertia ±10 %, thrust curve ±5 %.

**Target speed.** Each episode draws `v_target ~ U[0.4, 1.2]` m/s (`env.target_speed_min/max`;
`EnvParams.from_config(target_speed=…)` fixes it for evaluation). Without it the progress reward pays
for speed and the policy pins the forward command at its bound; with it the expert flies at the
requested speed, which also gives the dataset speed diversity for instructions such as "fly slowly".
The episode budget is `time_base + n_gates · time_slack · spacing_max / v_target`.

**Policy** (`ppo.py`). 2×256 tanh actor-critic. The action is `tanh(u)` of a Gaussian sample `u`
(`net.squash=true`, default): commands approach the bounds without sitting on them. The log-density
includes the tanh change of variables, and the entropy bonus is the reparameterised estimate
`H(u) + E[Σ log(1 − tanh²u)]`, which penalises a mean deep in saturation. `net.squash=false` gives
the RPM expert's clipped Gaussian. Rollouts report `rollout/saturation_a{0..3}` (`|a| >= 0.99`)
and `rollout/speed_error`.

**Phase-1 lessons kept.** Rotors are pre-spun *exactly* at each world's hover speed at reset; the
"gates remaining" observation is capped; the default track family is `mixed`.

## Layout

```
src/expert_data/
  velocity.py   rate grid, CommandConfig, velocity -> Control.state setpoints, reset helper, eq. (3)
  tracking.py   open-loop tracking runs and metrics (tests, scripts/measure_tracking.py)
  env.py        GateVelocityEnv: batched, functional, autoreset
  geometry.py   quaternions, heading frame, gate crossing test       (ported)
  track.py      chain / circle track samplers, start states           (ported)
  ppo.py        PPO iteration in one jax.jit, tanh-squashed policy   (ported + new)
  networks.py   2x256 actor-critic                                    (ported)
  checkpoint.py pickle checkpoints + content hash                     (ported)
  config.py     dataclass schema for Hydra
  train.py      expert-train entry point                              (ported)
  evaluate.py   expert-eval: outcomes, saturation, speed, eq. (3)     (ported + new)
configs/        config.yaml, experiment/{single_gate,mixed_tracks}.yaml
docs/           crazyflow040_state_control.md (interface findings), plan.md (next modules)
results/        tracking measurements, smoke-run summary
outputs/        run records (config, metrics, checkpoint)
```

## Gotchas (crazyflow 0.4.0)

- Always pass `drone="cf2x_L250"` (the `Sim` default is a different drone). Its model mass is now
  32.8 g (hover 19,217 RPM; 0.3.2: 31.9 g, 18,968 RPM); the Mellinger controller assumes 29 g.
- The rotor command is clipped to [8,398, 23,216] RPM: thrust-to-weight ≈ 1.49. The Mellinger
  position stage has no tilt limit; infeasible references saturate the motors and can tumble the
  drone. Keep the acceleration limits of `CommandConfig`.
- `Sim(..., control_freq=...)` does not exist: use `state_freq`, `attitude_freq`,
  `force_torque_freq`. A staged command is applied at the next tick of its stage.
- Dynamics parameters have no world axis by default; `GateVelocityEnv.init_state` broadcasts them to
  `(N, 1, ...)` for per-world randomisation. `crazyflow.dynamics.load_params` takes the `Dynamics`
  enum.
- After a reset, the controller memory must be restored too (integral errors, last tick = −1);
  `velocity.reset_drones` does it.
- Import `expert_data` (or crazyflow) before scipy: crazyflow needs `SCIPY_ARRAY_API=1`.
