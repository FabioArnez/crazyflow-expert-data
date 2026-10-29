# Smoke runs: single gate, velocity action space (2026-10-29)

Command: `uv run expert-train experiment=single_gate` (512 worlds, 10 M env steps, seed 0, RTX 2080
shared with another training job, crazyflow 0.4.0, 480/120/30 Hz). Three configurations, each a
change on top of the previous one:

- **clip**: the RPM expert's Gaussian, clipped to `[-1, 1]` by the env (`net.squash=false`; the
  first run predates the key). Record: `outputs/single_gate/2026-10-29_01-48-21/`.
- **tanh**: tanh-squashed Gaussian (`net.squash=true`). Record:
  `outputs/single_gate_tanh/2026-10-29_02-11-52/`.
- **tanh + target speed**: per-episode `v_target ~ U[0.4, 1.2]` m/s in the observation and the
  reward term `−0.5 (|v| − v_target)²`; the current default. Record:
  `outputs/single_gate_vtarget/2026-10-29_02-24-41/`.

All numbers are tier-1. "Saturated" means `|a| >= 0.99` in normalised units (for clip this is the
bound itself). The deterministic rows use the mean action (`tanh(mu)` or `clip(mu)`) on 512 worlds,
first episode each, DR on, from a scratch evaluation script (the evaluation module is step 1 of
`docs/plan.md`); Wilson 95 % intervals in brackets.

| Quantity | clip | tanh | tanh + target speed |
|---|---|---|---|
| Wall-clock (incl. compilation) | 1.3 min | 1.2 min | 1.2 min |
| Throughput (decisions/s) | ≈ 190–200 k | ≈ 170–190 k | ≈ 165–190 k |
| First iteration with training-rollout success ≥ 95 % | 175 (2.87 M steps) | 127 (2.08 M) | 216 (3.54 M) |
| Deterministic success | 512/512 [99.3, 100] | 512/512 [99.3, 100] | 510/512 [98.6, 99.9] |
| Mean speed while flying | 1.26 m/s | 1.24 m/s | 0.77 m/s |
| Deterministic `|a| >= 0.99` (vx, vy, vz, ω_z) | 98.8, 56.0, 2.5, 1.3 % | 36.0, 0, 0, 0 % | 0, 0, 0, 0 % |
| Deterministic `|a| >= 0.95` (vx, vy, vz, ω_z) | 99.0, 57.3, 2.9, 1.7 % | 74.3, 8.4, 0, 0 % | 3.2, 0, 3.9, 0.1 % |
| Deterministic `|a|` median (vx, vy, vz, ω_z) | 1.0, 1.0, 0.14, 0.32 | 0.98, 0.66, 0.23, 0.29 | 0.61, 0.14, 0.17, 0.11 |
| Stochastic training-rollout saturation, last iteration | ≈ 40 % (`|u| >= 1`) | 9.6 % (vx 38.4 %) | 0.1 % |

**Target-speed tracking** (deterministic, 512 episodes per row; speed = mean `|v|` over the
episode, which includes the acceleration from hover):

| Target | Success | Mean speed | Speed − target | Mean episode length | `|a| >= 0.99` (any axis) |
|---|---|---|---|---|---|
| U[0.4, 1.2] | 510/512 [98.6, 99.9] | 0.77 m/s | −0.011 m/s (mean `|err|` 0.066; r = 0.985 per episode) | 71.8 decisions | 0 % |
| 0.4 m/s | 512/512 [99.3, 100] | 0.52 m/s | +0.117 m/s | 105.2 | 0 % |
| 0.8 m/s | 512/512 [99.3, 100] | 0.79 m/s | −0.006 m/s | 69.4 | 0 % |
| 1.2 m/s | 501/512 [96.2, 98.8] | 1.05 m/s | −0.154 m/s | 53.8 | 0.3 % (vz) |

**Reading.** The tanh distribution removed the labels on the bound and the lateral, vertical and
yaw saturation, but the progress reward still paid for speed, so the forward command crowded its
bound. The per-episode target speed removes that incentive: no axis reaches `|a| >= 0.99` in the
deterministic policy, and the spec's QC limit (< 2 % saturated) is met with a wide margin. The
expert also tracks the requested speed (r = 0.985), which gives the dataset the speed diversity that
speed words in the instructions need. Costs: about 70 % more samples to reach 95 % in training, and
the fastest target loses 2 % success (11 crashes) while realising 1.05 of 1.2 m/s on these short
single-gate episodes; at 0.4 m/s the expert flies 0.12 m/s too fast. Confirm on mixed multi-gate
tracks (5 seeds) before collection.

## Command-gap penalty (2026-10-29, morning)

The target-speed expert dithered its forward command at 15 Hz (`docs/plan.md`, "Command chatter").
Fix: reward term `−k_cmd_gap |R_ψ v^h − v_ref|²` with `k_cmd_gap = 1` (the part of the command the
rate-limited reference cannot reach in one decision) and `k_action_rate` 0.01 → 0.1. Record:
`outputs/single_gate_gap/2026-10-29_09-36-16/`; `results/eval_single_gate_gap.json` and
`results/label_consistency_single_gate_gap.json`. Deterministic, 512 episodes, tier-1.

| Quantity (vx, vy, vz) | target speed only | + command-gap penalty | Acceptance |
|---|---|---|---|
| Consecutive steps that flip sign | 82, 70, 58 % | 3.8, 3.2, 4.0 % | < 30 % |
| Command power above 5 Hz | 62, 28, 8 % | 2.8, 3.6, 3.5 % | < 5 % |
| Mean command step per decision | 0.40, 0.11, 0.08 m/s | 0.018, 0.017, 0.010 m/s | — |
| Rate limit active after the first second | 97, 95, 71 % | 0, 0, 0.4 % | — |
| eq. (3) r at best lag (lag, decisions) | 0.58 (22), 0.69 (7), 0.87 (9) | 0.981 (4), 0.980 (1), 0.944 (2) | ≥ 0.95 |
| Reference → realised r (lag) | 0.976 (1), 0.973 (2), 0.954 (1) | 0.989 (2), 0.987 (1), 0.951 (0) | — |
| Success | 510/512 | 508/512 [98.0, 99.7] | ≥ 95 % |
| `|a| >= 0.99`, any axis | 0 % | 0 % | < 2 % |
| Speed vs target, per-episode r | 0.985 | 0.966 | — |
| First iteration with training success ≥ 95 % | 216 (3.5 M steps) | 484 (7.9 M steps) | — |

The chatter is gone and two of the three criteria pass on every axis. eq. (3) passes for vx, vy
(and ω_z: 0.959) and misses for vz by 0.006; the controller itself tracks a feasible vz reference
with r = 0.951, so the remaining vz gap is a property of the z loop, not of the commands. Cost:
about 2.2× more samples to learn the single gate.
