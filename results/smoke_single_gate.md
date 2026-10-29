# Smoke runs: single gate, velocity action space (2026-10-29)

Command: `uv run expert-train experiment=single_gate` (512 worlds, 10 M env steps, seed 0, RTX 2080
shared with another training job, crazyflow 0.4.0, 480/120/30 Hz). Two policy distributions:

- **clip**: the RPM expert's Gaussian, clipped to `[-1, 1]` by the env (`net.squash=false`; the
  first run predates the key). Record: `outputs/single_gate/2026-10-29_01-48-21/`.
- **tanh**: tanh-squashed Gaussian, the default since commit "Tanh-squashed Gaussian policy".
  Record: `outputs/single_gate_tanh/2026-10-29_02-11-52/`.

All numbers are tier-1. "Saturated" means `|a| >= 0.99` in normalised units (for clip this is the
bound itself). The deterministic rows use the mean action (`tanh(mu)` or `clip(mu)`) on 512 worlds,
first episode each, DR on, from a scratch evaluation script (the evaluation module is step 1 of
`docs/plan.md`).

| Quantity | clip | tanh |
|---|---|---|
| Wall-clock (incl. compilation) | 1.3 min | 1.2 min |
| Throughput (decisions/s) | ≈ 190–200 k | ≈ 170–190 k |
| First iteration with training-rollout success ≥ 95 % | 175 (2.87 M steps, 36 s) | 127 (2.08 M steps, 30 s) |
| Final training-rollout success (stochastic) | 1.00 | 1.00 |
| Deterministic success | 512/512 (Wilson 95 %: 99.3–100 %) | 512/512 (99.3–100 %) |
| Mean speed while flying | 1.26 m/s | 1.24 m/s |
| Deterministic saturation, `|a| >= 0.99` (vx, vy, vz, ω_z) | 98.8, 56.0, 2.5, 1.3 % | 36.0, 0.0, 0.0, 0.0 % |
| Deterministic `|a| >= 0.95` (vx, vy, vz, ω_z) | 99.0, 57.3, 2.9, 1.7 % | 74.3, 8.4, 0.0, 0.0 % |
| Deterministic `|a|` median (vx, vy, vz, ω_z) | 1.0, 1.0, 0.14, 0.32 | 0.98, 0.66, 0.23, 0.29 |
| Stochastic training-rollout saturation, last iteration | ≈ 40 % (old definition `|u| >= 1`) | 9.6 % (vx 38.4 %, others ≤ 0.1 %) |

**Reading.** The tanh policy removes the labels that sit exactly on the bound and the lateral,
vertical and yaw saturation, and it learns slightly faster. The forward command still crowds the
`v^h_x` bound (median 0.98 of 1.5 m/s, 36 % of decisions ≥ 0.99): the progress reward pays for speed
and the policy flies as fast as the bound allows. The spec's QC limit (< 2 % saturated) is therefore
still not met for `v^h_x`; that is a reward / speed-range question, not a distribution question
(see `docs/plan.md`, "Decisions needed before step 7").
