# Smoke run: single gate, velocity action space (2026-10-29)

Run: `uv run expert-train experiment=single_gate` (512 worlds, 10 M env steps, seed 0, RTX 2080
shared with another training job, crazyflow 0.4.0, 480/120/30 Hz). Record:
`outputs/single_gate/2026-10-29_01-48-21/` (config, metrics, checkpoint). All numbers tier-1.

| Quantity | Value |
|---|---|
| Wall-clock | 1.3 min (75.6 s incl. compilation) |
| Throughput | ≈ 190–200 k env steps/s (decisions/s) at 512 worlds |
| First iteration with training-rollout success ≥ 95 % | 175 (2.87 M steps, 36 s) |
| Final training-rollout success (stochastic, DR on) | 1.00 |
| Deterministic success, 512 worlds, first episode each (DR on) | 512/512 (Wilson 95 %: 99.3–100 %) |
| Mean speed while flying | 1.26 m/s |
| Deterministic action saturation (vx, vy, vz, ω_z) | 98.7 %, 55.7 %, 2.5 %, 1.2 % |

The loop learns. The saturation is the open issue for the data expert (see `docs/plan.md`,
"Decisions needed before step 7"). The deterministic numbers come from a scratch evaluation script;
the evaluation module is step 1 of the plan.
