# Mixed-track velocity expert, 5 seeds (2026-10-29)

**Training.** `uv run expert-train experiment=mixed_tracks run.seed=0` and
`uv run expert-train -m experiment=mixed_tracks run.seed=1,2,3,4`: from scratch, 1,024 worlds,
50 M env steps, 3–5 gates per track, chains (gate normals ±0.3 rad off the path) and imperfect
circles (30–72° per gate) at 50/50, yaw-range curriculum, target speed U[0.4, 1.2] m/s, tanh
policy, command-gap penalty. RTX 2080, crazyflow 0.4.0, 480/120/30 Hz. All numbers tier-1.

| Seed | Record | Wall-clock | First iteration with training success ≥ 95 % | Training success, last 10 iterations |
|---|---|---|---|---|
| 0 | `outputs/mixed_tracks/2026-10-29_09-38-52/` | 3.0 min | 411 (13.5 M steps) | 99.4 % |
| 1 | `outputs/multirun/mixed_tracks/2026-10-29_09-42-20/seed_1/` | 3.3 min | 362 (11.9 M) | 99.9 % |
| 2 | `…/seed_2/` | 3.1 min | 430 (14.1 M) | 99.5 % |
| 3 | `…/seed_3/` | 3.1 min | 431 (14.1 M) | 99.8 % |
| 4 | `…/seed_4/` | 3.1 min | 550 (18.0 M) | 99.3 % |

Throughput ≈ 320 k decisions/s at 1,024 worlds.

**Evaluation.** `scripts/eval_seeds.sh results/mixed_tracks/eval <run dirs>`, summarised by
`scripts/summarise_seeds.py results/mixed_tracks/eval`. Deterministic policy, DR on, random target
speed. Success: 1,024 episodes per condition, all 5 gates required, training noise levels (chains
±0.3 rad, circles 5–12 gates per turn). Label checks (`scripts/analyse_label_consistency.py`):
512 episodes of the training distribution. Speed r: per-episode mean speed vs target speed, chains.

| Seed | 5-gate chains, success % [Wilson 95 %] | 5-gate circles, success % | max `|a|>=0.99` % | speed r | sign flips % (vx, vy, vz) | power > 5 Hz % (vx, vy, vz) | eq. (3) r (lag) (vx, vy, vz) |
|---|---|---|---|---|---|---|---|
| 0 | 99.8 [99.3, 99.9] | 99.5 [98.9, 99.8] | 0.0 | 0.998 | 5.6, 5.2, 2.6 | 4.1, 1.5, 0.8 | 0.984 (3), 0.985 (1), 0.952 (2) |
| 1 | 99.8 [99.3, 99.9] | 99.9 [99.4, 100.0] | 0.0 | 0.998 | 5.8, 5.5, 2.4 | 4.5, 2.0, 0.7 | 0.987 (3), 0.982 (2), 0.954 (2) |
| 2 | 99.8 [99.3, 99.9] | 99.9 [99.4, 100.0] | 0.0 | 0.999 | 9.0, 4.5, 3.1 | 4.9, 0.9, 0.9 | 0.989 (2), 0.980 (3), 0.953 (2) |
| 3 | 100.0 [99.6, 100.0] | 100.0 [99.6, 100.0] | 0.0 | 0.999 | 10.7, 6.0, 2.3 | 4.5, 1.0, 0.7 | 0.988 (2), 0.979 (3), 0.956 (2) |
| 4 | 99.7 [99.1, 99.9] | 99.5 [98.9, 99.8] | 0.0 | 0.997 | 7.9, 4.6, 2.6 | 4.6, 0.8, 0.8 | 0.988 (2), 0.975 (3), 0.949 (2) |

Mean over 5 seeds: chains 99.8 % (min 99.7), circles 99.8 % (min 99.5); worst sign-flip rate 10.7 %, worst power above 5 Hz 4.9 %, lowest eq. (3) r 0.949.

**Acceptance** (from the single-gate chatter fix, `docs/plan.md`):

| Criterion | Result over 5 seeds |
|---|---|
| Success ≥ 95 % (spec §6 action 1) | 99.5–100 % on chains and circles |
| Saturated commands (`|a| >= 0.99`) < 2 % (spec §3.9) | 0 % |
| Consecutive command steps that flip sign < 30 % | 2.3–10.7 % |
| Command power above 5 Hz < 5 % | 0.7–4.9 % (seed 2, vx: 4.9 %) |
| eq. (3) r ≥ 0.95 at the controller's lag | vx 0.984–0.989, vy 0.975–0.985; vz 0.949–0.956 (seed 4 misses by 0.001) |

Other observations:
- The rate limit is active in ≤ 1.6 % of decisions after the first second (single-gate expert
  before the fix: 71–97 %).
- The controller itself follows the feasible reference with r = 0.952–0.958 on vz, so vz cannot
  pass r ≥ 0.95 with a large margin by changing the commands.
- The expert flies 0.06–0.09 m/s faster than the target on average; failures concentrate at the
  fastest target bin (0.93–1.2 m/s: 98.8–100 % success).
- Comparison with the RPM expert retrained on 0.4.0 (sibling branch of crazyflow-gate-ppo,
  tier-1: 94.3 % on 5-gate chains, 92.6 % on full 8-gate circles): the velocity expert succeeds
  more often but flies 3–10× slower (0.4–1.2 m/s vs ≈ 4 m/s measured on 0.3.2), and the test
  conditions are not the same, so this is not a like-for-like comparison.
- Not yet tested: longer chains (7, 10 gates), gates per turn outside the training range,
  stochastic policy and DART noise (needed before collection, spec §3.8).
