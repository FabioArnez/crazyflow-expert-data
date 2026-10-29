# Plan: from the velocity expert to the demonstration dataset

**Scope / Context:** Maps the remaining items of the UAVLA data-collection spec (2026-09-24, v0.1)
to future modules of this repository, in build order. It is a plan, not an implementation: none of
the modules below exists yet. What exists is listed in `README.md` (velocity-command layer,
batched gate environment, PPO, smoke run).

## Guiding constraints

- **Option A.** The label is the clean command `a*` that the expert issued, in the units of
  `info["cmd"]` (heading-frame m/s and rad/s). DART noise changes the executed command `u`, never
  the label (spec eq. 14).
- **One simulator build.** crazyflow 0.4.0, 480/120/30 Hz, `cf2x_L250`, the velocity layer of
  `velocity.py` with its `CommandConfig`. Every episode records the versions and the configuration
  needed to regenerate it: `(scenario_id, seed, crazyflow version, expert checkpoint hash,
  command config)` (spec §3.1 item 6; `checkpoint.checkpoint_hash` exists).
- **Two passes** (spec §3.6). Pass 1 is batched JAX dynamics without images; pass 2 renders only the
  successful episodes, one world per call, from logged states. The raw pass-1/pass-2 output is a
  format of this repository; the LeRobot dataset is one export of it. World-model training reads the
  raw format directly.
- **No JAX and PyTorch in the same process** for collection: the LeRobot writer runs in its own
  process (and environment, if the CUDA wheels conflict) and reads the raw files.

## Build order

| # | Module (planned) | Spec | Content | Done when |
|---|---|---|---|---|
| 1 | `evaluate.py` — **done** | §1.4, §3.9, §3.10 | `expert-eval`: deterministic and stochastic rollouts of a checkpoint on chain / circle families, success with Wilson 95 % intervals, per gate count and per target-speed bin, saturation per axis, speed tracking, label consistency (eq. 3) on the expert's own rollouts, `--save-traj`. | The mixed-track expert is evaluated on ≥ 1,000 episodes per family. |
| 2 | expert training — **done** | §1.4, §6 action 1 | `experiment=mixed_tracks`, 5 seeds (`results/mixed_tracks/README.md`): 99.5–100 % on 5-gate chains and circles, 0 % saturation, no chatter, eq. (3) r ≥ 0.975 in xy and 0.949–0.956 in z. Still to test: longer chains, turns outside the training range, the stochastic policy and DART noise. | ≥ 95 % deterministic success with DR on, per seed; saturation < 2 %. |
| 3 | `scenario.py` | §3.7, §3.1 | Versioned scenario schema (dataclass ↔ JSON): `scenario_id`, behaviour (B1…B8), seed, scene (gates: pose, shape, aperture, frame, colour, texture; distractors; pads; background set; lighting rig; camera `fovy` / pitch), initial state, `T_max`, dynamics DR draw, split tag, held-out flags, provenance (versions, checkpoint hash, command config). Language fields are filled after the success check. | Round-trip tests; schema version in every file. |
| 4 | `spatial.py` | §3.4 | Eq. (12) azimuth / elevation in the heading frame at the first frame; the six azimuth × three elevation bins; the three distance bands; the stratified initial-pose sampler with the FOV margin (B1/B3) and the out-of-FOV sampler (B4). | Bin tests; needs the camera decision (`fovy`) first. |
| 5 | `world.py` | §3.5, §6 action 2 | MuJoCo world-spec loader: gates (square, circle/arch, rectangle) and pads as **mocap bodies** added to `sim.spec`, so that each world has its own poses in `mjx_data.mocap_*` (the MuJoCo model is shared across worlds); materials and textures applied per world at render time (rendering is one world per call). Geometry parameters shared with the crossing test. | A rendered FPV frame shows the gate where eq. (13) expects it. |
| 6 | `sampler.py` | §3.3, §3.5, §3.10 | Scenario sampler per behaviour, starting with B1, then B6 (hover / hold / stop) and B7 (motion primitives: setpoint programs through the same velocity layer). Held-out factors fixed before collection: two colour–shape pairs, two background sets, the outer 10 % of every pose range; seed ranges 85 / 5 / 10 %. | Coverage matrix of a dry run (no physics) is balanced. |
| 7 | `collect.py` (pass 1) | §3.6, §3.8 | Batched rollouts with the trained expert: `a* = π(s)` (stochastic sampling, 5–10 seeds; the target speed is a scenario factor and is recorded), DART `u = a* + ε`, `Σ = diag(0.15², 0.15², 0.10², 0.20²)` on 50 % of episodes; log per decision `pos, quat, vel, ang_vel, rotor_vel`, the command reference, `a*`, `u`, gate events; success by eq. (13) (the env's `gate_crossing` with `r_safe`). Failed episodes go to a separate `failures` store. | 300 B1 episodes in minutes; DART tuned so the perturbed expert stays > 90 %. |
| 8 | `render.py` (pass 2) | §3.6, §3.9 | Replay logged poses of successful episodes at 30 Hz with `Sim.render(mode="rgb_array", camera="fpv_cam:0", width=224, height=224, world=w)` under `MUJOCO_GL=egl`, multiprocess. Image `I_t` from the state at the *start* of the label interval `[t, t+Δ)`. Optional segmentation render for the reprojection test. | Throughput measured on the pilot (spec: 500–2,000 frames/s per process is a tier-2 estimate). |
| 9 | `qc.py` | §3.9 | Success filter; replay fidelity (re-execute `u` open-loop from the recorded initial state: final error < 5 cm); label consistency (eq. 3: r ≥ 0.95 at the controller's measured lag, see `docs/crazyflow040_state_control.md` §5); image–state sync (gate-centre reprojection ≤ 3 px); coverage matrix (≥ 20 per cell); action statistics (percentiles, saturation < 2 %, near-zero `v_z` fraction); language audit. | Every check is a function with a test and a report entry. |
| 10 | `language.py` + an offline paraphrase script | §3.7 | Canonical instruction from the behaviour template and the verified outcome (bins, discriminators); five offline LLM paraphrases per canonical string, one family-level paraphrase held out; counterfactual pairs for B3. The on-disk `tasks` table stays canonical; paraphrases are sampled at training time (UAVLA side). | No instruction contradicts its scenario metadata. |
| 11 | `export_lerobot.py` | §2.2, §3.11 | `LeRobotDataset.create(fps=30, …)` / `add_frame` / `save_episode` / `finalize` with `observation.images.fpv` (224×224×3, video), `observation.state` (9-D: heading-frame velocity, roll, pitch, body rates, altitude — computed from the logged states), `action` (4-D `a*`), `task`; per-episode `meta/scenario` sidecar. Pin `lerobot` (0.5.2 locally) and record `codebase_version = v3.0`. | The 300-episode pilot loads in `lerobot-train` (5k-step smoke fine-tune on HPC). |
| 12 | `dagger.py` | §3.8 | Label the states visited by the fine-tuned VLA with the expert (the env's `observe` needs only `SimData` + gates); store as a separate dataset (`uavla_dagger_r1`). | After the first fine-tune. |

## Decisions needed before step 7

1. **Saturation — resolved on the single gate, confirm on mixed tracks.** A tanh-squashed policy
   plus a per-episode target speed (observed, tracked in the reward) brought the deterministic
   saturation (`|a| >= 0.99`) from 98.8 / 56.0 / 2.5 / 1.3 % (clipped Gaussian; vx, vy, vz, ω_z) to
   0 % on every axis, with 510/512 successes and realised speed tracking the target (r = 0.985;
   tier-1, `results/smoke_single_gate.md`). Check the same numbers on the 5-seed mixed-track
   expert (step 2), in particular success at the fastest target (97.9 % on the single gate).
2. **Camera** (spec §5 risk 6): AI-deck vs external camera sets `fovy`, resolution and channels;
   needed by steps 4, 5 and 8. The crazyflow FPV camera is `fovy = 70°`, 640×480.
3. **QC lag criterion** (`docs/crazyflow040_state_control.md` §5): restate "lag ≤ 1 step".
4. **Command chatter — fixed on the single gate (2026-10-29).** A command-gap penalty plus a
   larger action-rate weight removed it (results in `results/smoke_single_gate.md`, last section:
   sign flips 3.8 / 3.2 / 4.0 %, power above 5 Hz < 4 %, eq. (3) r = 0.981 / 0.980 / 0.944). vz
   misses r ≥ 0.95 by 0.006, at the limit of the z loop itself (reference → realised r = 0.951).
   The diagnosis that led to the fix: On the single-gate target-speed expert's own
   rollouts, the eq. (3) check fails (best-lag r = 0.58 / 0.69 / 0.87 for vx, vy, vz). Splitting
   it with `scripts/analyse_label_consistency.py` (tier-1, 512 episodes,
   `results/label_consistency_single_gate_vtarget.json`):
   - The controller is not the problem: rate-limited reference → realised velocity has
     r = 0.976 / 0.973 / 0.954 at a lag of 1–2 decisions, and where the rate limit is inactive,
     command → realised has r = 0.971 / 0.965 / 0.971.
   - The rate limit is active in 96 / 95 / 78 % of decisions, also after the first second, so the
     cause is not only the take-off step (the first hypothesis).
   - The cause is **period-2 chatter** of the deterministic policy in cruise: the vx command
     alternates by 0.40 m/s per decision on average (e.g. 1.1 ↔ 0.55 m/s while the drone flies a
     steady 0.79 m/s), consecutive steps flip sign in 82 % of decisions, and 62 % of the
     within-episode vx command power lies above 5 Hz (vy: 0.11 m/s, 70 %, 28 %; vz: 0.08 m/s,
     58 %, 8 %). The rate-limited reference averages the dither, so the drone flies smoothly, but
     the label is a 15 Hz oscillation that a VLA cannot learn from images and that has no meaning
     on the real vehicle. Not measured for the clipped and tanh-only experts (their 44-D
     observation predates the script); their forward command sat at or near the bound.
   - Likely mechanism (not tested): a limit cycle through the `prev_action` and `v_ref`
     observations; the action-rate penalty (`k_action_rate = 0.01` on normalised actions, about
     1.4e-3 per decision for this step) is of the same size as the speed term and too weak.
   - Proposed fix: a penalty on the gap between the command and the reference it produced,
     `−k (|R_ψ a_v − v_ref,end|²)` ("command only what the layer can realise"), plus a larger
     `k_action_rate`. Accept the expert when the step sign-flip rate is < 0.3, the power above
     5 Hz is < 5 % and eq. (3) gives r ≥ 0.95 at the controller's lag.
5. **Raw format.** Per-episode NumPy archives plus frame folders (or one MP4 per episode) are enough
   for the pilot; revisit when the MVP set (3,000–5,000 episodes) is collected.
