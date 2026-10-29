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
| 1 | `evaluate.py` | §1.4, §3.9, §3.10 | Port `gate-ppo-eval`: deterministic and stochastic rollouts of a checkpoint on chain / circle families, success with Wilson 95 % intervals, `--save-traj`; label consistency (eq. 3) on the expert's own rollouts. | The mixed-track expert is evaluated on ≥ 1,000 episodes per family. |
| 2 | expert training (configs only) | §1.4, §6 action 1 | `experiment=mixed_tracks`, 5 seeds; decide the action-saturation fix first (below). | ≥ 95 % deterministic success with DR on, per seed; saturation < 2 %. |
| 3 | `scenario.py` | §3.7, §3.1 | Versioned scenario schema (dataclass ↔ JSON): `scenario_id`, behaviour (B1…B8), seed, scene (gates: pose, shape, aperture, frame, colour, texture; distractors; pads; background set; lighting rig; camera `fovy` / pitch), initial state, `T_max`, dynamics DR draw, split tag, held-out flags, provenance (versions, checkpoint hash, command config). Language fields are filled after the success check. | Round-trip tests; schema version in every file. |
| 4 | `spatial.py` | §3.4 | Eq. (12) azimuth / elevation in the heading frame at the first frame; the six azimuth × three elevation bins; the three distance bands; the stratified initial-pose sampler with the FOV margin (B1/B3) and the out-of-FOV sampler (B4). | Bin tests; needs the camera decision (`fovy`) first. |
| 5 | `world.py` | §3.5, §6 action 2 | MuJoCo world-spec loader: gates (square, circle/arch, rectangle) and pads as **mocap bodies** added to `sim.spec`, so that each world has its own poses in `mjx_data.mocap_*` (the MuJoCo model is shared across worlds); materials and textures applied per world at render time (rendering is one world per call). Geometry parameters shared with the crossing test. | A rendered FPV frame shows the gate where eq. (13) expects it. |
| 6 | `sampler.py` | §3.3, §3.5, §3.10 | Scenario sampler per behaviour, starting with B1, then B6 (hover / hold / stop) and B7 (motion primitives: setpoint programs through the same velocity layer). Held-out factors fixed before collection: two colour–shape pairs, two background sets, the outer 10 % of every pose range; seed ranges 85 / 5 / 10 %. | Coverage matrix of a dry run (no physics) is balanced. |
| 7 | `collect.py` (pass 1) | §3.6, §3.8 | Batched rollouts with the trained expert: `a* = π(s)` (stochastic sampling, 5–10 seeds), DART `u = a* + ε`, `Σ = diag(0.15², 0.15², 0.10², 0.20²)` on 50 % of episodes; log per decision `pos, quat, vel, ang_vel, rotor_vel`, the command reference, `a*`, `u`, gate events; success by eq. (13) (the env's `gate_crossing` with `r_safe`). Failed episodes go to a separate `failures` store. | 300 B1 episodes in minutes; DART tuned so the perturbed expert stays > 90 %. |
| 8 | `render.py` (pass 2) | §3.6, §3.9 | Replay logged poses of successful episodes at 30 Hz with `Sim.render(mode="rgb_array", camera="fpv_cam:0", width=224, height=224, world=w)` under `MUJOCO_GL=egl`, multiprocess. Image `I_t` from the state at the *start* of the label interval `[t, t+Δ)`. Optional segmentation render for the reprojection test. | Throughput measured on the pilot (spec: 500–2,000 frames/s per process is a tier-2 estimate). |
| 9 | `qc.py` | §3.9 | Success filter; replay fidelity (re-execute `u` open-loop from the recorded initial state: final error < 5 cm); label consistency (eq. 3: r ≥ 0.95 at the controller's measured lag, see `docs/crazyflow040_state_control.md` §5); image–state sync (gate-centre reprojection ≤ 3 px); coverage matrix (≥ 20 per cell); action statistics (percentiles, saturation < 2 %, near-zero `v_z` fraction); language audit. | Every check is a function with a test and a report entry. |
| 10 | `language.py` + an offline paraphrase script | §3.7 | Canonical instruction from the behaviour template and the verified outcome (bins, discriminators); five offline LLM paraphrases per canonical string, one family-level paraphrase held out; counterfactual pairs for B3. The on-disk `tasks` table stays canonical; paraphrases are sampled at training time (UAVLA side). | No instruction contradicts its scenario metadata. |
| 11 | `export_lerobot.py` | §2.2, §3.11 | `LeRobotDataset.create(fps=30, …)` / `add_frame` / `save_episode` / `finalize` with `observation.images.fpv` (224×224×3, video), `observation.state` (9-D: heading-frame velocity, roll, pitch, body rates, altitude — computed from the logged states), `action` (4-D `a*`), `task`; per-episode `meta/scenario` sidecar. Pin `lerobot` (0.5.2 locally) and record `codebase_version = v3.0`. | The 300-episode pilot loads in `lerobot-train` (5k-step smoke fine-tune on HPC). |
| 12 | `dagger.py` | §3.8 | Label the states visited by the fine-tuned VLA with the expert (the env's `observe` needs only `SimData` + gates); store as a separate dataset (`uavla_dagger_r1`). | After the first fine-tune. |

## Decisions needed before step 7

1. **Action saturation.** The single-gate smoke expert is bang-bang in the horizontal plane: with
   the deterministic policy, the command sits at the bound in 98.7 % of decisions for `v^h_x`
   (1.5 m/s) and 55.7 % for `v^h_y`, against 2.5 % for `v^h_z` and 1.2 % for `ω_z` (tier-1, 512
   episodes). As a label distribution this violates the QC limit (< 2 %) and puts mass on the
   boundary. Options: tanh-squashed Gaussian policy (the
   label approaches but never reaches the bound), a cruise-speed term in the reward, or wider
   bounds with the same acceleration limits. Decide with the 5-seed expert run (step 2).
2. **Camera** (spec §5 risk 6): AI-deck vs external camera sets `fovy`, resolution and channels;
   needed by steps 4, 5 and 8. The crazyflow FPV camera is `fovy = 70°`, 640×480.
3. **QC lag criterion** (`docs/crazyflow040_state_control.md` §5): restate "lag ≤ 1 step".
4. **Raw format.** Per-episode NumPy archives plus frame folders (or one MP4 per episode) are enough
   for the pilot; revisit when the MVP set (3,000–5,000 episodes) is collected.
