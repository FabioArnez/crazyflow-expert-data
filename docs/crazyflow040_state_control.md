# crazyflow 0.4.0 state control: what the velocity expert builds on

**Scope / Context:** Verification of the `Control.state` interface of crazyflow 0.4.0 for the
Option-A velocity-setpoint expert of the UAVLA data-collection spec (2026-09-24 spec, §1.3–1.5,
§3.9), and the design of the velocity-command layer in `src/expert_data/velocity.py`. All file:line
references are to the installed package `crazyflow==0.4.0` (`.venv/lib/python3.12/site-packages/crazyflow/`).
Measured numbers are **tier-1** (this laptop, CPU, 2026-10-29) and come from
`scripts/measure_tracking.py`; the raw outputs are in `results/tracking_*.json`.

## 1. Answers in short

| Question (spec) | Answer for 0.4.0 |
|---|---|
| Setpoint layout of `Control.state` (spec §1.4 says 13-D) | **16-D**: `[x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz]`, SI units, world frame for position/velocity/acceleration, body frame for the rates. Only the yaw of the quaternion is used. |
| Is velocity feed-forward? | Velocity enters as the **reference of the D term**, `kd·(v_ref − v)`; there is no separate velocity feed-forward gain. A reference that moves with `v_ref` is therefore followed without a velocity-induced lag. **Acceleration is a true feed-forward**, `m_ctrl·a_ref`. The body-rate entries are forwarded to the attitude stage as a **rate setpoint** (feed-forward on the rate error), so `wz` carries the yaw-rate command. |
| `Sim(freq=…, control_freq=…)` | There is **no `control_freq` argument**. The constructor takes `freq` (physics, default 500) and one rate per controller stage: `state_freq` (100), `attitude_freq` (500), `body_rate_freq` (500), `force_torque_freq` (500). `sim.control_freq` is a read-only property. |
| Rate grid | **480 / 120 / 30 Hz** (16 physics and 4 state-controller ticks per decision), with `attitude_freq = force_torque_freq = 480`. |

## 2. The interface, with references

**Setpoint layout.** `Control.state` is documented as "State control takes [x, y, z, vx, vy, vz,
ax, ay, az, qx, qy, qz, qw, wx, wy, wz]", recommended at ≥ 20 Hz, with the warning that only the yaw
of the attitude quaternion is used (`control/core.py:57-66`). The setter asserts the shape
`(n_worlds, n_drones, 16)` and writes the command into a staging buffer, `staged_cmd`
(`sim/functional.py:16-24`). The default command is all zeros with `qw = 1`
(`control/mellinger/control.py:514`).

**The position stage (Mellinger, firmware port).** `state2attitude` slices the setpoint into
position, velocity, acceleration and quaternion (`control/mellinger/control.py:84-87`) and computes

    e_p = p_ref − p,  e_v = v_ref − v,  I += e_p·dt (clipped)                       (l. 90-94)
    F_des = m_ctrl·(a_ref − g) + kp·e_p + kd·e_v + ki·I                              (l. 98-100)

The desired yaw comes from the setpoint quaternion (l. 103); the thrust direction and the yaw give
the desired attitude (l. 112-135); the collective thrust is rescaled through the firmware's
`mass_thrust` PWM conversion (l. 143). Gains for `cf2x_L250`
(`control/mellinger/params.toml:21-25`): `kp = [0.4, 0.4, 1.25]`, `kd = [0.2, 0.2, 0.5]`,
`ki = [0.05, 0.05, 0.05]`, `int_err_max = [2, 2, 0.4]`. The controller uses `mass = 0.029` kg
("The controller is using the wrong mass by default", `params.toml:3`), while the first-principles
model has `mass = 0.0328` kg (`dynamics/first_principles/params.toml:35`). Consequences: the
acceleration feed-forward is 12 % weak, and holding altitude needs a standing position error (or
the slow integral term).

**Body rates.** `control_state2attitude` stages the attitude command and forwards `cmd[13:16]` as the
attitude stage's rate setpoint (`control/mellinger/control.py:655-657`); the attitude stage uses
`e_ω = ω_des − ω` (l. 393). The yaw rate can therefore be fed forward through `wz`.

**No tilt limit, saturation.** The Mellinger position stage has no roll/pitch limit; the attitude
stage clips the torque PWM (l. 408) and the motor PWM (l. 413), and the pipeline clips the rotor
command to the motor limits (`sim/sim.py:166-169`, `sim/sim.py:738-745`): **[8,398, 23,216] RPM**
for `cf2x_L250` (per-motor thrust 0.0128–0.12 N). Hover is at 19,217 RPM, so the thrust-to-weight
ratio is ≈ 1.49 and the top of the motor range is only 21 % above hover.

**Timing.** The step pipeline for `Control.state` with first-principles dynamics is state →
attitude → force-torque → rotor clip → integration → floor clip → step counter
(`sim/sim.py:160-175`, `sim/sim.py:614-649`). Every stage runs inside every physics tick but only
updates when `(step − last_step) >= freq / stage_freq` or `last_step == −1`
(`control/core.py:176-190`); a staged command is committed at the next update of its stage
(`control/mellinger/control.py:643-644`). With `freq = 480` and the default `attitude_freq = 500`,
the attitude stage would update on every tick but use `dt = 1/500` in its D and I terms; the
velocity layer therefore sets `attitude_freq = force_torque_freq = sim_freq`.

**Frequency arguments.** `Sim.__init__(…, freq=500, state_freq=100, attitude_freq=500,
body_rate_freq=500, force_torque_freq=500, …)` (`sim/sim.py:79-97`); `control_freq` is a property
that returns the active stage's rate (`sim/sim.py:554-564`). The spec's pseudo-code
`Sim(…, freq=480, control_freq=120)` (§3.6) must read `state_freq=120`.

**Other 0.4.0 facts that the code depends on.**

- The dynamics parameters are created **without a world axis** (`mass` has shape `(1,)`,
  `dynamics/first_principles/dynamics.py:343-360`). Per-world domain randomisation needs them
  broadcast to `(N, 1, …)` first (`GateVelocityEnv.init_state`).
- `crazyflow.dynamics.load_params` takes the `Dynamics` enum, not the dynamics function
  (`dynamics/core.py:89-111`). crazyflow-gate-ppo calls it with the function.
- Drones are MuJoCo mocap bodies (`sim/sim.py:298-313`); nothing here uses MJX.
- The default integrator is explicit Euler (`sim/integration.py:16-20`); the quaternion update is the
  body-frame exponential map (`sim/integration.py:147`).
- The FPV camera exists: `fpv_cam`, `fovy = 70°`, 640×480 (`drones/cf2x_L250.xml:91`).

## 3. Rate grid: 480 / 120 / 30 Hz

The spec allows 480/120/30 or 500/100/25 (§1.5); mixing is not allowed. The choice is
**480/120/30**:

1. The decision rate stays at 30 Hz, on which the rest of the programme is defined: LeRobot
   `fps = 30`, the flow-matching chunk (50, execute 10) and the NSF chunk (16, execute 8).
2. The physics tick (480 Hz) is the Phase-1 tick, so RPM-expert and velocity-expert dynamics use the
   same integration step.
3. The state loop at 120 Hz is above the firmware's 100 Hz; the attitude loop runs at 480 Hz instead
   of the firmware's 500 Hz (−4 %).
4. Measured: the two grids track identically in physical time (table below; the lag in decisions is
   the same, 4 decisions = 133 ms at 30 Hz versus 160 ms at 25 Hz). Nothing favours 25 Hz.

`RateGrid` refuses non-integer nestings (`RateGrid(500, 100, 30)` raises).

## 4. From a velocity command to state setpoints

Spec eq. (2) anchors the position reference at the measured position at every decision,
`p_ref = p_t + Δ·R_ψ v^h`. Measured with one setpoint per decision ("measured" anchor):

- At zero command the drone **sinks 0.45 m in 5 s** (0.09 m/s). The controller's 12 % mass
  deficit needs a standing error, and re-anchoring discards it every 33 ms.
- Steady-state errors: −15 % on a 0.5 m/s climb step, +16 % on a 1 rad/s yaw-rate step.

The layer therefore carries an **integrated reference** `(p_ref, v_ref, ψ_ref)` across decisions and
re-feeds it at every state-controller tick (4 per decision):

    v_ref ← v_ref + clip(R_ψ v^h − v_ref, a_max·h)      h = 1/120 s; a_max = 2 m/s² (xy), 1 m/s² (z)
    a_ref = Δv_ref / h                                  acceleration feed-forward
    p_ref ← p_ref + v_ref·h + a_ref·h²/2
    ψ_ref ← ψ_ref + ω_z·h,   rate setpoint (0, 0, ω_z)

with `|p_ref − p| ≤ 0.3 m` and `|ψ_ref − ψ| ≤ 0.3 rad` enforced at the start of every decision
(anti-windup). The heading frame is the measured yaw at the start of the decision (spec eq. 1).

**Why the rate limit.** Without it, a velocity step is an infeasible reference: errors grow, the
Mellinger stage demands tens of degrees of tilt, the motors saturate at 23.2 k RPM and the drone
tumbles. A plain diagonal position step of (1, 1) m tumbles it (133° tilt after 0.5 s). With
smooth random commands (σ = 0.6 m/s, 1 Hz cutoff), 8/8 worlds diverged without the limit
(x, y, z commands only), 16/16 with `a_max = 4 / 2 m/s²`, 15/16 with `3 / 1.5 m/s²` and 0/16 with
`2 / 1 m/s²` (all four axes commanded). `a_max` plays the role of the firmware's
roll/pitch limit in its velocity mode; 2 m/s² is ≈ 12° of tilt.

## 5. Tracking, tier-1

Default layer (integrated, `a_max = 2 / 1 m/s²`), 480/120/30 Hz. Steps are applied from hover; the
realised value is spec eq. (3), the mean velocity over each decision. Random commands are white
noise through a 2nd-order low-pass, 16 worlds × 10 s, first second dropped.

| Test | vx | vy | vz | ω_z |
|---|---|---|---|---|
| Step amplitude | 1 m/s | 1 m/s | 0.5 m/s | 1 rad/s |
| Rise time 10–90 % | 0.33 s | 0.33 s | 0.33 s | 0.10 s |
| Steady-state error | +0.3 % | +0.3 % | +0.3 % | 0.0 % |
| Overshoot | 21 % | 21 % | 10 % | 39 % |
| Gentle (σ 0.4/0.4/0.2/0.4, 0.5 Hz): r at lag 0 | 0.88 | 0.90 | 0.95 | 0.87 |
| Gentle: best lag (decisions) / r at best lag | 4 / 0.967 | 4 / 0.972 | 2 / 0.974 | 1 / 0.878 |
| Moderate (σ 0.6/0.6/0.3/0.6, 0.5 Hz): best lag / r | 6 / 0.950 | 6 / 0.959 | 3 / 0.968 | 1 / 0.899 |
| Aggressive (same σ, 1 Hz): best lag / r | 9 / 0.878 | 8 / 0.899 | 5 / 0.925 | 2 / 0.866 |

Hover at zero command: 3.3 cm steady sag after 5 s, then at rest (the mass deficit, absorbed
inside the leash). No world diverged in any of the three random-command sets.

For comparison, the "measured" anchor has r = 0.986 / 0.989 / 0.992 / 0.954 at its best lag on the
gentle set, but its best lag is 6 / 6 / 3 / 1 decisions, it sinks at zero command, and 13/16 worlds
diverged on the aggressive set. The 500/100/25 grid gives best lags 4 / 4 / 2 / 1 and
r = 0.966 / 0.969 / 0.971 / 0.869 on the gentle set.

**Against the spec's QC criterion (§3.9: r ≥ 0.95 per dimension, lag ≤ 1 step).** The criterion
**is not met** in xy: the cascade (position loop at ω_n ≈ 3.5 rad/s, attitude loop ≈ 0.1 s, rotor
dynamics) lags the command by ≈ 4 decisions (133 ms) even for gentle commands, and no setting of
the layer changed that (the same lag for `a_max` from 2 to 4 m/s², a larger one with the
spec-literal anchor). The yaw rate meets the lag but not the correlation (0.88): under combined
roll and pitch, the Euler yaw that eq. (3) uses moves with the tilt. `tests/test_velocity.py`
asserts what the layer achieves (r ≥ 0.95 at lag ≤ 5 in xy, ≤ 3 in z, ≤ 1 in yaw) and keeps the
spec criterion as a strict `xfail`. Option A is not affected (the label is the command by
construction); the criterion should be restated as "r ≥ 0.95 at the controller's measured lag,
with that lag reported", or it needs a stiffer controller than the firmware-gain Mellinger.

## 6. Consequences for the expert

- **Speed range.** The action bounds are `v^h_x ∈ ±1.5`, `v^h_y ∈ ±1.0`, `v^h_z ∈ ±0.75` m/s,
  `ω_z ∈ ±1.5` rad/s (`EnvConfig`). The Phase-1 RPM expert (mixed tracks) flew at a mean 4.0 m/s
  with a mean rotor command of 22.2 k RPM (tier-1, last iteration of its `metrics.jsonl`), so the
  velocity expert will be about 3× slower; the episode budget is 2 s + 3 s per gate.
- **Hidden controller state.** The carried reference is state the policy cannot infer from the
  drone alone, so the observation adds `p_ref − p`, `v_ref` and `ψ_ref − ψ` (37-D → 44-D; the
  per-episode target speed added later makes it 45-D).
- **For the RPM expert on 0.4.0** (the other session): the new rotor clip at 23.2 k RPM is below
  its action range (`[0.65, 1.35]·hover`, up to 25.6 k RPM at the 0.3.2 hover) and below the top of
  its flown range; the hover RPM moved from 18,968 to 19,217 with the mass change; and
  `load_params` takes the enum.

## 7. Open questions

1. **Deployment controller.** The Crazyflie default is the PID controller, whose velocity mode has
   its own gains and a 20° roll/pitch limit; crazyflow only models Mellinger. The closed-loop
   velocity response in simulation (lag ≈ 133 ms) is therefore not the deployment one. Measure the
   real step response once the firmware interface (spec §5 risk 7) is fixed.
2. **QC criterion.** Restate §3.9 "lag ≤ 1 step" (see §5) or decide to tune the state-controller
   gains away from the firmware values.
3. **Acceleration limit.** 2 / 1 m/s² is conservative; the boundary between 2 and 3 m/s² was not
   resolved further. Revisit after the first expert run: if success is below 95 %, the limit and
   the speed range are the two knobs (spec §5 risk 2), not the action space.
4. **Yaw label check.** Use the Mellinger heading (projection of the body x axis) instead of the
   Euler yaw in eq. (3) if the yaw correlation must pass 0.95.
