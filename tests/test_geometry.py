import jax.numpy as jnp
import numpy as np

from expert_data.geometry import (
    gate_crossing,
    gate_frame,
    heading_to_world,
    hover_rpm,
    quat_from_rpy,
    quat_rotate,
    quat_rotate_inv,
    world_to_heading,
    wrap_angle,
    yaw_from_quat,
)


def test_quat_from_yaw_rotates_x_axis():
    yaw = jnp.array(0.7)
    q = quat_from_rpy(jnp.array(0.0), jnp.array(0.0), yaw)
    x_world = quat_rotate(q, jnp.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(x_world, [np.cos(0.7), np.sin(0.7), 0.0], atol=1e-6)


def test_quat_rotate_inverse_roundtrip():
    q = quat_from_rpy(jnp.array(0.3), jnp.array(-0.2), jnp.array(1.1))
    v = jnp.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(quat_rotate_inv(q, quat_rotate(q, v)), v, atol=1e-6)


def test_gate_frame_is_orthonormal():
    n, u = gate_frame(jnp.array(0.4))
    assert abs(float(jnp.dot(n, u))) < 1e-6
    np.testing.assert_allclose(jnp.linalg.norm(n), 1.0, atol=1e-6)


def test_gate_crossing_classification():
    gate = jnp.zeros(3)
    yaw = jnp.array(0.0)  # normal along +x
    kw = dict(half_opening=0.245, frame_thickness=0.06, drone_radius=0.07)

    def classify(prev, pos):
        out = gate_crossing(jnp.array(prev), jnp.array(pos), gate, yaw, **kw)
        return tuple(bool(x) for x in out)

    assert classify([-0.1, 0.0, 0.0], [0.1, 0.05, 0.0]) == (
        True,
        False,
        False,
    )  # through the opening
    assert classify([-0.1, 0.3, 0.0], [0.1, 0.3, 0.0]) == (
        False,
        True,
        False,
    )  # into the frame band
    assert classify([-0.1, 0.0, 0.3], [0.1, 0.0, 0.3]) == (False, True, False)  # top bar
    assert classify([-0.1, 0.6, 0.0], [0.1, 0.6, 0.0]) == (False, False, True)  # beside the gate
    assert classify([-0.3, 0.0, 0.0], [-0.1, 0.0, 0.0]) == (False, False, False)  # no crossing
    assert classify([0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]) == (False, False, False)  # backwards


def test_gate_crossing_uses_interpolated_point():
    # Lateral offset at the crossing point (x=0) is 0.1 even though the endpoints are far off-axis.
    gate = jnp.zeros(3)
    passed, hit, missed = gate_crossing(
        jnp.array([-1.0, -0.4, 0.0]),
        jnp.array([1.0, 0.6, 0.0]),
        gate,
        jnp.array(0.0),
        half_opening=0.245,
        frame_thickness=0.06,
        drone_radius=0.07,
    )
    assert bool(passed) and not bool(hit) and not bool(missed)


def test_hover_rpm_cf2x_l250():
    # crazyflow 0.3.2: mass 0.0319 kg, hover 18,968 RPM (the vault's figure); 0.4.0: 0.0328 kg.
    c = [0.0, -5.382196214637237e-7, 2.4582929831265485e-10]
    assert abs(hover_rpm(c, mass=0.0319) - 18968.0) < 5.0
    assert abs(hover_rpm(c, mass=0.0328) - 19217.0) < 5.0


def test_yaw_from_quat_inverts_quat_from_rpy():
    for roll, pitch, yaw in [(0.0, 0.0, 0.7), (0.2, -0.1, -2.5), (-0.3, 0.25, 3.0)]:
        q = quat_from_rpy(jnp.array(roll), jnp.array(pitch), jnp.array(yaw))
        np.testing.assert_allclose(yaw_from_quat(q), yaw, atol=1e-5)


def test_heading_frame_is_yaw_only_rotation():
    yaw = jnp.array(np.pi / 2)  # heading along world +y
    np.testing.assert_allclose(
        heading_to_world(jnp.array([1.0, 0.0, 0.5]), yaw), [0, 1, 0.5], atol=1e-6
    )
    np.testing.assert_allclose(
        heading_to_world(jnp.array([0.0, 1.0, 0.0]), yaw), [-1, 0, 0], atol=1e-6
    )
    v = jnp.array([0.3, -1.2, 0.4])
    np.testing.assert_allclose(world_to_heading(heading_to_world(v, yaw), yaw), v, atol=1e-6)


def test_wrap_angle():
    np.testing.assert_allclose(
        wrap_angle(jnp.array([3.5, -3.5, 0.1])), [3.5 - 2 * np.pi, 2 * np.pi - 3.5, 0.1], atol=1e-6
    )
