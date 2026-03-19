import numpy as np

from solsys_emulator.gravity_field import acceleration


def test_single_body_acceleration_magnitude():
    mu = {"earth": 398600.435507}
    states = {"earth": {"r": np.array([0.0, 0.0, 0.0])}}
    point = np.array([7000.0, 0.0, 0.0])

    a = acceleration(point, states, mu, epsilon=0.0)
    expected = np.array([-mu["earth"] / (7000.0**2), 0.0, 0.0])
    assert np.allclose(a, expected, rtol=1e-7, atol=1e-12)


def test_symmetry_two_equal_bodies_cancel_at_center():
    states = np.array([[10_000.0, 0.0, 0.0], [-10_000.0, 0.0, 0.0]])
    mu = np.array([1.0e5, 1.0e5])
    point = np.array([0.0, 0.0, 0.0])

    a = acceleration(point, states, mu, epsilon=0.0)
    assert np.allclose(a, np.zeros(3), atol=1e-12)
