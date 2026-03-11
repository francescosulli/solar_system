import astropy.units as u

from solsys_emulator.config import DEFAULT_BODIES
from solsys_emulator.constants import MU_DICT, get_mu, validate_mu_dict


def test_mu_positive_and_complete():
    validate_mu_dict(DEFAULT_BODIES)
    for body in DEFAULT_BODIES:
        assert get_mu(body) > 0.0


def test_mu_units_are_consistent():
    for body in DEFAULT_BODIES:
        quantity = MU_DICT[body]
        assert quantity.unit.is_equivalent(u.km**3 / u.s**2)
        assert quantity.to_value(u.km**3 / u.s**2) > 0.0
