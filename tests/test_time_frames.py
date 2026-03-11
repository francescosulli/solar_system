import numpy as np

from solsys_emulator.config import TIME_SCALE
from solsys_emulator.time_frames import from_model_time, parse_time, to_model_time


def test_parse_time_iso_and_jd_are_tdb():
    t_iso = parse_time("2000-01-01T12:00:00")
    t_jd = parse_time(2451545.0)
    assert t_iso.scale == TIME_SCALE
    assert t_jd.scale == TIME_SCALE
    assert np.isclose(t_iso.jd, t_jd.jd, rtol=0.0, atol=1e-9)


def test_model_time_round_trip():
    seconds = float(to_model_time("2000-01-02T12:00:00"))
    restored = from_model_time(seconds)
    assert np.isclose(seconds, 86400.0, rtol=0.0, atol=1e-6)
    assert restored.scale == TIME_SCALE
    assert np.isclose(restored.jd, parse_time("2000-01-02T12:00:00").jd, rtol=0.0, atol=1e-9)
