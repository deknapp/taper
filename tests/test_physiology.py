import pytest

from taper.physiology import (
    gradient_cost, grade_adjusted_factor, max_hr_estimate,
    predict_time, riegel_predict, vdot_from_race,
)


def test_vdot_matches_published_table():
    """The VDOT 50 row of Daniels' table, across all four road distances.

    That one fitness number reproduces every distance is the whole point of the
    index, so this is the test that actually pins the implementation down.
    """
    row_50 = [
        (5000, 19 * 60 + 57),          # 19:57
        (10000, 41 * 60 + 21),         # 41:21
        (21097.5, 91 * 60 + 35),       # 1:31:35
        (42195, 3 * 3600 + 10 * 60 + 49),  # 3:10:49
    ]
    for distance, time_s in row_50:
        assert vdot_from_race(distance, time_s) == pytest.approx(50, abs=0.1)


def test_vdot_increases_with_faster_time():
    assert vdot_from_race(5000, 18 * 60) > vdot_from_race(5000, 20 * 60)


def test_predict_time_inverts_vdot():
    for distance in (1500, 5000, 10000, 21097.5, 42195):
        vdot = vdot_from_race(distance, 1800 if distance <= 10000 else 9000)
        recovered = predict_time(distance, vdot)
        assert vdot_from_race(distance, recovered) == pytest.approx(vdot, abs=0.01)


def test_riegel_scales_up():
    half = 90 * 60
    full = riegel_predict(21097.5, half, 42195)
    # The folk 'double it and add ten minutes' lands near the Riegel figure.
    assert full == pytest.approx(2 * half + 10 * 60, rel=0.05)


def test_minetti_flat_cost():
    assert gradient_cost(0.0) == pytest.approx(3.6, abs=0.01)
    assert grade_adjusted_factor(0.0) == pytest.approx(1.0, abs=0.01)


def test_minetti_downhill_is_cheaper_than_flat():
    # The curve bottoms out near -10%: gentle downhill costs less than flat.
    assert gradient_cost(-0.10) < gradient_cost(0.0)
    assert gradient_cost(-0.10) < gradient_cost(-0.30)


def test_minetti_uphill_is_expensive():
    assert grade_adjusted_factor(0.10) > 1.6
    assert grade_adjusted_factor(0.20) > 2.2


def test_max_hr():
    assert max_hr_estimate(30) == pytest.approx(187.0)
