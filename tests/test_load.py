import pytest

from taper.athlete import Sex, Surface, TrainingDay
from taper.load import (
    day_load, heart_rate_reserve, session_rpe, trimp_heart_rate, trimp_pace,
)
from datetime import date

DAY = date(2026, 5, 1)


def test_heart_rate_reserve_clamps():
    assert heart_rate_reserve(150, 50, 190) == pytest.approx(0.714, abs=0.001)
    assert heart_rate_reserve(40, 50, 190) == 0.0
    assert heart_rate_reserve(220, 50, 190) == 1.0


def test_trimp_rises_with_both_duration_and_intensity():
    easy = trimp_heart_rate(60, 130, 50, 190)
    longer = trimp_heart_rate(120, 130, 50, 190)
    harder = trimp_heart_rate(60, 170, 50, 190)
    assert longer == pytest.approx(2 * easy)
    assert harder > easy


def test_trimp_weights_intensity_superlinearly():
    """A hard hour must outscore an easy two hours, or the model would reward
    plodding and the game would degenerate into logging volume."""
    hard_hour = trimp_heart_rate(60, 178, 50, 190)
    easy_two_hours = trimp_heart_rate(120, 120, 50, 190)
    assert hard_hour > easy_two_hours


def test_session_rpe_is_duration_times_effort():
    assert session_rpe(60, 5) == pytest.approx(30.0)
    assert session_rpe(60, 0) == 0.0


def test_pace_load_responds_to_gradient():
    flat = trimp_pace(10.0, 50.0, vdot=50)
    hilly = trimp_pace(10.0, 50.0, vdot=50, gradient_factor=1.4)
    assert hilly > flat


def test_pace_load_falls_as_fitness_rises():
    """The same run is less of a stimulus to a fitter runner."""
    assert trimp_pace(10.0, 45.0, vdot=45) > trimp_pace(10.0, 45.0, vdot=60)


def test_day_load_prefers_heart_rate_then_rpe_then_pace():
    hr_day = TrainingDay(DAY, distance_km=10, duration_s=3000, avg_hr=150, rpe=5)
    assert day_load(hr_day, rest_hr=50, max_hr=190, vdot=50).method == "hr"

    rpe_day = TrainingDay(DAY, distance_km=10, duration_s=3000, rpe=5)
    assert day_load(rpe_day, rest_hr=50, max_hr=190, vdot=50).method == "rpe"

    pace_day = TrainingDay(DAY, distance_km=10, duration_s=3000)
    assert day_load(pace_day, vdot=50).method == "pace"


def test_day_load_handles_a_rest_day():
    result = day_load(TrainingDay(DAY, kind="off"))
    assert result.value == 0.0
    assert result.method == "rest"


def test_day_load_estimates_duration_from_distance_alone():
    result = day_load(TrainingDay(DAY, distance_km=10), vdot=50)
    assert result.value > 0
    assert result.method == "distance"
