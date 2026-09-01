from datetime import date

import pytest

from taper.athlete import (
    AthleteProfile, GoalRace, Injury, Occupation, RaceResult, Sex, Surface, Tissue,
)
from taper.profile_io import from_dict, load, save, to_dict


def sample_profile() -> AthleteProfile:
    p = AthleteProfile(name="Jane Doe", birth_date=date(1990, 3, 4), sex=Sex.FEMALE)
    p.physiology.height_cm = 170.0
    p.physiology.body_mass_kg = 58.0
    p.physiology.max_hr = 188
    p.training.current_weekly_km = 64.4
    p.training.years_running = 9.0
    p.training.primary_surface = Surface.TRAIL
    p.training.strength_days_per_week = 2.0
    p.life.occupation = Occupation.ON_FEET
    p.life.sleep_hours = 7.5
    p.races = [
        RaceResult(42195.0, 11564.0, date(2024, 4, 15), "Boston", 1204, 25000),
        RaceResult(5000.0, 1182.0, date(2023, 11, 23), "Turkey Trot", 7, 850),
    ]
    p.injuries = [Injury(Tissue.TENDON, "left achilles", date(2022, 6, 1), 8.0, 2)]
    p.goal = GoalRace(42195.0, date(2026, 11, 1), "Goal Marathon", 11400.0)
    return p


def test_roundtrip_preserves_everything():
    original = sample_profile()
    restored = from_dict(to_dict(original))
    assert restored == original


def test_roundtrip_through_disk(tmp_path):
    original = sample_profile()
    path = save(original, tmp_path / "jane.json")
    assert load(path) == original


def test_empty_profile_roundtrips():
    assert from_dict(to_dict(AthleteProfile())) == AthleteProfile()


def test_enums_and_dates_serialize_as_strings():
    payload = to_dict(sample_profile())
    assert payload["sex"] == "female"
    assert payload["birth_date"] == "1990-03-04"
    assert payload["races"][0]["race_date"] == "2024-04-15"
    assert payload["injuries"][0]["tissue"] == "tendon"


def test_future_schema_version_is_refused():
    payload = to_dict(sample_profile())
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="schema version 999"):
        from_dict(payload)
