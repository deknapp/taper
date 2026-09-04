"""Tests for the SQLite store.

The store's job is to hand back exactly what it was given. Most of these are
round-trip tests for that reason: a field that is written but not read back is
a silent data loss, and silent data loss in a training log is indistinguishable
from the training never having happened.
"""
from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from taper.athlete import (
    AthleteProfile, GoalRace, Injury, LifeLoad, Occupation, Physiology, RaceResult,
    Sex, Surface, Tissue, TrainingBackground, TrainingDay,
)
from taper.db import SCHEMA_VERSION, Database


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def full_profile() -> AthleteProfile:
    return AthleteProfile(
        name="Test Runner",
        birth_date=date(1990, 6, 1),
        sex=Sex.FEMALE,
        physiology=Physiology(resting_hr=48, max_hr=188, vo2max_tested=57.5,
                              height_cm=170.0, body_mass_kg=62.5),
        training=TrainingBackground(
            current_weekly_km=64.0, peak_weekly_km_ever=110.0,
            longest_recent_run_km=28.0, runs_per_week=6.0, years_running=11.0,
            strength_days_per_week=2.0, cross_training_hours_per_week=1.5,
            primary_surface=Surface.TRAIL, hilliness_m_per_km=18.0),
        life=LifeLoad(sleep_hours=7.5, occupation=Occupation.ON_FEET, life_stress=4),
        races=[RaceResult(distance_m=21097.5, finish_time_s=5400.0,
                          race_date=date(2024, 4, 14), name="Spring Half",
                          place_overall=42, field_size=1200, surface=Surface.ROAD,
                          elevation_gain_m=110.0, source="paste")],
        injuries=[Injury(tissue=Tissue.TENDON, body_part="left achilles",
                         start_date=date(2023, 9, 2), weeks_out=6.0, recurrences=2,
                         notes="crept up over a hill block")],
        goal=GoalRace(distance_m=42195.0, race_date=date(2025, 10, 12),
                      name="Autumn Marathon", target_time_s=11400.0,
                      elevation_gain_m=180.0, surface=Surface.ROAD),
    )


# -- schema ----------------------------------------------------------------

def test_a_new_file_is_stamped_with_the_schema_version(tmp_path):
    with Database(tmp_path / "new.db") as database:
        version = database.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_reopening_an_existing_file_does_not_wipe_it(tmp_path):
    path = tmp_path / "keep.db"
    with Database(path) as first:
        athlete_id = first.save_profile(AthleteProfile(name="Persisted"))
    with Database(path) as second:
        assert second.load_profile(athlete_id).name == "Persisted"


def test_a_future_schema_is_refused_rather_than_migrated_blindly(tmp_path):
    path = tmp_path / "future.db"
    Database(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="Upgrade taper"):
        Database(path)


def test_the_parent_directory_is_created_if_missing(tmp_path):
    with Database(tmp_path / "nested" / "deeper" / "taper.db") as database:
        assert database.path.exists()


def test_foreign_keys_are_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.conn:
            db.conn.execute(
                "INSERT INTO race (athlete_id, distance_m, finish_time_s, created_at) "
                "VALUES (9999, 5000, 900, '2024-01-01')")


# -- profile round trip ----------------------------------------------------

def test_a_full_profile_survives_a_round_trip(db):
    athlete_id = db.save_profile(full_profile())
    loaded = db.load_profile(athlete_id)
    original = full_profile()

    assert loaded.name == original.name
    assert loaded.birth_date == original.birth_date
    assert loaded.sex is Sex.FEMALE
    assert loaded.physiology == original.physiology
    assert loaded.training == original.training
    assert loaded.life == original.life


def test_races_survive_a_round_trip_including_their_source(db):
    athlete_id = db.save_profile(full_profile())
    race = db.load_profile(athlete_id).races[0]
    assert race == full_profile().races[0]


def test_injuries_survive_a_round_trip(db):
    athlete_id = db.save_profile(full_profile())
    injury = db.load_profile(athlete_id).injuries[0]
    assert injury == full_profile().injuries[0]


def test_the_goal_race_survives_a_round_trip(db):
    athlete_id = db.save_profile(full_profile())
    assert db.load_profile(athlete_id).goal == full_profile().goal


def test_an_empty_profile_round_trips_to_its_defaults(db):
    athlete_id = db.save_profile(AthleteProfile())
    loaded = db.load_profile(athlete_id)
    assert loaded.name == ""
    assert loaded.sex is Sex.UNSPECIFIED
    assert loaded.races == [] and loaded.injuries == [] and loaded.goal is None


def test_loading_an_unknown_athlete_returns_none(db):
    assert db.load_profile(404) is None


def test_saving_over_an_athlete_replaces_child_rows_rather_than_duplicating(db):
    athlete_id = db.save_profile(full_profile())
    db.save_profile(full_profile(), athlete_id=athlete_id)
    loaded = db.load_profile(athlete_id)
    assert len(loaded.races) == 1
    assert len(loaded.injuries) == 1


def test_saving_over_an_athlete_keeps_the_same_id(db):
    athlete_id = db.save_profile(full_profile())
    profile = full_profile()
    profile.name = "Renamed"
    assert db.save_profile(profile, athlete_id=athlete_id) == athlete_id
    assert db.load_profile(athlete_id).name == "Renamed"


def test_races_come_back_in_date_order(db):
    profile = AthleteProfile(races=[
        RaceResult(distance_m=5000, finish_time_s=1100, race_date=date(2024, 5, 1)),
        RaceResult(distance_m=5000, finish_time_s=1000, race_date=date(2022, 5, 1)),
        RaceResult(distance_m=5000, finish_time_s=1050, race_date=date(2023, 5, 1)),
    ])
    athlete_id = db.save_profile(profile)
    dates = [r.race_date for r in db.load_profile(athlete_id).races]
    assert dates == sorted(dates)


# -- athletes --------------------------------------------------------------

def test_the_default_athlete_is_the_most_recently_updated(db):
    first = db.save_profile(AthleteProfile(name="First"))
    second = db.save_profile(AthleteProfile(name="Second"))
    assert db.default_athlete_id() in {first, second}
    assert db.default_athlete_id() == second


def test_default_athlete_is_none_on_an_empty_database(db):
    assert db.default_athlete_id() is None


def test_listing_athletes_reports_each_one(db):
    db.save_profile(AthleteProfile(name="A"))
    db.save_profile(AthleteProfile(name="B"))
    assert {a["name"] for a in db.list_athletes()} == {"A", "B"}


def test_deleting_an_athlete_takes_their_history_with_them(db):
    athlete_id = db.save_profile(full_profile())
    db.upsert_training_days(athlete_id, [TrainingDay(day=date(2024, 8, 1), distance_km=10)])
    db.delete_athlete(athlete_id)

    assert db.load_profile(athlete_id) is None
    assert db.training_days(athlete_id) == []
    assert db.conn.execute("SELECT COUNT(*) FROM race").fetchone()[0] == 0


# -- training days ---------------------------------------------------------

def training_day() -> TrainingDay:
    return TrainingDay(
        day=date(2024, 8, 15), distance_km=16.2, duration_s=4980.0, avg_hr=152,
        rpe=6.5, elevation_gain_m=240.0, elevation_loss_m=310.0,
        surface=Surface.TRAIL, name="Foothills loop", kind="long", source="strava",
        notes="warm")


def test_a_training_day_survives_a_round_trip(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id, [training_day()])
    assert db.training_days(athlete_id)[0] == training_day()


def test_re_importing_an_overlapping_range_corrects_rather_than_duplicates(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id, [TrainingDay(day=date(2024, 8, 1), distance_km=10)])
    db.upsert_training_days(athlete_id, [TrainingDay(day=date(2024, 8, 1), distance_km=12)])

    days = db.training_days(athlete_id)
    assert len(days) == 1
    assert days[0].distance_km == pytest.approx(12.0)


def test_two_athletes_do_not_share_a_training_log(db):
    a = db.save_profile(AthleteProfile(name="A"))
    b = db.save_profile(AthleteProfile(name="B"))
    db.upsert_training_days(a, [TrainingDay(day=date(2024, 8, 1), distance_km=10)])
    db.upsert_training_days(b, [TrainingDay(day=date(2024, 8, 1), distance_km=20)])

    assert db.training_days(a)[0].distance_km == pytest.approx(10.0)
    assert db.training_days(b)[0].distance_km == pytest.approx(20.0)


def test_training_days_come_back_in_day_order(db):
    athlete_id = db.save_profile(AthleteProfile())
    days = [TrainingDay(day=date(2024, 8, n)) for n in (5, 1, 3)]
    db.upsert_training_days(athlete_id, days)
    assert [d.day for d in db.training_days(athlete_id)] == [
        date(2024, 8, 1), date(2024, 8, 3), date(2024, 8, 5)]


def test_training_days_can_be_windowed_by_date(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id,
                            [TrainingDay(day=date(2024, 8, n)) for n in range(1, 11)])

    assert len(db.training_days(athlete_id, since=date(2024, 8, 5))) == 6
    assert len(db.training_days(athlete_id, until=date(2024, 8, 5))) == 5
    assert len(db.training_days(athlete_id, since=date(2024, 8, 4),
                                until=date(2024, 8, 6))) == 3


def test_upserting_nothing_writes_nothing(db):
    athlete_id = db.save_profile(AthleteProfile())
    assert db.upsert_training_days(athlete_id, []) == 0


def test_deleting_a_window_leaves_the_rest_of_the_log(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id,
                            [TrainingDay(day=date(2024, 8, n)) for n in range(1, 11)])
    removed = db.delete_training_days(athlete_id, since=date(2024, 8, 4),
                                      until=date(2024, 8, 6))
    assert removed == 3
    assert len(db.training_days(athlete_id)) == 7


def test_loading_a_profile_brings_its_training_days_with_it(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id, [training_day()])
    assert db.load_profile(athlete_id).training_days == [training_day()]


def test_saving_a_profile_does_not_disturb_the_training_log(db):
    athlete_id = db.save_profile(AthleteProfile())
    db.upsert_training_days(athlete_id, [training_day()])
    db.save_profile(AthleteProfile(name="Edited"), athlete_id=athlete_id)
    assert db.training_days(athlete_id) == [training_day()]


# -- settings --------------------------------------------------------------

def test_a_setting_round_trips(db):
    db.set_setting("units", "imperial")
    assert db.get_setting("units") == "imperial"


def test_an_unset_setting_returns_the_default(db):
    assert db.get_setting("nope") is None
    assert db.get_setting("nope", "fallback") == "fallback"


def test_setting_the_same_key_twice_overwrites(db):
    db.set_setting("units", "metric")
    db.set_setting("units", "imperial")
    assert db.all_settings() == {"units": "imperial"}


# -- migrations ------------------------------------------------------------

def _downgrade(path, version: int) -> None:
    """Strip a file back to an older training_day shape, to test the upgrade."""
    dropped = {2: ["sessions"], 1: ["sessions", "elevation_loss_m", "name"]}[version]
    conn = sqlite3.connect(path)
    for column in dropped:
        conn.execute(f"ALTER TABLE training_day DROP COLUMN {column}")
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


def test_a_v1_file_gains_the_missing_columns_on_open(tmp_path):
    path = tmp_path / "old.db"
    with Database(path) as database:
        athlete_id = database.save_profile(AthleteProfile(name="Old"))
        database.upsert_training_days(
            athlete_id, [TrainingDay(day=date(2024, 8, 1), distance_km=10.0)])
    _downgrade(path, 1)

    with Database(path) as upgraded:
        assert upgraded.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        day = upgraded.training_days(athlete_id)[0]
        assert day.elevation_loss_m is None
        assert day.name == ""
        assert day.sessions == 1


def test_upgrading_from_v1_keeps_the_training_already_logged(tmp_path):
    path = tmp_path / "old.db"
    with Database(path) as database:
        athlete_id = database.save_profile(AthleteProfile(name="Old"))
        database.upsert_training_days(
            athlete_id, [TrainingDay(day=date(2024, 8, n), distance_km=float(n))
                         for n in range(1, 8)])
    _downgrade(path, 1)

    with Database(path) as upgraded:
        days = upgraded.training_days(athlete_id)
        assert len(days) == 7
        assert [d.distance_km for d in days] == [float(n) for n in range(1, 8)]
        assert upgraded.load_profile(athlete_id).name == "Old"


def test_an_upgraded_file_can_then_store_elevation_loss(tmp_path):
    path = tmp_path / "old.db"
    with Database(path) as database:
        athlete_id = database.save_profile(AthleteProfile())
    _downgrade(path, 1)

    with Database(path) as upgraded:
        upgraded.upsert_training_days(athlete_id, [training_day()])
        assert upgraded.training_days(athlete_id)[0] == training_day()


def test_opening_an_already_current_file_is_a_no_op(tmp_path):
    path = tmp_path / "current.db"
    with Database(path) as database:
        athlete_id = database.save_profile(AthleteProfile())
        database.upsert_training_days(athlete_id, [training_day()])
    with Database(path) as reopened:
        assert reopened.training_days(athlete_id) == [training_day()]


def test_a_v2_file_gains_the_session_count_on_open(tmp_path):
    path = tmp_path / "v2.db"
    with Database(path) as database:
        athlete_id = database.save_profile(AthleteProfile())
        database.upsert_training_days(
            athlete_id, [TrainingDay(day=date(2024, 8, 1), distance_km=10.0,
                                     duration_s=2400.0)])
    _downgrade(path, 2)

    with Database(path) as upgraded:
        assert upgraded.conn.execute("PRAGMA user_version").fetchone()[0] == 3
        day = upgraded.training_days(athlete_id)[0]
        # Days logged before the column existed are assumed to be single runs,
        # which is what a hand-entered day almost always is.
        assert day.sessions == 1
        assert day.distance_km == pytest.approx(10.0)


def test_the_session_count_survives_a_round_trip(tmp_path):
    with Database(tmp_path / "s.db") as database:
        athlete_id = database.save_profile(AthleteProfile())
        double = TrainingDay(day=date(2024, 8, 1), distance_km=10.0,
                             duration_s=2400.0, sessions=2)
        database.upsert_training_days(athlete_id, [double])
        assert database.training_days(athlete_id)[0].sessions == 2
