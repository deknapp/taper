"""Tests for the provenance report.

This module's whole job is to stop the app sounding more certain than it is, so
these tests are mostly about the warnings: that thin history is called thin,
that manufactured days are never counted as evidence, and that an uncalibrated
model says it is uncalibrated instead of quietly using population defaults.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from taper.athlete import (
    AthleteProfile, Injury, Physiology, RaceResult, Surface, Tissue,
    TrainingBackground, TrainingDay,
)
from taper.history import (
    calibrate, daily_loads, seed_training_days_from_summary, summarise,
    training_coverage,
)

TODAY = date(2024, 9, 1)


def profile(**kwargs) -> AthleteProfile:
    base = AthleteProfile(
        name="Test", birth_date=date(1990, 1, 1),
        physiology=Physiology(resting_hr=48, max_hr=190),
        training=TrainingBackground(current_weekly_km=60.0, runs_per_week=5.0,
                                    longest_recent_run_km=20.0),
        races=[RaceResult(distance_m=10000.0, finish_time_s=2400.0,
                          race_date=date(2024, 5, 1))],
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def run_days(n: int, *, end: date = TODAY, km: float = 10.0, source: str = "manual",
             every: int = 1) -> list[TrainingDay]:
    """A contiguous block of n days ending on `end`, running every `every` days."""
    start = end - timedelta(days=n - 1)
    days = []
    for i in range(n):
        running = (i % every) == 0
        days.append(TrainingDay(
            day=start + timedelta(days=i),
            distance_km=km if running else 0.0,
            duration_s=(km * 300.0) if running else None,
            kind="easy" if running else "off", source=source))
    return days


# -- coverage --------------------------------------------------------------

def test_an_empty_log_has_empty_coverage():
    coverage = training_coverage([], profile())
    assert coverage.days_logged == 0
    assert coverage.coverage_fraction == 0.0
    assert coverage.first_day is None


def test_coverage_spans_the_first_and_last_logged_day():
    coverage = training_coverage(run_days(30), profile())
    assert coverage.days_in_span == 30
    assert coverage.days_logged == 30
    assert coverage.coverage_fraction == pytest.approx(1.0)


def test_a_half_filled_log_reports_half_coverage():
    # Only the running days are logged; the rest are absent, not rest rows.
    days = [d for d in run_days(30, every=2) if d.distance_km > 0]
    coverage = training_coverage(days, profile())
    assert coverage.days_logged == 15
    assert coverage.days_in_span == 29
    assert coverage.coverage_fraction < 0.6


def test_total_distance_adds_up():
    coverage = training_coverage(run_days(10, km=8.0), profile())
    assert coverage.total_km == pytest.approx(80.0)


def test_rest_days_are_counted_as_rest():
    coverage = training_coverage(run_days(30, every=2), profile())
    assert coverage.rest_days == 15


def test_sources_are_tallied_so_the_ui_can_show_where_history_came_from():
    days = run_days(10, source="strava") + run_days(10, end=TODAY - timedelta(days=20),
                                                    source="manual")
    coverage = training_coverage(days, profile())
    assert coverage.sources == {"strava": 10, "manual": 10}


def test_manufactured_days_do_not_count_towards_real_days():
    days = run_days(10, source="manual") + run_days(
        10, end=TODAY - timedelta(days=20), source="estimated")
    coverage = training_coverage(days, profile())
    assert coverage.days_logged == 20
    assert coverage.real_days == 10


def test_the_load_method_is_recorded_per_day():
    with_hr = TrainingDay(day=TODAY, distance_km=10, duration_s=3000, avg_hr=150)
    coverage = training_coverage([with_hr], profile())
    assert coverage.methods == {"hr": 1}


def test_days_without_heart_rate_fall_back_to_a_weaker_method():
    bare = TrainingDay(day=TODAY, distance_km=10, duration_s=3000)
    coverage = training_coverage([bare], profile())
    assert "hr" not in coverage.methods


# -- daily loads -----------------------------------------------------------

def test_every_logged_day_gets_a_load():
    p = profile(training_days=run_days(10))
    loads = daily_loads(p)
    assert len(loads) == 10
    assert all(v > 0 for v in loads.values())


def test_a_rest_day_carries_no_load():
    p = profile(training_days=[TrainingDay(day=TODAY, kind="off")])
    assert daily_loads(p)[TODAY] == 0.0


def test_a_harder_day_carries_more_load_than_an_easy_one():
    p = profile(training_days=[
        TrainingDay(day=TODAY - timedelta(days=1), distance_km=8, duration_s=2880,
                    avg_hr=135),
        TrainingDay(day=TODAY, distance_km=8, duration_s=2100, avg_hr=175),
    ])
    loads = daily_loads(p)
    assert loads[TODAY] > loads[TODAY - timedelta(days=1)]


# -- calibration -----------------------------------------------------------

def test_an_empty_history_falls_back_to_population_defaults_and_says_so():
    result = calibrate(profile(training_days=[]))
    assert not result.fitted
    assert "population defaults" in result.note


def test_too_few_dated_races_will_not_fit_a_personal_model():
    result = calibrate(profile(training_days=run_days(200)))
    assert not result.fitted
    assert "at least three" in result.note


def test_an_unfitted_calibration_still_hands_back_usable_parameters():
    result = calibrate(profile(training_days=[]))
    assert result.params is not None
    assert result.rmse is None
    assert result.n_observations == 0


# -- the summary warnings --------------------------------------------------

def test_an_empty_log_is_called_empty():
    report = summarise(profile(training_days=[]), today=TODAY)
    assert any("nothing to run on" in w for w in report.warnings)


def test_a_well_covered_recent_log_raises_no_coverage_warning():
    report = summarise(profile(training_days=run_days(200)), today=TODAY)
    assert not any("are logged" in w for w in report.warnings)
    assert not any("stops" in w for w in report.warnings)


def test_a_sparse_log_warns_that_gaps_read_as_rest():
    days = [d for d in run_days(200, every=3) if d.distance_km > 0]
    report = summarise(profile(training_days=days), today=TODAY)
    assert any("treated as rest" in w for w in report.warnings)


def test_a_stale_log_warns_how_far_behind_it_is():
    days = run_days(60, end=TODAY - timedelta(days=45))
    report = summarise(profile(training_days=days), today=TODAY)
    assert any("stops 45 days ago" in w for w in report.warnings)


def test_a_log_that_stops_yesterday_is_not_called_stale():
    days = run_days(60, end=TODAY - timedelta(days=1))
    report = summarise(profile(training_days=days), today=TODAY)
    assert not any("stops" in w for w in report.warnings)


def test_manufactured_days_are_flagged_as_not_evidence():
    days = run_days(60, source="estimated")
    report = summarise(profile(training_days=days), today=TODAY)
    assert any("not evidence" in w for w in report.warnings)


def test_the_manufactured_warning_quotes_the_share_of_the_log():
    days = run_days(30, source="estimated") + run_days(
        30, end=TODAY - timedelta(days=30), source="manual")
    report = summarise(profile(training_days=days), today=TODAY)
    warning = next(w for w in report.warnings if "not evidence" in w)
    assert "30 of 60" in warning and "50%" in warning


def test_days_with_neither_heart_rate_nor_rpe_are_flagged():
    # Real distance and time, but nothing about intensity. This is what a Strava
    # import from a runner without a monitor looks like.
    days = [TrainingDay(day=TODAY - timedelta(days=i), distance_km=10,
                        duration_s=3000) for i in range(60)]
    report = summarise(profile(training_days=days), today=TODAY)
    assert any("Adding RPE" in w for w in report.warnings)


def test_days_with_an_assumed_duration_are_called_out_separately():
    days = [TrainingDay(day=TODAY - timedelta(days=i), distance_km=10)
            for i in range(60)]
    report = summarise(profile(training_days=days), today=TODAY)
    warning = next(w for w in report.warnings if "Adding RPE" in w)
    assert "duration was assumed too" in warning


def test_a_log_of_measured_days_is_not_flagged_as_thin():
    days = [TrainingDay(day=TODAY - timedelta(days=i), distance_km=10,
                        duration_s=3000, avg_hr=150) for i in range(60)]
    report = summarise(profile(training_days=days), today=TODAY)
    assert not any("Adding RPE" in w for w in report.warnings)


def test_a_log_of_rpe_days_is_not_flagged_as_thin():
    days = [TrainingDay(day=TODAY - timedelta(days=i), distance_km=10,
                        duration_s=3000, rpe=5) for i in range(60)]
    report = summarise(profile(training_days=days), today=TODAY)
    assert not any("Adding RPE" in w for w in report.warnings)


def test_undated_races_are_reported_as_uncalibratable():
    races = [RaceResult(distance_m=5000.0, finish_time_s=1200.0),
             RaceResult(distance_m=10000.0, finish_time_s=2400.0,
                        race_date=date(2024, 5, 1))]
    report = summarise(profile(races=races, training_days=run_days(60)), today=TODAY)
    assert any("has no date" in w for w in report.warnings)


def test_the_undated_race_warning_agrees_with_itself_in_the_plural():
    races = [RaceResult(distance_m=5000.0, finish_time_s=1200.0),
             RaceResult(distance_m=5000.0, finish_time_s=1210.0)]
    report = summarise(profile(races=races, training_days=run_days(60)), today=TODAY)
    warning = next(w for w in report.warnings if "no date" in w)
    assert "2 races have no date" in warning
    assert "they cannot" in warning


# -- the summary counts ----------------------------------------------------

def test_the_summary_counts_races_and_injuries():
    p = profile(injuries=[Injury(tissue=Tissue.TENDON, body_part="achilles")],
                training_days=run_days(60))
    report = summarise(p, today=TODAY)
    assert report.races == 1
    assert report.dated_races == 1
    assert report.injuries == 1


def test_the_summary_reports_the_first_and_last_race():
    races = [RaceResult(distance_m=5000.0, finish_time_s=1200.0,
                        race_date=date(2021, 5, 1)),
             RaceResult(distance_m=10000.0, finish_time_s=2400.0,
                        race_date=date(2024, 5, 1))]
    report = summarise(profile(races=races, training_days=run_days(60)), today=TODAY)
    assert report.first_race == date(2021, 5, 1)
    assert report.last_race == date(2024, 5, 1)


def test_a_profile_with_no_races_reports_no_race_dates():
    report = summarise(profile(races=[], training_days=run_days(60)), today=TODAY)
    assert report.first_race is None and report.last_race is None


# -- seeding a history from the weekly summary ----------------------------

def test_a_seeded_history_covers_every_day_in_the_window():
    days = seed_training_days_from_summary(profile(), weeks=4, today=TODAY)
    assert len(days) == 28
    assert days[-1].day == TODAY


def test_every_seeded_day_is_marked_as_estimated():
    days = seed_training_days_from_summary(profile(), weeks=4, today=TODAY)
    assert {d.source for d in days} == {"estimated"}
    assert all(d.notes for d in days if d.distance_km > 0)


def test_a_seeded_week_lands_near_the_stated_weekly_volume():
    days = seed_training_days_from_summary(profile(), weeks=4, today=TODAY)
    first_week = sum(d.distance_km for d in days[:7])
    assert first_week == pytest.approx(60.0, rel=0.05)


def test_a_seeded_week_has_the_stated_number_of_runs():
    days = seed_training_days_from_summary(profile(), weeks=4, today=TODAY)
    assert sum(1 for d in days[:7] if d.distance_km > 0) == 5


def test_the_seeded_long_run_is_the_longest_run_of_the_week():
    days = seed_training_days_from_summary(profile(), weeks=4, today=TODAY)
    week = days[:7]
    longest = max(week, key=lambda d: d.distance_km)
    assert longest.kind == "long"


def test_nothing_is_seeded_without_a_weekly_mileage_to_seed_from():
    p = profile(training=TrainingBackground(current_weekly_km=None))
    assert seed_training_days_from_summary(p, weeks=4, today=TODAY) == []


def test_seeded_days_never_count_as_real_history():
    p = profile(training_days=seed_training_days_from_summary(
        profile(), weeks=12, today=TODAY))
    report = summarise(p, today=TODAY)
    assert report.training.real_days == 0
    assert any("not evidence" in w for w in report.warnings)
