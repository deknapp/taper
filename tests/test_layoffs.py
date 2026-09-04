"""Tests for layoff detection.

This module proposes; it never concludes. So the tests check two things in
equal measure: that a real hole in a log is found, and that what comes back is
honestly hedged -- a gap with no history behind it must not be dressed up as a
confident finding.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from taper.athlete import Tissue, TrainingDay
from taper.layoffs import MIN_STOPPAGE_DAYS, Layoff, find_layoffs

START = date(2023, 1, 2)   # a Monday


def build(pattern: list[float], start: date = START) -> list[TrainingDay]:
    """One day per entry: kilometres run, 0 for a rest day."""
    return [
        TrainingDay(day=start + timedelta(days=i), distance_km=km,
                    duration_s=(km * 300.0) if km else None,
                    kind="easy" if km else "off")
        for i, km in enumerate(pattern)
    ]


def steady(days: int, weekly_km: float = 70.0, start: date = START) -> list[float]:
    """A consistent runner: six days on, one off, at the given weekly volume."""
    daily = weekly_km / 6.0
    return [0.0 if (i % 7) == 6 else daily for i in range(days)]


# -- not enough to say anything -------------------------------------------

def test_an_empty_log_proposes_nothing():
    assert find_layoffs([]) == []


def test_a_log_too_short_to_have_context_proposes_nothing():
    # Under six weeks there is not enough either side of a gap to judge it.
    assert find_layoffs(build(steady(20) + [0.0] * 15)) == []


def test_a_normal_rest_day_is_not_a_layoff():
    assert find_layoffs(build(steady(200))) == []


def test_a_long_weekend_off_is_not_a_layoff():
    pattern = steady(100) + [0.0] * 4 + steady(100)
    assert find_layoffs(build(pattern)) == []


# -- stoppages -------------------------------------------------------------

def test_a_month_off_in_a_consistent_log_is_found():
    pattern = steady(120) + [0.0] * 30 + steady(120)
    layoffs = find_layoffs(build(pattern))
    assert len(layoffs) == 1
    assert layoffs[0].kind == "stoppage"
    assert layoffs[0].days == 30


def test_a_stoppage_spans_exactly_the_days_without_running():
    pattern = steady(120) + [0.0] * 30 + steady(120)
    layoff = find_layoffs(build(pattern))[0]
    assert layoff.start == START + timedelta(days=120)
    assert layoff.end == START + timedelta(days=149)
    assert layoff.during_weekly_km == 0.0


def test_the_threshold_is_the_documented_minimum():
    below = steady(120) + [0.0] * (MIN_STOPPAGE_DAYS - 1) + steady(120)
    at = steady(120) + [0.0] * MIN_STOPPAGE_DAYS + steady(120)
    assert find_layoffs(build(below)) == []
    assert len(find_layoffs(build(at))) == 1


def test_two_separate_stoppages_are_both_proposed():
    pattern = steady(90) + [0.0] * 20 + steady(90) + [0.0] * 20 + steady(90)
    assert len(find_layoffs(build(pattern))) == 2


def test_a_long_stop_by_a_high_mileage_runner_is_reported_as_strong():
    pattern = steady(120, weekly_km=70) + [0.0] * 30 + steady(120, weekly_km=70)
    layoff = find_layoffs(build(pattern))[0]
    assert layoff.confidence == "strong"
    assert "70 km/week" in layoff.reason


def test_a_fortnight_off_is_only_moderate():
    pattern = steady(120, weekly_km=70) + [0.0] * 15 + steady(120, weekly_km=70)
    assert find_layoffs(build(pattern))[0].confidence == "moderate"


def test_a_low_mileage_runner_does_not_get_a_strong_verdict():
    pattern = steady(120, weekly_km=15) + [0.0] * 30 + steady(120, weekly_km=15)
    assert find_layoffs(build(pattern))[0].confidence != "strong"


def test_the_reason_always_says_something_a_person_can_read():
    pattern = steady(120) + [0.0] * 30 + steady(120)
    layoff = find_layoffs(build(pattern))[0]
    assert layoff.reason.endswith(".")
    assert str(layoff.days) in layoff.reason


def test_the_baseline_looks_only_at_training_before_the_gap():
    # 30 km/week beforehand, 90 after. The baseline is what was lost, so it must
    # not be contaminated by the comeback.
    pattern = steady(120, weekly_km=30) + [0.0] * 30 + steady(120, weekly_km=90)
    layoff = find_layoffs(build(pattern))[0]
    assert layoff.baseline_weekly_km == pytest.approx(30.0, rel=0.1)


def test_the_baseline_is_a_ninety_day_window_not_the_most_recent_week():
    # Ramping 30 -> 90 over the 90 days before stopping gives a blended
    # baseline, deliberately: chronic load is what the body adapted to, and one
    # big week before an injury is not a fitness level.
    pattern = steady(60, weekly_km=30) + steady(60, weekly_km=90) + [0.0] * 30 \
        + steady(60, weekly_km=90)
    layoff = find_layoffs(build(pattern))[0]
    assert 60.0 < layoff.baseline_weekly_km < 80.0


# -- dips ------------------------------------------------------------------

def test_running_through_something_at_half_volume_is_found():
    pattern = steady(120, weekly_km=70) + steady(28, weekly_km=20) \
        + steady(120, weekly_km=70)
    dips = [l for l in find_layoffs(build(pattern)) if l.kind == "dip"]
    assert dips
    assert dips[0].during_weekly_km < dips[0].baseline_weekly_km


def test_a_dip_that_is_not_deep_enough_is_left_alone():
    # 80% of baseline is a normal down week, not a layoff.
    pattern = steady(120, weekly_km=70) + steady(28, weekly_km=56) \
        + steady(120, weekly_km=70)
    assert [l for l in find_layoffs(build(pattern)) if l.kind == "dip"] == []


def test_a_dip_too_short_to_be_meaningful_is_left_alone():
    pattern = steady(120, weekly_km=70) + steady(7, weekly_km=15) \
        + steady(120, weekly_km=70)
    assert [l for l in find_layoffs(build(pattern)) if l.kind == "dip"] == []


def test_a_dip_reason_quotes_the_share_of_baseline_it_fell_to():
    pattern = steady(120, weekly_km=70) + steady(28, weekly_km=20) \
        + steady(120, weekly_km=70)
    dip = [l for l in find_layoffs(build(pattern)) if l.kind == "dip"][0]
    assert "%" in dip.reason
    assert "km/week" in dip.reason


def test_a_long_dip_is_more_confident_than_a_short_one():
    long_dip = steady(120, weekly_km=70) + steady(35, weekly_km=15) \
        + steady(120, weekly_km=70)
    dip = [l for l in find_layoffs(build(long_dip)) if l.kind == "dip"][0]
    assert dip.confidence == "moderate"


# -- overlap ---------------------------------------------------------------

def test_a_stoppage_inside_a_dip_is_reported_once_as_the_stoppage():
    # Tapering off, stopping outright, then coming back is one event.
    pattern = (steady(120, weekly_km=70) + steady(14, weekly_km=15) + [0.0] * 21
               + steady(120, weekly_km=70))
    layoffs = find_layoffs(build(pattern))
    starts = [l.start for l in layoffs]
    assert len(starts) == len(set(starts))
    for earlier, later in zip(layoffs, layoffs[1:]):
        assert later.start > earlier.end


def test_proposals_come_back_in_chronological_order():
    pattern = steady(90) + [0.0] * 20 + steady(90) + [0.0] * 20 + steady(90)
    layoffs = find_layoffs(build(pattern))
    assert [l.start for l in layoffs] == sorted(l.start for l in layoffs)


# -- cross training --------------------------------------------------------

def test_cross_training_does_not_disguise_a_running_layoff():
    # Cycling through an injury is exactly when a layoff must still be found.
    days = build(steady(120) + [0.0] * 30 + steady(120))
    for day in days[120:150]:
        day.kind = "cross"
        day.distance_km = 40.0
        day.duration_s = 5400.0
    assert len(find_layoffs(days)) == 1


# -- promotion to an episode ----------------------------------------------

def test_a_confirmed_layoff_becomes_a_dated_injury_episode():
    layoff = Layoff(start=date(2024, 3, 1), end=date(2024, 4, 1), kind="stoppage",
                    days=32, baseline_weekly_km=70.0, during_weekly_km=0.0,
                    confidence="strong", reason="")
    episode = layoff.to_episode("left achilles", Tissue.TENDON)
    assert episode.onset_date == date(2024, 3, 1)
    assert episode.resolved_date == date(2024, 4, 1)
    assert episode.days_lost == 32
    assert episode.tissue is Tissue.TENDON
    assert not episode.is_open()


def test_an_episode_promoted_from_a_layoff_says_where_it_came_from():
    layoff = Layoff(start=date(2024, 3, 1), end=date(2024, 4, 1), kind="dip",
                    days=32, baseline_weekly_km=70.0, during_weekly_km=20.0,
                    confidence="moderate", reason="")
    assert "training log" in layoff.to_episode("shin", Tissue.BONE).notes


def test_weeks_is_days_over_seven():
    layoff = Layoff(start=date(2024, 3, 1), end=date(2024, 3, 14), kind="stoppage",
                    days=14, baseline_weekly_km=70.0, during_weekly_km=0.0,
                    confidence="moderate", reason="")
    assert layoff.weeks == pytest.approx(2.0)


def test_nothing_is_confirmed_until_a_person_confirms_it():
    pattern = steady(120) + [0.0] * 30 + steady(120)
    assert all(not l.confirmed for l in find_layoffs(build(pattern)))


# -- a gap the log ends inside --------------------------------------------

def test_a_stoppage_running_to_the_end_of_the_log_is_still_proposed():
    # The runner is not running now. This is the case that matters most, and
    # the one a fresh import from an injured runner will always hit.
    layoffs = find_layoffs(build(steady(120) + [0.0] * 40))
    assert len(layoffs) == 1
    assert layoffs[0].kind == "stoppage"
    assert layoffs[0].days == 40


def test_an_open_gap_is_marked_ongoing():
    layoff = find_layoffs(build(steady(120) + [0.0] * 40))[0]
    assert layoff.ongoing
    assert "still open" in layoff.reason
    assert "so far" in layoff.reason


def test_a_gap_that_was_returned_from_is_not_ongoing():
    pattern = steady(120) + [0.0] * 40 + steady(30)
    assert not find_layoffs(build(pattern))[0].ongoing


def test_a_trailing_gap_shorter_than_the_threshold_is_still_ignored():
    pattern = steady(120) + [0.0] * (MIN_STOPPAGE_DAYS - 1)
    assert find_layoffs(build(pattern)) == []


def test_the_open_gap_ends_on_the_last_day_of_the_log():
    days = build(steady(120) + [0.0] * 40)
    layoff = find_layoffs(days)[0]
    assert layoff.end == days[-1].day
    assert layoff.start == days[120].day


def test_an_open_gap_and_an_earlier_closed_one_are_both_reported():
    pattern = steady(90) + [0.0] * 20 + steady(90) + [0.0] * 25
    layoffs = find_layoffs(build(pattern))
    assert [l.ongoing for l in layoffs] == [False, True]
