"""Tests for personal-record detection and the terrain screen.

The screen is the interesting part. A record that counts a net-downhill
point-to-point alongside a flat road race is not a record, it is a comparison
between two different sports, so the tests here care as much about what gets
rejected -- and whether the rejection explains itself -- as about what wins.
"""
from __future__ import annotations

from datetime import date

import pytest

from taper.athlete import RaceResult, Surface, TrainingDay
from taper.records import (
    DISTANCE_TOLERANCE, Effort, _match_distance, detect_records, efforts_from_log,
    efforts_from_races, progression, rejected_efforts, screen_terrain,
)


def effort(distance_m=5000.0, time_s=1200.0, *, surface=Surface.ROAD,
           gain=None, loss=None, day=None, label="5K") -> Effort:
    return Effort(day=day, distance_m=distance_m, time_s=time_s, label=label,
                  surface=surface, elevation_gain_m=gain, elevation_loss_m=loss)


def logged(day: date, km: float, seconds: float, *, surface=Surface.ROAD,
           gain=None, loss=None, kind="easy", name="") -> TrainingDay:
    return TrainingDay(day=day, distance_km=km, duration_s=seconds, surface=surface,
                       elevation_gain_m=gain, elevation_loss_m=loss, kind=kind,
                       name=name)


# -- matching a standard distance -----------------------------------------

@pytest.mark.parametrize("metres, label", [
    (1609.344, "1 mile"),
    (5000.0, "5K"),
    (10000.0, "10K"),
    (21097.5, "Half marathon"),
    (42195.0, "Marathon"),
    (50000.0, "50K"),
])
def test_exact_standard_distances_match(metres, label):
    assert _match_distance(metres)[0] == label


def test_a_small_overshoot_still_counts_because_the_true_split_was_faster():
    assert _match_distance(5000.0 * 1.01)[0] == "5K"


def test_running_well_past_the_distance_does_not_count():
    assert _match_distance(5000.0 * 1.5) is None


def test_falling_short_of_the_distance_does_not_count():
    assert _match_distance(4999.0) is None


def test_the_tolerance_boundary_is_inclusive():
    assert _match_distance(5000.0 * DISTANCE_TOLERANCE)[0] == "5K"
    assert _match_distance(5000.0 * DISTANCE_TOLERANCE + 1.0) is None


# -- the terrain screen ----------------------------------------------------

def test_a_flat_road_effort_is_eligible():
    screened = screen_terrain(effort(gain=10.0, loss=10.0))
    assert screened.eligible
    assert "Flat enough" in screened.reason


def test_a_treadmill_effort_is_rejected_on_belt_calibration():
    screened = screen_terrain(effort(surface=Surface.TREADMILL, gain=0.0, loss=0.0))
    assert not screened.eligible
    assert "calibration" in screened.reason


def test_a_trail_effort_is_rejected_because_the_surface_itself_costs_time():
    screened = screen_terrain(effort(surface=Surface.TRAIL, gain=10.0, loss=10.0))
    assert not screened.eligible
    assert "Trail" in screened.reason


def test_a_track_effort_is_eligible():
    assert screen_terrain(effort(surface=Surface.TRACK, gain=0.0, loss=0.0)).eligible


def test_a_steep_climb_is_rejected_and_the_gradient_is_quoted():
    # 100 m of climb over 5 km is 20 m/km, past the 12 m/km ceiling.
    screened = screen_terrain(effort(gain=100.0, loss=100.0))
    assert not screened.eligible
    assert "20 m of climb per km" in screened.reason


def test_a_gently_rolling_effort_is_still_eligible():
    # 50 m over 5 km is 10 m/km, inside the ceiling.
    assert screen_terrain(effort(gain=50.0, loss=50.0)).eligible


def test_a_net_downhill_effort_is_rejected_before_the_climb_test():
    # Drops 200 m and climbs 20 m over 5 km: 36 m/km net drop.
    screened = screen_terrain(effort(gain=20.0, loss=200.0))
    assert not screened.eligible
    assert "Net downhill" in screened.reason


def test_a_mild_net_drop_is_tolerated():
    # 15 m net drop over 5 km is 3 m/km, inside the 4 m/km allowance.
    assert screen_terrain(effort(gain=5.0, loss=20.0)).eligible


def test_missing_elevation_data_is_allowed_but_says_the_screen_did_not_run():
    screened = screen_terrain(effort())
    assert screened.eligible
    assert "could not be applied" in screened.reason


def test_a_zero_distance_effort_has_no_gradient_to_judge():
    assert effort(distance_m=0.0, gain=50.0).climb_per_km is None
    assert effort(distance_m=0.0, gain=50.0, loss=50.0).net_drop_per_km is None


def test_net_drop_needs_both_gain_and_loss():
    assert effort(gain=20.0).net_drop_per_km is None
    assert effort(loss=20.0).net_drop_per_km is None


# -- pulling efforts out of a log -----------------------------------------

def test_a_logged_run_at_a_standard_distance_becomes_an_effort():
    efforts = efforts_from_log([logged(date(2024, 8, 1), 10.0, 2400.0, gain=20, loss=20)])
    assert [e.label for e in efforts] == ["10K"]
    assert efforts[0].time_s == 2400.0


def test_cross_training_days_are_not_efforts():
    assert efforts_from_log([logged(date(2024, 8, 1), 10.0, 2400.0, kind="cross")]) == []


def test_a_day_with_no_duration_cannot_be_timed():
    day = TrainingDay(day=date(2024, 8, 1), distance_km=10.0)
    assert efforts_from_log([day]) == []


def test_rest_days_are_not_efforts():
    assert efforts_from_log([TrainingDay(day=date(2024, 8, 1), kind="off")]) == []


def test_an_odd_distance_is_not_an_effort_at_any_standard():
    assert efforts_from_log([logged(date(2024, 8, 1), 7.3, 1800.0)]) == []


def test_an_effort_carries_the_activity_name_through():
    efforts = efforts_from_log([logged(date(2024, 8, 1), 10.0, 2400.0, name="Tempo")])
    assert efforts[0].name == "Tempo"


def test_elevation_loss_reaches_the_screen_from_the_log():
    # The whole point of storing loss: without it this downhill 10K would pass.
    day = logged(date(2024, 8, 1), 10.0, 2100.0, gain=20, loss=500)
    assert not efforts_from_log([day])[0].eligible


# -- races -----------------------------------------------------------------

def test_a_race_at_a_standard_distance_gets_that_label():
    races = [RaceResult(distance_m=21097.5, finish_time_s=5400.0,
                        race_date=date(2024, 4, 1), elevation_gain_m=50.0)]
    assert efforts_from_races(races)[0].label == "Half marathon"


def test_a_race_at_an_odd_distance_is_labelled_by_its_length():
    races = [RaceResult(distance_m=8000.0, finish_time_s=1800.0)]
    assert efforts_from_races(races)[0].label == "8.0 km"


def test_a_race_is_screened_on_climb_alone_because_loss_is_rarely_recorded():
    races = [RaceResult(distance_m=5000.0, finish_time_s=1200.0, elevation_gain_m=10.0)]
    assert efforts_from_races(races)[0].eligible


def test_a_mountain_race_is_marked_ineligible_but_not_discarded():
    races = [RaceResult(distance_m=10000.0, finish_time_s=3600.0,
                        race_date=date(2024, 6, 1), elevation_gain_m=800.0)]
    screened = efforts_from_races(races)
    assert len(screened) == 1
    assert not screened[0].eligible


def test_races_are_sourced_as_races():
    races = [RaceResult(distance_m=5000.0, finish_time_s=1200.0)]
    assert efforts_from_races(races)[0].source == "race"


# -- records ---------------------------------------------------------------

def test_the_fastest_eligible_effort_at_each_distance_wins():
    days = [
        logged(date(2024, 5, 1), 5.0, 1300.0, gain=10, loss=10),
        logged(date(2024, 6, 1), 5.0, 1180.0, gain=10, loss=10),
        logged(date(2024, 7, 1), 10.0, 2500.0, gain=10, loss=10),
    ]
    records = {r.label: r for r in detect_records(days, [])}
    assert records["5K"].effort.time_s == 1180.0
    assert records["5K"].set_on == date(2024, 6, 1)
    assert records["10K"].effort.time_s == 2500.0


def test_an_ineligible_effort_never_becomes_a_record():
    days = [
        logged(date(2024, 5, 1), 5.0, 1300.0, gain=10, loss=10),
        logged(date(2024, 6, 1), 5.0, 1000.0, gain=10, loss=400),   # downhill
    ]
    records = {r.label: r for r in detect_records(days, [])}
    assert records["5K"].effort.time_s == 1300.0


def test_a_record_reports_what_it_beat():
    days = [
        logged(date(2024, 5, 1), 5.0, 1300.0, gain=10, loss=10),
        logged(date(2024, 6, 1), 5.0, 1180.0, gain=10, loss=10),
    ]
    record = detect_records(days, [])[0]
    assert record.previous.time_s == 1300.0
    assert record.improvement_s == pytest.approx(120.0)


def test_a_first_ever_effort_has_nothing_to_beat():
    days = [logged(date(2024, 5, 1), 5.0, 1300.0, gain=10, loss=10)]
    record = detect_records(days, [])[0]
    assert record.previous is None
    assert record.improvement_s is None


def test_a_slower_run_after_the_record_is_not_counted_as_the_previous_one():
    days = [
        logged(date(2024, 5, 1), 5.0, 1180.0, gain=10, loss=10),   # the record
        logged(date(2024, 6, 1), 5.0, 1300.0, gain=10, loss=10),   # later, slower
    ]
    assert detect_records(days, [])[0].previous is None


def test_records_come_back_shortest_distance_first():
    days = [
        logged(date(2024, 5, 1), 10.0, 2500.0, gain=10, loss=10),
        logged(date(2024, 5, 8), 5.0, 1180.0, gain=10, loss=10),
    ]
    assert [r.label for r in detect_records(days, [])] == ["5K", "10K"]


def test_races_and_training_efforts_compete_for_the_same_record():
    days = [logged(date(2024, 5, 1), 5.0, 1180.0, gain=10, loss=10)]
    races = [RaceResult(distance_m=5000.0, finish_time_s=1100.0,
                        race_date=date(2024, 6, 1), elevation_gain_m=10.0)]
    record = detect_records(days, races)[0]
    assert record.effort.source == "race"
    assert record.effort.time_s == 1100.0


def test_an_empty_log_has_no_records():
    assert detect_records([], []) == []
    assert detect_records(None, None) == []


# -- progression -----------------------------------------------------------

def test_progression_keeps_only_the_efforts_that_were_records_at_the_time():
    days = [
        logged(date(2024, 1, 1), 5.0, 1300.0, gain=10, loss=10),
        logged(date(2024, 2, 1), 5.0, 1320.0, gain=10, loss=10),   # slower, dropped
        logged(date(2024, 3, 1), 5.0, 1250.0, gain=10, loss=10),
        logged(date(2024, 4, 1), 5.0, 1260.0, gain=10, loss=10),   # slower, dropped
        logged(date(2024, 5, 1), 5.0, 1180.0, gain=10, loss=10),
    ]
    chain = progression(days, [], "5K")
    assert [e.time_s for e in chain] == [1300.0, 1250.0, 1180.0]


def test_progression_is_oldest_first():
    days = [
        logged(date(2024, 5, 1), 5.0, 1180.0, gain=10, loss=10),
        logged(date(2024, 1, 1), 5.0, 1300.0, gain=10, loss=10),
    ]
    chain = progression(days, [], "5K")
    assert [e.day for e in chain] == [date(2024, 1, 1), date(2024, 5, 1)]


def test_progression_at_a_distance_never_run_is_empty():
    days = [logged(date(2024, 5, 1), 5.0, 1180.0, gain=10, loss=10)]
    assert progression(days, [], "Marathon") == []


# -- rejections ------------------------------------------------------------

def test_rejected_efforts_are_returned_fastest_first_so_the_ui_can_explain():
    days = [
        logged(date(2024, 5, 1), 5.0, 1000.0, gain=10, loss=400),
        logged(date(2024, 6, 1), 5.0, 1100.0, surface=Surface.TREADMILL),
        logged(date(2024, 7, 1), 5.0, 1180.0, gain=10, loss=10),    # eligible
    ]
    rejected = rejected_efforts(days, [])
    assert [e.time_s for e in rejected] == [1000.0, 1100.0]
    assert all(e.reason for e in rejected)


# -- days that hold more than one activity ---------------------------------

def test_a_day_of_two_runs_is_not_offered_as_one_continuous_effort():
    # Two easy 5Ks twelve hours apart merge to 10 km in 40:00. That is not a
    # 10K, and it must never be presented as one.
    double = logged(date(2024, 8, 1), 10.0, 2400.0, gain=20, loss=20)
    double.sessions = 2
    assert efforts_from_log([double]) == []
    assert detect_records([double], []) == []


def test_a_single_run_day_is_still_an_effort():
    single = logged(date(2024, 8, 1), 10.0, 2400.0, gain=20, loss=20)
    assert single.sessions == 1
    assert len(efforts_from_log([single])) == 1


def test_a_run_paired_with_cross_training_is_excluded_too():
    # Distance counts only the run but duration counts the bike as well, so the
    # implied pace is nonsense.
    mixed = logged(date(2024, 8, 1), 10.0, 6600.0, gain=20, loss=20)
    mixed.sessions = 2
    assert efforts_from_log([mixed]) == []


def test_a_merged_day_is_not_reported_as_a_rejected_effort_either():
    # It was not screened out on terrain; it simply is not an effort.
    double = logged(date(2024, 8, 1), 10.0, 2400.0, gain=20, loss=20)
    double.sessions = 2
    assert rejected_efforts([double], []) == []
