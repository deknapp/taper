"""Tests for the plain-text record export.

The export is a backup, so the tests care about the things that make a backup
worth having twenty years later: that it says who and when, that it never
silently omits a fast run without explaining why, and that it is stable enough
to diff two exports against each other.
"""
from __future__ import annotations

from datetime import date

import pytest

from taper.athlete import AthleteProfile, RaceResult, Surface, TrainingDay
from taper.export import LINE, records_report, suggested_filename

TODAY = date(2026, 9, 4)


def race(distance_m=5000.0, time_s=1180.0, when=date(2024, 6, 1), name="Spring 5K",
         gain=10.0, **kwargs) -> RaceResult:
    return RaceResult(distance_m=distance_m, finish_time_s=time_s, race_date=when,
                      name=name, elevation_gain_m=gain, **kwargs)


def logged(when, km, seconds, *, gain=10.0, loss=10.0, name="", surface=Surface.ROAD):
    return TrainingDay(day=when, distance_km=km, duration_s=seconds,
                       elevation_gain_m=gain, elevation_loss_m=loss, name=name,
                       surface=surface)


def profile(**kwargs) -> AthleteProfile:
    base = AthleteProfile(name="Nate Knapp")
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


# -- the shape of the file -------------------------------------------------

def test_the_export_names_the_runner_and_the_day_it_was_made():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "Nate Knapp" in text
    assert "2026-09-04" in text


def test_an_unnamed_runner_still_gets_a_readable_header():
    text = records_report(AthleteProfile(), today=TODAY)
    assert "unnamed runner" in text


def test_the_export_ends_with_a_single_newline():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert text.endswith("\n")
    assert not text.endswith("\n\n\n")


def test_no_line_carries_trailing_whitespace():
    # Invisible trailing space survives every copy and makes two exports diff
    # against each other for no reason.
    days = [logged(date(2024, 5, 1), 5.0, 1150.0, gain=200, loss=400)]
    text = records_report(profile(races=[race()], training_days=days), today=TODAY)
    assert all(line == line.rstrip() for line in text.splitlines())


def test_lines_stay_narrow_enough_to_read_in_a_terminal():
    days = [logged(date(2024, 5, 1), 5.0, 1150.0, gain=200, loss=400)]
    races = [race(name="A Race With A Really Very Long Name Indeed",
                  place_overall=1234, field_size=9876)]
    text = records_report(profile(races=races, training_days=days), today=TODAY)
    assert max(len(line) for line in text.splitlines()) <= len(LINE) + 2


def test_every_section_is_present_even_when_empty():
    text = records_report(AthleteProfile(), today=TODAY)
    for section in ("PERSONAL RECORDS", "PROGRESSION", "RACES", "HOW TO READ THIS"):
        assert section in text


def test_an_empty_profile_exports_without_raising():
    text = records_report(AthleteProfile(), today=TODAY)
    assert "Nothing yet" in text
    assert "No races on record" in text


# -- records ---------------------------------------------------------------

def test_a_record_appears_with_its_time_and_date():
    text = records_report(profile(races=[race(time_s=1122.0, when=date(2024, 6, 12))]),
                          today=TODAY)
    assert "18:42" in text
    assert "2024-06-12" in text


def test_a_record_says_whether_it_came_from_a_race_or_from_training():
    days = [logged(date(2024, 5, 1), 10.0, 2400.0, name="Tempo")]
    text = records_report(profile(races=[race()], training_days=days), today=TODAY)
    assert "race" in text
    assert "training" in text


def test_an_improvement_on_a_previous_best_is_shown_as_a_negative_delta():
    races = [race(time_s=1180.0, when=date(2023, 4, 1)),
             race(time_s=1122.0, when=date(2024, 6, 12), name="Bosque")]
    text = records_report(profile(races=races), today=TODAY)
    assert "-0:58" in text
    assert "on the previous best" in text


def test_a_first_record_has_no_delta_line():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "on the previous best" not in text


# -- progression -----------------------------------------------------------

def test_a_progression_lists_each_successive_best():
    races = [race(time_s=1200.0, when=date(2022, 4, 1)),
             race(time_s=1180.0, when=date(2023, 4, 1)),
             race(time_s=1122.0, when=date(2024, 6, 1))]
    text = records_report(profile(races=races), today=TODAY)
    progression_block = text.split("PROGRESSION")[1].split("RACES")[0]
    for stamp in ("2022-04-01", "2023-04-01", "2024-06-01"):
        assert stamp in progression_block


def test_a_single_result_is_not_a_progression():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "No distance has been improved on yet" in text


def test_a_slower_result_does_not_enter_the_progression():
    races = [race(time_s=1122.0, when=date(2022, 4, 1)),
             race(time_s=1300.0, when=date(2023, 4, 1))]
    text = records_report(profile(races=races), today=TODAY)
    assert "No distance has been improved on yet" in text


# -- races -----------------------------------------------------------------

def test_races_are_listed_oldest_first():
    races = [race(when=date(2024, 6, 1), name="Later"),
             race(when=date(2022, 6, 1), name="Earlier")]
    text = records_report(profile(races=races), today=TODAY)
    block = text.split("RACES")[1]
    assert block.index("Earlier") < block.index("Later")


def test_a_placing_is_recorded_with_the_field_size():
    text = records_report(profile(races=[race(place_overall=12, field_size=430)]),
                          today=TODAY)
    assert "12/430" in text


def test_a_placing_without_a_field_size_still_shows():
    text = records_report(profile(races=[race(place_overall=12)]), today=TODAY)
    assert "12" in text


def test_an_undated_race_is_still_exported():
    text = records_report(profile(races=[race(when=None, name="Some Race")]),
                          today=TODAY)
    assert "Some Race" in text
    assert "undated" in text


# -- the rejected efforts --------------------------------------------------

def test_a_screened_out_effort_is_kept_with_its_reason():
    days = [logged(date(2024, 5, 1), 5.0, 1150.0, gain=200, loss=400,
                   name="Downhill blast")]
    text = records_report(profile(training_days=days), today=TODAY)
    assert "EFFORTS THE TERRAIN SCREEN EXCLUDED" in text
    assert "Net downhill" in text


def test_a_clean_log_has_no_exclusions_section():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "EXCLUDED" not in text


def test_a_long_reason_is_wrapped_rather_than_running_off_the_page():
    days = [logged(date(2024, 5, 1), 5.0, 1150.0, gain=200, loss=400)]
    text = records_report(profile(training_days=days), today=TODAY)
    reason_lines = [l for l in text.splitlines() if "downhill" in l.lower()]
    assert len(reason_lines) >= 1
    assert all(len(l) <= len(LINE) for l in reason_lines)


def test_a_flood_of_exclusions_is_truncated_with_a_count():
    days = [logged(date(2024, 1, 1) + __import__("datetime").timedelta(days=i),
                   5.0, 1150.0 + i, gain=200, loss=400) for i in range(40)]
    text = records_report(profile(training_days=days), today=TODAY)
    assert "and 15 more" in text


def test_the_export_explains_the_screen_it_applied():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "12 m/km" in text
    assert "4 m/km" in text


def test_the_export_explains_why_a_multi_activity_day_is_absent():
    text = records_report(profile(races=[race()]), today=TODAY)
    assert "more than one" in text


# -- the filename ----------------------------------------------------------

def test_the_suggested_filename_carries_the_runner_and_the_date():
    assert suggested_filename(profile(), TODAY) == "nate-knapp-records-2026-09-04.txt"


def test_a_messy_name_becomes_a_safe_filename():
    assert suggested_filename(AthleteProfile(name="Ann-Marie O'Neill!"), TODAY) == \
        "ann-marie-o-neill-records-2026-09-04.txt"


def test_an_unnamed_runner_gets_a_usable_filename():
    assert suggested_filename(AthleteProfile(), TODAY) == "runner-records-2026-09-04.txt"


# -- stability -------------------------------------------------------------

def test_exporting_the_same_profile_twice_gives_the_same_text():
    p = profile(races=[race(), race(time_s=2400.0, distance_m=10000.0)])
    assert records_report(p, today=TODAY) == records_report(p, today=TODAY)


def test_a_merged_day_is_not_exported_as_a_record():
    double = logged(date(2024, 5, 1), 10.0, 2400.0, name="AM / PM")
    double.sessions = 2
    text = records_report(profile(training_days=[double]), today=TODAY)
    assert "Nothing yet" in text
