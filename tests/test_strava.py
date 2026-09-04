"""Tests for the Strava bulk-export importer.

The fixtures here imitate the real `activities.csv` closely on the points that
have actually bitten us: repeated header names, a display-units distance column
sitting next to a metres one, and Strava's US-style date stamps.
"""
from __future__ import annotations

import csv
import io
from datetime import date

import pytest

from taper.athlete import Surface
from taper.importers.strava import (
    Activity, _parse_datetime, _pick_distance_column, parse_activities_csv,
)


def csv_text(header: list[str], rows: list[list[object]]) -> str:
    """Build a CSV the way Strava does -- quoted, because its dates hold commas."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if c is None else c for c in row])
    return buf.getvalue()


# The columns the importer actually reads, in roughly export order. 'Distance'
# appears twice on purpose: Strava writes display units first, metres second.
HEADER = ["Activity ID", "Activity Date", "Activity Name", "Activity Type",
          "Elapsed Time", "Distance", "Max Heart Rate", "Moving Time",
          "Distance", "Elevation Gain", "Elevation Loss", "Average Heart Rate",
          "Perceived Exertion"]


def row(day: str, name: str, kind: str, *, km: float = 0.0, moving_s: float = 0.0,
        elapsed_s: float | None = None, avg_hr: object = None, max_hr: object = None,
        gain: object = None, loss: object = None, rpe: object = None,
        activity_id: int = 1) -> list[object]:
    return [activity_id, day, name, kind, elapsed_s if elapsed_s is not None else moving_s,
            km, max_hr, moving_s, km * 1000.0, gain, loss, avg_hr, rpe]


# -- date parsing ----------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Aug 15, 2024, 6:12:03 AM", date(2024, 8, 15)),
    ("Aug 15, 2024, 18:12:03", date(2024, 8, 15)),
    ("15 Aug 2024, 06:12:03", date(2024, 8, 15)),
    ("2024-08-15 06:12:03", date(2024, 8, 15)),
    ("2024-08-15T06:12:03Z", date(2024, 8, 15)),
    ("2024-08-15", date(2024, 8, 15)),
])
def test_parses_every_date_format_strava_has_emitted(text, expected):
    assert _parse_datetime(text) == expected


def test_falls_back_to_a_bare_iso_date_inside_unknown_text():
    assert _parse_datetime("recorded on 2024-08-15 somewhere") == date(2024, 8, 15)


def test_unparseable_dates_are_none_rather_than_raising():
    assert _parse_datetime("last Tuesday") is None
    assert _parse_datetime("") is None


def test_rows_with_unparseable_dates_are_skipped_and_counted():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "Good", "Run", km=10, moving_s=3000),
        row("whenever", "Bad", "Run", km=10, moving_s=3000),
    ])
    result = parse_activities_csv(text)
    assert result.rows_seen == 2
    assert result.runs_used == 1
    assert result.skipped_types["unparseable date"] == 1


# -- the two Distance columns ---------------------------------------------

def test_picks_the_metres_distance_column_over_the_kilometre_one():
    rows = [["10.0", "10000.0"], ["21.1", "21100.0"], ["5.0", "5000.0"]]
    idx, factor = _pick_distance_column(rows, [0, 1])
    assert (idx, factor) == (1, 1.0)


def test_picks_the_metres_column_when_display_units_are_miles():
    rows = [["6.2", "10000.0"], ["13.1", "21100.0"], ["3.1", "5000.0"]]
    idx, factor = _pick_distance_column(rows, [0, 1])
    assert (idx, factor) == (1, 1.0)


def test_a_lone_kilometre_column_is_scaled_up_to_metres():
    rows = [["10.0"], ["21.1"], ["5.0"]]
    assert _pick_distance_column(rows, [0]) == (0, 1000.0)


def test_a_lone_metres_column_is_left_alone():
    rows = [["10000"], ["21100"], ["5000"]]
    assert _pick_distance_column(rows, [0]) == (0, 1.0)


def test_a_pandas_style_dedup_suffix_is_treated_as_the_same_column():
    header = [h if h != "Distance" else "Distance" for h in HEADER]
    header[8] = "Distance.1"          # what pandas does to a repeated name
    text = csv_text(header, [row("Aug 15, 2024, 6:00:00 AM", "AM", "Run",
                                 km=10, moving_s=3000)])
    result = parse_activities_csv(text)
    assert result.days[0].distance_km == pytest.approx(10.0)


def test_distance_column_choice_survives_blank_cells():
    rows = [["", ""], ["10.0", "10000.0"], ["", "5000.0"]]
    idx, factor = _pick_distance_column(rows, [0, 1])
    assert (idx, factor) == (1, 1.0)


# -- classification --------------------------------------------------------

def test_runs_and_cross_training_are_counted_separately():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "AM", "Run", km=10, moving_s=3000),
        row("Aug 16, 2024, 6:00:00 AM", "Spin", "Ride", km=30, moving_s=3600),
    ])
    result = parse_activities_csv(text)
    assert (result.runs_used, result.cross_used) == (1, 1)


def test_unrecognised_types_are_skipped_and_reported_by_name():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "AM", "Run", km=10, moving_s=3000),
        row("Aug 16, 2024, 6:00:00 AM", "?", "Kitesurf", km=0, moving_s=600),
        row("Aug 17, 2024, 6:00:00 AM", "?", "Kitesurf", km=0, moving_s=600),
    ])
    result = parse_activities_csv(text)
    assert result.skipped_types == {"kitesurf": 2}


def test_activity_type_matching_ignores_case_and_spacing():
    text = csv_text(HEADER, [row("Aug 15, 2024, 6:00:00 AM", "AM", "  TRAIL   Run ",
                                 km=10, moving_s=3000)])
    result = parse_activities_csv(text)
    assert result.runs_used == 1
    assert result.days[0].surface == Surface.TRAIL


@pytest.mark.parametrize("kind, surface", [
    ("Run", Surface.ROAD),
    ("Trail Run", Surface.TRAIL),
    ("Treadmill Run", Surface.TREADMILL),
    ("Virtual Run", Surface.TREADMILL),
    ("Track Run", Surface.TRACK),
])
def test_surface_is_inferred_from_the_activity_type(kind, surface):
    assert Activity(day=date(2024, 8, 15), activity_type=kind.lower()).surface == surface


# -- merging a day ---------------------------------------------------------

def test_two_runs_on_one_day_merge_into_a_single_row():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "AM", "Run", km=10, moving_s=3000, activity_id=1),
        row("Aug 15, 2024, 6:00:00 PM", "PM", "Run", km=5, moving_s=1500, activity_id=2),
    ])
    result = parse_activities_csv(text)
    assert len(result.days) == 1
    day = result.days[0]
    assert day.distance_km == pytest.approx(15.0)
    assert day.duration_s == pytest.approx(4500)
    assert day.name == "AM / PM"


def test_heart_rate_is_weighted_by_duration_not_averaged_flat():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "Easy", "Run", km=6, moving_s=1800,
            avg_hr=140, activity_id=1),
        row("Aug 15, 2024, 6:00:00 PM", "Long", "Run", km=14, moving_s=3600,
            avg_hr=160, activity_id=2),
    ])
    # (140*1800 + 160*3600) / 5400 = 153.3, not the flat mean of 150.
    assert parse_activities_csv(text).days[0].avg_hr == 153


def test_the_hardest_reported_effort_wins_the_day():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "Easy", "Run", km=6, moving_s=1800,
            rpe=3, activity_id=1),
        row("Aug 15, 2024, 6:00:00 PM", "Reps", "Run", km=8, moving_s=2400,
            rpe=9, activity_id=2),
    ])
    assert parse_activities_csv(text).days[0].rpe == 9


def test_cross_training_time_counts_but_its_distance_does_not():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "Run", "Run", km=10, moving_s=3000, activity_id=1),
        row("Aug 15, 2024, 5:00:00 PM", "Bike", "Ride", km=40, moving_s=3600, activity_id=2),
    ])
    day = parse_activities_csv(text).days[0]
    assert day.distance_km == pytest.approx(10.0)   # the ride's 40 km is not running
    assert day.kind == "easy"                       # a day with a run is a running day


def test_a_cross_training_only_day_is_marked_as_cross():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "Swim", "Swim", km=2, moving_s=2400),
    ])
    day = parse_activities_csv(text).days[0]
    assert day.kind == "cross"
    assert day.distance_km == 0.0


def test_elapsed_time_stands_in_when_moving_time_is_missing():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "AM", "Run", km=10, moving_s=0, elapsed_s=3300),
    ])
    # moving_s of 0 is falsy, so the elapsed column is used instead.
    assert parse_activities_csv(text).days[0].duration_s == pytest.approx(3300)


def test_elevation_gain_and_loss_both_survive_the_merge():
    text = csv_text(HEADER, [
        row("Aug 15, 2024, 6:00:00 AM", "AM", "Run", km=10, moving_s=3000,
            gain=120, loss=340, activity_id=1),
        row("Aug 15, 2024, 6:00:00 PM", "PM", "Run", km=5, moving_s=1500,
            gain=30, loss=10, activity_id=2),
    ])
    day = parse_activities_csv(text).days[0]
    assert day.elevation_gain_m == pytest.approx(150)
    assert day.elevation_loss_m == pytest.approx(350)


def test_only_the_first_three_activity_names_are_kept():
    rows = [row(f"Aug 15, 2024, {h}:00:00 AM", f"Run {i}", "Run", km=3,
                moving_s=900, activity_id=i)
            for i, h in enumerate([6, 7, 8, 9, 10], start=1)]
    day = parse_activities_csv(csv_text(HEADER, rows)).days[0]
    assert day.name == "Run 1 / Run 2 / Run 3"


# -- rest days -------------------------------------------------------------

def test_days_with_no_activity_are_filled_in_as_rest():
    text = csv_text(HEADER, [
        row("Aug 1, 2024, 6:00:00 AM", "First", "Run", km=10, moving_s=3000, activity_id=1),
        row("Aug 5, 2024, 6:00:00 AM", "Last", "Run", km=10, moving_s=3000, activity_id=2),
    ])
    result = parse_activities_csv(text)
    assert [d.day for d in result.days] == [date(2024, 8, n) for n in range(1, 6)]
    assert result.rest_days_filled == 3
    assert all(d.kind == "off" and d.is_rest for d in result.days[1:4])


def test_rest_filling_does_not_run_past_the_ends_of_the_log():
    text = csv_text(HEADER, [
        row("Aug 1, 2024, 6:00:00 AM", "Only", "Run", km=10, moving_s=3000),
    ])
    result = parse_activities_csv(text)
    assert result.first_day == result.last_day == date(2024, 8, 1)
    assert result.rest_days_filled == 0


def test_every_imported_day_is_stamped_with_its_source():
    text = csv_text(HEADER, [
        row("Aug 1, 2024, 6:00:00 AM", "A", "Run", km=10, moving_s=3000, activity_id=1),
        row("Aug 4, 2024, 6:00:00 AM", "B", "Run", km=10, moving_s=3000, activity_id=2),
    ])
    assert {d.source for d in parse_activities_csv(text).days} == {"strava"}


# -- refusing to guess -----------------------------------------------------

def test_an_empty_file_says_so_instead_of_crashing():
    result = parse_activities_csv("")
    assert result.days == []
    assert "empty" in result.warnings[0].lower()


def test_a_header_with_no_rows_is_reported():
    result = parse_activities_csv(",".join(HEADER) + "\n")
    assert result.days == []
    assert "no activity rows" in result.warnings[0].lower()


def test_the_wrong_csv_is_named_as_the_wrong_csv():
    result = parse_activities_csv("Name,Colour\nfoo,red\n")
    assert result.days == []
    assert "activities.csv" in result.warnings[0]


def test_a_file_of_only_unknown_types_explains_itself():
    text = csv_text(HEADER, [row("Aug 15, 2024, 6:00:00 AM", "?", "Kitesurf",
                                 km=0, moving_s=600)])
    result = parse_activities_csv(text)
    assert result.days == []
    assert "does not recognise" in result.warnings[0]


def test_missing_heart_rate_is_flagged_without_blocking_the_import():
    text = csv_text(HEADER, [row("Aug 15, 2024, 6:00:00 AM", "AM", "Run",
                                 km=10, moving_s=3000, rpe=5)])
    result = parse_activities_csv(text)
    assert result.days
    assert any("heart-rate" in w for w in result.warnings)
    assert not any("perceived-exertion" in w for w in result.warnings)


def test_missing_rpe_is_flagged_separately():
    text = csv_text(HEADER, [row("Aug 15, 2024, 6:00:00 AM", "AM", "Run",
                                 km=10, moving_s=3000, avg_hr=150)])
    result = parse_activities_csv(text)
    assert any("perceived-exertion" in w for w in result.warnings)
    assert not any("heart-rate" in w for w in result.warnings)


def test_short_rows_do_not_raise_on_missing_trailing_columns():
    text = "Activity Date,Activity Type,Distance\nAug 15, 2024 6:00:00 AM,Run\n"
    # Ragged rows happen in hand-edited exports; the importer must not explode.
    parse_activities_csv(text)


def test_blank_lines_between_rows_are_ignored():
    text = csv_text(HEADER, [row("Aug 15, 2024, 6:00:00 AM", "AM", "Run",
                                 km=10, moving_s=3000)])
    result = parse_activities_csv(text.replace("\n", "\n\n"))
    assert result.rows_seen == 1
