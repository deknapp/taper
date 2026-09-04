"""Import a Strava bulk archive export.

This is the *export*, not the API. Strava's API Policy s5.3 forbids using
API-sourced data in connection with AI applications, but s6.6 explicitly
preserves the separate user-facing right: "Each Strava user has the right to
access and export the user's own Strava data, free of charge, through the Bulk
Data Export Tool... Nothing in this Agreement is intended to limit or condition
that user-facing right." A runner's own archive, imported into their own local
log, is not API-sourced and is not governed by the developer policy.

Get one at: Settings -> My Account -> Download or Delete Your Account ->
Request Archive. The file we want is `activities.csv`.
"""
from __future__ import annotations

import csv
import io
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from taper.athlete import Surface, TrainingDay

# Activity types that count as running. Strava's labels vary by client version.
RUN_TYPES = {"run", "trail run", "treadmill run", "virtual run", "track run",
             "race", "long run", "workout run"}
# Counted as training load but not as running mileage.
CROSS_TYPES = {"ride", "virtual ride", "e-bike ride", "swim", "hike", "walk",
               "elliptical", "rowing", "weight training", "workout", "crossfit",
               "stair-stepper", "nordic ski", "backcountry ski", "alpine ski",
               "snowboard", "yoga", "canoeing", "kayaking", "stand up paddling"}

TRAIL_TYPES = {"trail run"}
TREADMILL_TYPES = {"treadmill run", "virtual run"}
TRACK_TYPES = {"track run"}


def _normalise(header: str) -> str:
    return re.sub(r"\s+", " ", header.strip().lower())


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_DATE_FORMATS = (
    "%b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %H:%M:%S",
    "%d %b %Y, %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d",
)


def _parse_datetime(value: str) -> date | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Last resort: pull a bare ISO date out of whatever this is.
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


@dataclass
class Activity:
    """One row of activities.csv, normalised."""

    day: date
    activity_type: str
    name: str = ""
    distance_m: float | None = None
    moving_time_s: float | None = None
    elapsed_time_s: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    rpe: float | None = None
    elevation_gain_m: float | None = None
    elevation_loss_m: float | None = None

    @property
    def is_run(self) -> bool:
        return self.activity_type in RUN_TYPES

    @property
    def is_cross(self) -> bool:
        return self.activity_type in CROSS_TYPES

    @property
    def surface(self) -> Surface:
        if self.activity_type in TRAIL_TYPES:
            return Surface.TRAIL
        if self.activity_type in TREADMILL_TYPES:
            return Surface.TREADMILL
        if self.activity_type in TRACK_TYPES:
            return Surface.TRACK
        return Surface.ROAD


@dataclass
class StravaImport:
    activities: list[Activity] = field(default_factory=list)
    days: list[TrainingDay] = field(default_factory=list)
    rows_seen: int = 0
    runs_used: int = 0
    cross_used: int = 0
    skipped_types: dict[str, int] = field(default_factory=dict)
    rest_days_filled: int = 0
    first_day: date | None = None
    last_day: date | None = None
    warnings: list[str] = field(default_factory=list)


def _column_indices(header: list[str]) -> dict[str, list[int]]:
    """Map normalised header name -> every column index that carries it.

    Strava's export genuinely repeats header names -- 'Distance' and 'Elapsed
    Time' each appear twice, in different units -- so csv.DictReader silently
    drops one. We index by position instead and disambiguate below.
    """
    indices: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(header):
        base = re.sub(r"\.\d+$", "", _normalise(name))  # strip a '.1' de-dup suffix
        indices[base].append(i)
    return dict(indices)


def _pick_distance_column(rows: list[list[str]], candidates: list[int]) -> tuple[int, float]:
    """Choose which 'Distance' column is metres, and the factor to reach metres.

    Strava writes distance twice: once in the athlete's display units (km or
    miles) and once in metres. Rather than trust column order, compare the
    magnitudes -- the metres column is ~1000x the kilometre one.
    """
    medians: list[tuple[int, float]] = []
    for idx in candidates:
        values = [v for v in (_to_float(r[idx]) if idx < len(r) else None for r in rows)
                  if v and v > 0]
        if values:
            medians.append((idx, statistics.median(values)))

    if not medians:
        return candidates[0], 1.0
    if len(medians) >= 2:
        medians.sort(key=lambda pair: pair[1])
        smallest, largest = medians[0], medians[-1]
        if largest[1] > smallest[1] * 100:
            return largest[0], 1.0  # already metres

    idx, median = max(medians, key=lambda pair: pair[1])
    if median > 1000:
        return idx, 1.0            # metres
    if median > 100:
        return idx, 1.0            # implausible as km for a run; assume metres
    return idx, 1000.0             # kilometres


def parse_activities_csv(text: str) -> StravaImport:
    """Parse activities.csv into per-day training entries.

    Multiple activities on one day are merged, because the log stores one row
    per day: distances and durations add, heart rate is duration-weighted, and
    the hardest reported effort wins.
    """
    result = StravaImport()
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        result.warnings.append("The file is empty.")
        return result

    rows = [r for r in reader if any(cell.strip() for cell in r)]
    result.rows_seen = len(rows)
    if not rows:
        result.warnings.append("No activity rows found in the file.")
        return result

    cols = _column_indices(header)

    def first_index(*names: str) -> int | None:
        for name in names:
            if name in cols:
                return cols[name][0]
        return None

    date_idx = first_index("activity date", "date")
    type_idx = first_index("activity type", "type")
    if date_idx is None or type_idx is None:
        result.warnings.append(
            "This does not look like a Strava activities.csv -- it has no "
            "'Activity Date' / 'Activity Type' columns. Make sure you picked "
            "activities.csv out of the archive rather than another file.")
        return result

    name_idx = first_index("activity name", "name")
    moving_idx = first_index("moving time")
    elapsed_idx = first_index("elapsed time")
    avg_hr_idx = first_index("average heart rate")
    max_hr_idx = first_index("max heart rate")
    rpe_idx = first_index("perceived exertion")
    gain_idx = first_index("elevation gain")
    loss_idx = first_index("elevation loss")

    distance_idx, distance_factor = (None, 1.0)
    if "distance" in cols:
        distance_idx, distance_factor = _pick_distance_column(rows, cols["distance"])

    def cell(row: list[str], idx: int | None) -> str | None:
        return row[idx] if idx is not None and idx < len(row) else None

    skipped: Counter[str] = Counter()
    activities: list[Activity] = []

    for row in rows:
        when = _parse_datetime(cell(row, date_idx) or "")
        if when is None:
            skipped["unparseable date"] += 1
            continue

        activity_type = _normalise(cell(row, type_idx) or "")
        distance = _to_float(cell(row, distance_idx))
        activity = Activity(
            day=when,
            activity_type=activity_type,
            name=(cell(row, name_idx) or "").strip(),
            distance_m=(distance * distance_factor) if distance else None,
            moving_time_s=_to_float(cell(row, moving_idx)),
            elapsed_time_s=_to_float(cell(row, elapsed_idx)),
            avg_hr=_to_float(cell(row, avg_hr_idx)),
            max_hr=_to_float(cell(row, max_hr_idx)),
            rpe=_to_float(cell(row, rpe_idx)),
            elevation_gain_m=_to_float(cell(row, gain_idx)),
            elevation_loss_m=_to_float(cell(row, loss_idx)),
        )

        if not (activity.is_run or activity.is_cross):
            skipped[activity_type or "(blank type)"] += 1
            continue
        activities.append(activity)

    result.activities = activities
    result.skipped_types = dict(skipped)

    if not activities:
        result.warnings.append(
            "No runs or cross-training found. Every row was a type the importer "
            "does not recognise.")
        return result

    result.days = _merge_into_days(activities, result)
    result.first_day = min(d.day for d in result.days)
    result.last_day = max(d.day for d in result.days)
    result.runs_used = sum(1 for a in activities if a.is_run)
    result.cross_used = sum(1 for a in activities if a.is_cross)

    if not any(a.avg_hr for a in activities):
        result.warnings.append(
            "No heart-rate data in this export, so training load is derived from "
            "pace and distance rather than measured. It still works; it is just "
            "less precise.")
    if not any(a.rpe for a in activities):
        result.warnings.append(
            "No perceived-exertion ratings in this export. Strava records them "
            "only if you fill them in, and they are the cheapest way to improve "
            "load accuracy going forward.")
    return result


def _merge_into_days(activities: list[Activity], result: StravaImport) -> list[TrainingDay]:
    """Collapse activities to one row per day, then fill the rest days.

    Filling rest days matters more than it looks: a Strava export is a complete
    record, so a day with no activity is a day that was genuinely not trained.
    Recording that explicitly is what lets the Banister model decay fatigue
    correctly instead of treating the gap as unknown.
    """
    by_day: dict[date, list[Activity]] = defaultdict(list)
    for a in activities:
        by_day[a.day].append(a)

    days: list[TrainingDay] = []
    for day, entries in sorted(by_day.items()):
        runs = [a for a in entries if a.is_run]
        chosen = runs or entries

        def _seconds(activity: Activity) -> float:
            return activity.moving_time_s or activity.elapsed_time_s or 0.0

        distance_m = sum(a.distance_m or 0.0 for a in runs)
        duration = sum(_seconds(a) for a in entries)
        run_duration = sum(_seconds(a) for a in runs)

        hr_entries = [(a.avg_hr, (a.moving_time_s or a.elapsed_time_s or 0.0))
                      for a in entries if a.avg_hr]
        total_hr_time = sum(t for _, t in hr_entries)
        avg_hr = (round(sum(hr * t for hr, t in hr_entries) / total_hr_time)
                  if total_hr_time else None)

        rpes = [a.rpe for a in entries if a.rpe]
        gains = [a.elevation_gain_m for a in entries if a.elevation_gain_m is not None]
        losses = [a.elevation_loss_m for a in entries if a.elevation_loss_m is not None]

        names = [a.name for a in chosen if a.name]
        days.append(TrainingDay(
            day=day,
            distance_km=round(distance_m / 1000.0, 3),
            duration_s=duration or None,
            avg_hr=avg_hr,
            rpe=max(rpes) if rpes else None,
            elevation_gain_m=sum(gains) if gains else None,
            elevation_loss_m=sum(losses) if losses else None,
            surface=chosen[0].surface,
            kind="easy" if runs else "cross",
            sessions=len(entries),
            runs=len(runs),
            run_duration_s=run_duration or None,
            source="strava",
            name=" / ".join(names[:3]),
        ))

    filled = _fill_rest_days(days)
    result.rest_days_filled = len(filled) - len(days)
    return filled


def _fill_rest_days(days: list[TrainingDay]) -> list[TrainingDay]:
    if not days:
        return days
    have = {d.day for d in days}
    start, end = min(have), max(have)

    out = list(days)
    day = start
    while day <= end:
        if day not in have:
            out.append(TrainingDay(day=day, kind="off", source="strava",
                                   notes="No activity recorded in the Strava export."))
        day += timedelta(days=1)
    return sorted(out, key=lambda d: d.day)
