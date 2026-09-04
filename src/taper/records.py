"""Finding personal records in a training log, on terrain that makes them mean something.

A time is only comparable to another time if the ground underneath was
comparable. A net-downhill point-to-point flatters a runner by minutes over a
marathon; a rolling trail loop punishes them by more. So efforts are screened on
terrain before they are allowed to count as a record, and the screen is reported
so a rejected effort can say why it did not qualify.

Scope note: `activities.csv` carries whole-activity totals only, so a 5K PR run
inside a longer workout is invisible here -- what can be detected is an activity
whose *whole* distance lands on a standard one. Extracting best efforts from
within an activity needs the per-activity GPX/FIT files, which the Strava archive
also contains; that is a later upgrade, not a limitation of the idea.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from taper.athlete import RaceResult, Surface, TrainingDay
from taper.physiology import vdot_from_race
from taper.units import format_duration

# Standard distances a record is worth tracking at.
STANDARD_DISTANCES: list[tuple[str, float]] = [
    ("1 mile", 1609.344),
    ("5K", 5000.0),
    ("10K", 10000.0),
    ("15K", 15000.0),
    ("10 mile", 16093.44),
    ("Half marathon", 21097.5),
    ("Marathon", 42195.0),
    ("50K", 50000.0),
]

# An activity counts towards a distance if it covered at least that distance and
# no more than this much extra. Running long and claiming the whole time is
# conservative -- the true split was faster -- so a small overshoot is safe.
DISTANCE_TOLERANCE = 1.02

# Terrain screens, in metres of elevation per kilometre.
MAX_CLIMB_M_PER_KM = 12.0      # beyond gently rolling
MAX_NET_DROP_M_PER_KM = 4.0    # a net-downhill course is not a fair record

# Surfaces where a clock time means what it says.
NORMALISED_SURFACES = {Surface.TRACK, Surface.ROAD}


@dataclass
class Effort:
    """One timed distance, screened for whether it can count as a record."""

    day: date | None
    distance_m: float
    time_s: float
    label: str
    name: str = ""
    source: str = "log"
    surface: Surface = Surface.ROAD
    elevation_gain_m: float | None = None
    elevation_loss_m: float | None = None
    eligible: bool = True
    reason: str = ""

    @property
    def vdot(self) -> float:
        return vdot_from_race(self.distance_m, self.time_s)

    @property
    def formatted_time(self) -> str:
        return format_duration(self.time_s)

    @property
    def climb_per_km(self) -> float | None:
        if self.elevation_gain_m is None or self.distance_m <= 0:
            return None
        return self.elevation_gain_m / (self.distance_m / 1000.0)

    @property
    def net_drop_per_km(self) -> float | None:
        if self.elevation_gain_m is None or self.elevation_loss_m is None:
            return None
        if self.distance_m <= 0:
            return None
        return (self.elevation_loss_m - self.elevation_gain_m) / (self.distance_m / 1000.0)


def _match_distance(distance_m: float) -> tuple[str, float] | None:
    """The standard distance this effort covers, if any."""
    for label, target in STANDARD_DISTANCES:
        if target <= distance_m <= target * DISTANCE_TOLERANCE:
            return label, target
    return None


def screen_terrain(effort: Effort) -> Effort:
    """Decide whether the ground makes this time comparable, and say why not."""
    if effort.surface == Surface.TREADMILL:
        effort.eligible = False
        effort.reason = ("Treadmill. Belt calibration varies enough between machines "
                         "that the time is not comparable to ground running.")
        return effort

    if effort.surface not in NORMALISED_SURFACES:
        effort.eligible = False
        effort.reason = (f"{effort.surface.value.title()} rather than track or road, so "
                         f"the surface itself costs time that a clock cannot separate out.")
        return effort

    climb = effort.climb_per_km
    net_drop = effort.net_drop_per_km

    if climb is None:
        # No elevation data at all. Allow it, but say the screen did not run.
        effort.reason = "No elevation data, so the terrain screen could not be applied."
        return effort

    if net_drop is not None and net_drop > MAX_NET_DROP_M_PER_KM:
        effort.eligible = False
        effort.reason = (f"Net downhill, losing {net_drop:.0f} m/km more than it climbs. "
                         f"Point-to-point drops flatter a time rather than earning it.")
        return effort

    if climb > MAX_CLIMB_M_PER_KM:
        effort.eligible = False
        effort.reason = (f"{climb:.0f} m of climb per km, beyond gently rolling. The time "
                         f"is honest but it is not comparable to a flat one.")
        return effort

    effort.reason = f"Flat enough to count: {climb:.0f} m/km of climb."
    return effort


def efforts_from_log(days: list[TrainingDay]) -> list[Effort]:
    """Every logged day whose whole distance lands on a standard distance."""
    efforts: list[Effort] = []
    for day in days:
        if day.kind == "cross" or day.distance_km <= 0 or not day.duration_s:
            continue
        # Two runs on one day were added together, so the row describes no
        # single continuous run -- two easy 5Ks twelve hours apart would read as
        # a 10K record. One run is fine however much else was done that day,
        # provided the time counted is the time spent running.
        if day.runs > 1:
            continue
        time_s = day.run_duration_s or day.duration_s
        if not time_s:
            continue
        distance_m = day.distance_km * 1000.0
        matched = _match_distance(distance_m)
        if matched is None:
            continue
        label, _ = matched
        efforts.append(screen_terrain(Effort(
            day=day.day, distance_m=distance_m, time_s=time_s, label=label,
            name=day.name, source=day.source, surface=day.surface,
            elevation_gain_m=day.elevation_gain_m,
            elevation_loss_m=day.elevation_loss_m)))
    return efforts


def efforts_from_races(races: list[RaceResult]) -> list[Effort]:
    """Races, screened on the same terms as training efforts.

    A race is still a race even if it does not qualify as a record -- a downhill
    marathon time is real, it just is not comparable. The screen marks it rather
    than discarding it.
    """
    efforts: list[Effort] = []
    for race in races:
        matched = _match_distance(race.distance_m)
        label = matched[0] if matched else f"{race.distance_m / 1000:.1f} km"
        effort = Effort(
            day=race.race_date, distance_m=race.distance_m, time_s=race.finish_time_s,
            label=label, name=race.name, source="race", surface=race.surface,
            elevation_gain_m=race.elevation_gain_m)
        # Races rarely record elevation loss; without it, judge on climb alone.
        efforts.append(screen_terrain(effort))
    return efforts


@dataclass
class PersonalRecord:
    label: str
    distance_m: float
    effort: Effort
    set_on: date | None
    previous: Effort | None = None

    @property
    def improvement_s(self) -> float | None:
        return (self.previous.time_s - self.effort.time_s) if self.previous else None


def detect_records(days: list[TrainingDay] | None = None,
                   races: list[RaceResult] | None = None) -> list[PersonalRecord]:
    """Best eligible effort at each standard distance, with what it beat."""
    efforts = efforts_from_log(days or []) + efforts_from_races(races or [])
    eligible = [e for e in efforts if e.eligible]

    by_label: dict[str, list[Effort]] = {}
    for effort in eligible:
        by_label.setdefault(effort.label, []).append(effort)

    records: list[PersonalRecord] = []
    for label, target in STANDARD_DISTANCES:
        candidates = by_label.get(label)
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda e: e.time_s)
        best = ranked[0]

        # What stood before this one was set, chronologically.
        earlier = [e for e in candidates
                   if e.day and best.day and e.day < best.day and e.time_s > best.time_s]
        previous = min(earlier, key=lambda e: e.time_s) if earlier else None

        records.append(PersonalRecord(label=label, distance_m=target, effort=best,
                                      set_on=best.day, previous=previous))
    return records


def progression(days: list[TrainingDay] | None, races: list[RaceResult] | None,
                label: str) -> list[Effort]:
    """The chain of successive bests at one distance, oldest first.

    Only efforts that were a record *at the time* appear, which is what makes
    the shape of a career legible rather than a scatter of every run.
    """
    efforts = [e for e in efforts_from_log(days or []) + efforts_from_races(races or [])
               if e.eligible and e.label == label and e.day]
    efforts.sort(key=lambda e: e.day)

    chain: list[Effort] = []
    best = float("inf")
    for effort in efforts:
        if effort.time_s < best:
            best = effort.time_s
            chain.append(effort)
    return chain


def rejected_efforts(days: list[TrainingDay] | None = None,
                     races: list[RaceResult] | None = None) -> list[Effort]:
    """Fast efforts the terrain screen threw out, so the UI can explain itself."""
    efforts = efforts_from_log(days or []) + efforts_from_races(races or [])
    return sorted((e for e in efforts if not e.eligible), key=lambda e: e.time_s)
