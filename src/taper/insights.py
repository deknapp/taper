"""Everything we can derive from a filled-in profile without simulating anything.

This is what makes the intake form feel worth filling in: type four race results
and the form can already tell you your fitness trend across years, what you
should be capable of at every other distance, and where your training is out of
step with your goal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from taper.athlete import AthleteProfile, RaceResult
from taper.physiology import predict_time, vdot_from_race
from taper.units import format_duration, km_to_miles

STANDARD_DISTANCES: list[tuple[str, float]] = [
    ("1 mile", 1609.344),
    ("5K", 5000.0),
    ("10K", 10000.0),
    ("Half marathon", 21097.5),
    ("Marathon", 42195.0),
]

# VO2max falls by roughly 10% per decade in trained masters runners who keep
# training (Tanaka & Seals, J Physiol 586:55-63, 2008). Used only to age a stale
# race result forward, and only as a rough prior -- it is not a detraining model.
_VDOT_DECAY_PER_YEAR = 0.01


@dataclass
class RaceInsight:
    race: RaceResult
    vdot: float

    @property
    def year(self) -> int | None:
        return self.race.race_date.year if self.race.race_date else None


@dataclass
class FitnessEstimate:
    vdot: float
    source: RaceInsight | None
    staleness_years: float | None
    confidence: str  # 'measured', 'aged', or 'estimated'
    note: str


def race_insights(profile: AthleteProfile) -> list[RaceInsight]:
    """Every race with its implied VDOT, most recent first."""
    out = [RaceInsight(race=r, vdot=vdot_from_race(r.distance_m, r.finish_time_s))
           for r in profile.races if r.distance_m > 0 and r.finish_time_s > 0]
    return sorted(out, key=lambda i: (i.race.race_date or date.min), reverse=True)


def best_per_year(profile: AthleteProfile) -> list[tuple[int, float]]:
    """Peak VDOT per calendar year: the career arc, and the thing worth charting.

    Gaps in this series are as informative as the values. A missing year usually
    means injury or life, and the sim can seed a more fragile athlete for it.
    """
    peaks: dict[int, float] = {}
    for insight in race_insights(profile):
        year = insight.year
        if year is None:
            continue
        peaks[year] = max(peaks.get(year, 0.0), insight.vdot)
    return sorted(peaks.items())


def current_fitness(profile: AthleteProfile, today: date | None = None) -> FitnessEstimate | None:
    """Best estimate of present-day VDOT, with an honest confidence label.

    A race in the last year is taken at face value. An older one is decayed by a
    masters-athlete prior, which assumes they kept training -- if they did not,
    the estimate is generous, and the form says so rather than hiding it.
    """
    insights = race_insights(profile)
    if not insights:
        return None
    today = today or date.today()

    dated = [i for i in insights if i.race.race_date]
    if not dated:
        best = max(insights, key=lambda i: i.vdot)
        return FitnessEstimate(
            vdot=best.vdot, source=best, staleness_years=None, confidence="estimated",
            note="No dates on any result, so this is your best race ever, whenever it was.",
        )

    recent = [i for i in dated if (today - i.race.race_date).days <= 365]
    if recent:
        best = max(recent, key=lambda i: i.vdot)
        return FitnessEstimate(
            vdot=best.vdot, source=best, staleness_years=0.0, confidence="measured",
            note="Based on your best race in the last 12 months.",
        )

    best = max(dated, key=lambda i: i.vdot)
    years_stale = (today - best.race.race_date).days / 365.2425
    decayed = best.vdot * (1.0 - _VDOT_DECAY_PER_YEAR) ** years_stale
    return FitnessEstimate(
        vdot=decayed, source=best, staleness_years=years_stale, confidence="aged",
        note=(f"Your best race is {years_stale:.1f} years old, so this ages it forward "
              f"from VDOT {best.vdot:.1f} assuming you kept training. If you took time "
              f"off, treat it as optimistic."),
    )


def equivalent_performances(vdot: float) -> list[tuple[str, float]]:
    """What this fitness predicts at every standard distance."""
    return [(label, predict_time(distance, vdot)) for label, distance in STANDARD_DISTANCES]


def formatted_equivalents(vdot: float) -> list[tuple[str, str]]:
    return [(label, format_duration(t)) for label, t in equivalent_performances(vdot)]


@dataclass
class Flag:
    """A readiness observation. `severity` is 'info' | 'watch' | 'warn'."""

    severity: str
    message: str


def readiness_flags(profile: AthleteProfile, today: date | None = None) -> list[Flag]:
    """Sanity checks between where the runner is and where they say they are going.

    These are heuristics from coaching practice, not findings -- they earn their
    place by catching the obvious mismatches before the sim wastes a season on
    them.
    """
    flags: list[Flag] = []
    today = today or date.today()
    training = profile.training
    goal = profile.goal

    if training.current_weekly_km is None:
        flags.append(Flag("warn", "No recent weekly mileage. This is the single most "
                                  "important input: it sets how much load you can absorb "
                                  "in week one without breaking."))

    if goal is not None:
        weeks = (goal.race_date - today).days / 7.0
        if weeks < 0:
            flags.append(Flag("warn", "Your goal race is in the past."))
        elif weeks < 8 and goal.distance_m >= 42195:
            flags.append(Flag("warn", f"Only {weeks:.0f} weeks to a marathon. That is a "
                                      f"short runway to build from."))

        longest = training.longest_recent_run_km
        if longest is not None and goal.distance_m >= 21097.5:
            goal_km = goal.distance_m / 1000.0
            if longest < goal_km * 0.5:
                flags.append(Flag("watch", f"Your longest recent run is "
                                           f"{km_to_miles(longest):.1f} mi, under half the "
                                           f"race distance. Expect the long run to be the "
                                           f"limiter."))

        if training.current_weekly_km and goal.distance_m >= 42195:
            weekly_mi = km_to_miles(training.current_weekly_km)
            if weekly_mi < 25:
                flags.append(Flag("watch", f"{weekly_mi:.0f} mi/week is a thin base for a "
                                           f"marathon. Most plans assume 30+."))

    if training.years_running is not None and training.years_running < 2:
        flags.append(Flag("info", "Under two years of running. Tendon and bone adapt over "
                                  "months to years, well behind cardiovascular fitness, so "
                                  "your engine will outrun your structure. The sim models "
                                  "that gap."))

    recurrent = [i for i in profile.injuries if i.recurrences > 0]
    if recurrent:
        parts = ", ".join(sorted({i.body_part for i in recurrent if i.body_part}))
        flags.append(Flag("watch", f"Recurrent injury history{f' ({parts})' if parts else ''}. "
                                   f"Prior injury is the strongest single predictor of future "
                                   f"injury, so the sim will start you with reduced tolerance "
                                   f"in that tissue."))

    if profile.life.sleep_hours is not None and profile.life.sleep_hours < 7:
        flags.append(Flag("watch", f"{profile.life.sleep_hours:.1f} h of sleep. Recovery is "
                                   f"where adaptation happens; short sleep shifts the whole "
                                   f"fitness/fatigue balance against you."))

    if training.strength_days_per_week == 0:
        flags.append(Flag("info", "No strength training. It has reasonable evidence behind "
                                  "it for injury reduction and is one of the few protective "
                                  "factors that shows up consistently."))

    return flags
