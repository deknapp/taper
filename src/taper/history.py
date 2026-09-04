"""What real history the simulator is standing on.

The engine will happily run on almost nothing -- three races and a guess at
weekly mileage -- and produce confident-looking numbers. This module exists so
the app can say out loud how much of what you are looking at is measured, how
much is estimated, and which parameters are yours rather than the population's.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

from taper.athlete import AthleteProfile, TrainingDay
from taper.banister import BanisterParams, FitResult, default_params_for, fit_from_history
from taper.insights import current_fitness, race_insights
from taper.load import day_load

# How the load for a day was arrived at, worst to best. Shown in the settings
# view so a runner can see which stretches of their history are solid.
METHOD_QUALITY = {"hr": "measured", "rpe": "reported", "pace": "derived",
                  "distance": "estimated", "rest": "rest"}


@dataclass
class TrainingCoverage:
    first_day: date | None = None
    last_day: date | None = None
    days_logged: int = 0
    days_in_span: int = 0
    total_km: float = 0.0
    rest_days: int = 0
    methods: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def real_days(self) -> int:
        """Days that actually happened, excluding anything we manufactured."""
        return sum(n for src, n in self.sources.items() if src != "estimated")

    @property
    def coverage_fraction(self) -> float:
        """Logged days as a share of the calendar they span.

        Low coverage is the single biggest threat to the model: unlogged days
        are treated as rest, so a half-filled log makes a runner look far
        fresher than they are.
        """
        return self.days_logged / self.days_in_span if self.days_in_span else 0.0


@dataclass
class Calibration:
    params: BanisterParams
    fitted: bool
    note: str
    rmse: float | None = None
    n_observations: int = 0


@dataclass
class RealHistory:
    """Everything true about this runner, and how well we know it."""

    training: TrainingCoverage
    races: int
    dated_races: int
    race_sources: dict[str, int]
    first_race: date | None
    last_race: date | None
    injuries: int
    calibration: Calibration
    warnings: list[str] = field(default_factory=list)


def training_coverage(days: list[TrainingDay], profile: AthleteProfile) -> TrainingCoverage:
    if not days:
        return TrainingCoverage()

    first, last = min(d.day for d in days), max(d.day for d in days)
    vdot_estimate = current_fitness(profile)
    methods: Counter[str] = Counter()
    for d in days:
        result = day_load(
            d, vdot=vdot_estimate.vdot if vdot_estimate else None,
            rest_hr=profile.physiology.resting_hr, max_hr=profile.physiology.max_hr,
            sex=profile.sex)
        methods[result.method] += 1

    return TrainingCoverage(
        first_day=first, last_day=last, days_logged=len(days),
        days_in_span=(last - first).days + 1,
        total_km=sum(d.distance_km for d in days),
        rest_days=sum(1 for d in days if d.is_rest),
        methods=dict(methods),
        sources=dict(Counter(d.source for d in days)),
    )


def daily_loads(profile: AthleteProfile) -> dict[date, float]:
    """The real daily impulse series, ready for the Banister engine."""
    fitness = current_fitness(profile)
    return {
        d.day: day_load(
            d, vdot=fitness.vdot if fitness else None,
            rest_hr=profile.physiology.resting_hr, max_hr=profile.physiology.max_hr,
            sex=profile.sex).value
        for d in profile.training_days}


def calibrate(profile: AthleteProfile) -> Calibration:
    """Fit the runner's own Banister parameters if their history supports it.

    Falls back to population defaults rather than to a bad fit, and says which
    one is in force. An uncalibrated model is fine; one pretending to be
    calibrated is not.
    """
    fitness = current_fitness(profile)
    baseline = (fitness.vdot * 0.6) if fitness else 30.0

    loads = daily_loads(profile)
    performances = {i.race.race_date: i.vdot
                    for i in race_insights(profile) if i.race.race_date}

    if not loads:
        return Calibration(
            params=default_params_for(baseline), fitted=False,
            note=("No training history logged, so the model is running on population "
                  "defaults. Log real training days and it can fit your own "
                  "fitness and fatigue time constants."))

    result: FitResult | None = fit_from_history(loads, performances)
    if result is None:
        return Calibration(
            params=default_params_for(baseline), fitted=False,
            note=("Not enough dated races overlapping your training history to fit "
                  "your own parameters -- the model needs at least three. Using "
                  "population defaults until then."))

    return Calibration(params=result.params, fitted=True, note=result.note,
                       rmse=result.rmse, n_observations=result.n_observations)


def summarise(profile: AthleteProfile, today: date | None = None) -> RealHistory:
    today = today or date.today()
    coverage = training_coverage(profile.training_days, profile)
    dated = [r for r in profile.races if r.race_date]
    calibration = calibrate(profile)

    warnings: list[str] = []
    estimated_days = sum(1 for d in profile.training_days if d.source == "estimated")
    if estimated_days:
        share = estimated_days / len(profile.training_days)
        warnings.append(
            f"{estimated_days} of {len(profile.training_days)} logged days "
            f"({share:.0%}) were manufactured from your weekly-mileage summary, not "
            f"logged as they happened. They give the engine a plausible shape to start "
            f"from, but they are not evidence -- replace them by logging real days.")

    if not profile.training_days:
        warnings.append(
            "No real training days logged. The engine has nothing to run on -- every "
            "projection below starts from an empty history.")
    else:
        if coverage.coverage_fraction < 0.8:
            warnings.append(
                f"Only {coverage.coverage_fraction:.0%} of days between "
                f"{coverage.first_day} and {coverage.last_day} are logged. Missing days "
                f"are treated as rest, so gaps make you look fresher than you were.")
        stale = (today - coverage.last_day).days
        if stale > 14:
            warnings.append(
                f"Your training log stops {stale} days ago. Fitness and fatigue are "
                f"both decayed forward from there, which gets less trustworthy the "
                f"longer the gap.")
        # Both 'pace' and 'distance' days lack a heart rate and an RPE; they
        # differ only in whether the duration was real or assumed. Counting the
        # 'distance' tier alone let a whole Strava import from someone without a
        # monitor pass without comment, while the importer was warning about
        # exactly that.
        inferred = coverage.methods.get("pace", 0) + coverage.methods.get("distance", 0)
        assumed = coverage.methods.get("distance", 0)
        if inferred > coverage.days_logged * 0.5:
            detail = (f" On {assumed} of them the duration was assumed too, leaving "
                      f"distance as the only real measurement.") if assumed else ""
            warnings.append(
                f"{inferred} of {coverage.days_logged} logged days have neither heart "
                f"rate nor a perceived-effort rating, so their load is inferred from "
                f"pace rather than measured.{detail} Adding RPE is the cheapest way to "
                f"improve this.")

    if len(profile.races) > len(dated):
        warnings.append(
            f"{len(profile.races) - len(dated)} race{'s' if len(profile.races) - len(dated) != 1 else ''} "
            f"{'have' if len(profile.races) - len(dated) != 1 else 'has'} no date, so "
            f"{'they cannot' if len(profile.races) - len(dated) != 1 else 'it cannot'} "
            f"be used to calibrate the model.")

    return RealHistory(
        training=coverage,
        races=len(profile.races),
        dated_races=len(dated),
        race_sources=dict(Counter(r.source for r in profile.races)),
        first_race=min((r.race_date for r in dated), default=None),
        last_race=max((r.race_date for r in dated), default=None),
        injuries=len(profile.injuries),
        calibration=calibration,
        warnings=warnings,
    )


def seed_training_days_from_summary(profile: AthleteProfile, weeks: int = 12,
                                    today: date | None = None) -> list[TrainingDay]:
    """Manufacture a plausible training history from the weekly-mileage summary.

    A stopgap for runners with no daily log: it spreads their stated weekly
    volume over their stated run frequency, with one long run. Every day it
    produces is marked `source='estimated'` so it never gets mistaken for
    something that actually happened, and the settings view counts it separately.
    """
    weekly_km = profile.training.current_weekly_km
    if not weekly_km or weekly_km <= 0:
        return []

    today = today or date.today()
    runs_per_week = int(round(profile.training.runs_per_week or 4))
    runs_per_week = max(1, min(7, runs_per_week))
    long_km = profile.training.longest_recent_run_km or weekly_km * 0.3
    long_km = min(long_km, weekly_km * 0.5)

    other_runs = max(runs_per_week - 1, 1)
    easy_km = max((weekly_km - long_km) / other_runs, 0.0)
    # Spread runs through the week rather than bunching them at the start.
    run_offsets = sorted(round(i * 7 / runs_per_week) for i in range(runs_per_week))

    days: list[TrainingDay] = []
    start = today - timedelta(days=weeks * 7 - 1)
    for week in range(weeks):
        week_start = start + timedelta(days=week * 7)
        for offset in range(7):
            day = week_start + timedelta(days=offset)
            if day > today:
                continue
            if offset not in run_offsets:
                days.append(TrainingDay(day=day, kind="off", source="estimated"))
                continue
            is_long = offset == run_offsets[-1]
            km = long_km if is_long else easy_km
            days.append(TrainingDay(
                day=day, distance_km=round(km, 2),
                duration_s=round(km * 6.0 * 60),  # assumed easy pace
                kind="long" if is_long else "easy",
                surface=profile.training.primary_surface,
                source="estimated",
                notes="Estimated from your weekly summary, not a real logged run."))
    return days
