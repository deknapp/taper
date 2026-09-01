"""Finding probable injuries in a training log by looking for the holes.

A runner's log records what they did, never why they stopped. But a long stretch
with no running in an otherwise consistent history is evidence of *something*,
and injury is the most common something. This module proposes those stretches as
candidate injury episodes.

It proposes, it does not conclude. A three-week hole is equally consistent with
a torn calf, a work trip, a newborn, a heatwave, or losing interest in January.
So candidates are surfaced for the runner to confirm, dismiss or annotate, and
only confirmed ones become `InjuryEpisode` records the injury model trains on.
An unconfirmed gap is a question, not a label.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from taper.athlete import InjuryEpisode, Tissue, TrainingDay

# A committed runner takes rest days; they do not take ten of them in a row.
MIN_STOPPAGE_DAYS = 10
# A sustained dip counts if volume falls this far below the preceding baseline.
DIP_FRACTION = 0.5
MIN_DIP_DAYS = 14
BASELINE_WINDOW_DAYS = 90
# Too little history either side and the "gap" is just the edge of the data.
MIN_CONTEXT_DAYS = 21


@dataclass
class Layoff:
    """A stretch where running stopped or dropped sharply."""

    start: date
    end: date
    kind: str                    # 'stoppage' | 'dip'
    days: int
    baseline_weekly_km: float
    during_weekly_km: float
    confidence: str              # 'strong' | 'moderate' | 'weak'
    reason: str
    confirmed: bool = False

    @property
    def weeks(self) -> float:
        return self.days / 7.0

    def to_episode(self, body_part: str, tissue: Tissue = Tissue.OTHER) -> InjuryEpisode:
        """Promote a confirmed layoff into a real injury episode."""
        return InjuryEpisode(
            body_part=body_part, tissue=tissue, onset_date=self.start,
            resolved_date=self.end, days_lost=self.days,
            notes=f"Confirmed from a {self.days}-day {self.kind} in the training log.")


def _running_km_by_day(days: list[TrainingDay]) -> dict[date, float]:
    return {d.day: (d.distance_km if d.kind != "cross" else 0.0) for d in days}


def _weekly_rate(km_by_day: dict[date, float], start: date, end: date) -> float:
    """Mean weekly running volume over a date range."""
    span = (end - start).days + 1
    if span <= 0:
        return 0.0
    total = sum(km for day, km in km_by_day.items() if start <= day <= end)
    return total / span * 7.0


def find_layoffs(days: list[TrainingDay], *,
                 min_stoppage_days: int = MIN_STOPPAGE_DAYS,
                 min_dip_days: int = MIN_DIP_DAYS) -> list[Layoff]:
    """Propose candidate layoffs from a training log.

    Two shapes are looked for. A *stoppage* is running going to zero outright.
    A *dip* is running continuing at a fraction of its former level -- which is
    what a runner managing a niggle actually looks like, and is easy to miss if
    you only search for zeroes.
    """
    if not days:
        return []

    km_by_day = _running_km_by_day(days)
    first, last = min(km_by_day), max(km_by_day)
    if (last - first).days < MIN_CONTEXT_DAYS * 2:
        return []

    layoffs: list[Layoff] = []

    # --- stoppages: runs of consecutive zero-running days ---
    run_start: date | None = None
    day = first
    while day <= last + timedelta(days=1):
        ran = km_by_day.get(day, 0.0) > 0 and day <= last
        if not ran and run_start is None:
            run_start = day
        elif ran and run_start is not None:
            gap_end = day - timedelta(days=1)
            length = (gap_end - run_start).days + 1
            if length >= min_stoppage_days:
                layoffs.append(_build_stoppage(km_by_day, run_start, gap_end, first, last))
            run_start = None
        day += timedelta(days=1)

    # --- dips: sustained partial reduction, week by week ---
    layoffs.extend(_find_dips(km_by_day, first, last, min_dip_days))

    # A stoppage inside a dip is the same event seen twice; keep the stoppage.
    layoffs.sort(key=lambda l: (l.start, l.end))
    merged: list[Layoff] = []
    for layoff in layoffs:
        if merged and layoff.start <= merged[-1].end:
            keep = merged[-1] if merged[-1].kind == "stoppage" else layoff
            merged[-1] = keep
            continue
        merged.append(layoff)
    return merged


def _baseline_before(km_by_day: dict[date, float], when: date, first: date) -> float:
    window_start = max(first, when - timedelta(days=BASELINE_WINDOW_DAYS))
    prior_end = when - timedelta(days=1)
    if (prior_end - window_start).days < MIN_CONTEXT_DAYS:
        return 0.0
    return _weekly_rate(km_by_day, window_start, prior_end)


def _build_stoppage(km_by_day: dict[date, float], start: date, end: date,
                    first: date, last: date) -> Layoff:
    days = (end - start).days + 1
    baseline = _baseline_before(km_by_day, start, first)

    # Confidence comes from how out of character the gap is, and from whether
    # there is enough history around it to judge.
    if baseline <= 0:
        confidence = "weak"
        reason = ("Running stopped, but there is not enough history before it to say "
                  "whether that was unusual for you.")
    elif days >= 28 and baseline >= 30:
        confidence = "strong"
        reason = (f"{days} days without a run, against a baseline of "
                  f"{baseline:.0f} km/week. That is a long stop for someone training "
                  f"that consistently.")
    elif days >= 14:
        confidence = "moderate"
        reason = (f"{days} days without a run, against {baseline:.0f} km/week "
                  f"beforehand.")
    else:
        confidence = "weak"
        reason = (f"{days} days without a run. Short enough to be a planned break "
                  f"or a busy fortnight.")

    return Layoff(start=start, end=end, kind="stoppage", days=days,
                  baseline_weekly_km=baseline, during_weekly_km=0.0,
                  confidence=confidence, reason=reason)


def _find_dips(km_by_day: dict[date, float], first: date, last: date,
               min_dip_days: int) -> list[Layoff]:
    """Weeks where volume held at less than half its recent baseline."""
    dips: list[Layoff] = []
    week_starts: list[date] = []
    cursor = first
    while cursor <= last:
        week_starts.append(cursor)
        cursor += timedelta(days=7)

    in_dip: date | None = None
    for week_start in week_starts:
        week_end = min(week_start + timedelta(days=6), last)
        baseline = _baseline_before(km_by_day, week_start, first)
        volume = sum(km for d, km in km_by_day.items() if week_start <= d <= week_end)

        depressed = baseline > 0 and volume < baseline * DIP_FRACTION and volume > 0
        if depressed and in_dip is None:
            in_dip = week_start
        elif not depressed and in_dip is not None:
            end = week_start - timedelta(days=1)
            days = (end - in_dip).days + 1
            if days >= min_dip_days:
                dips.append(_build_dip(km_by_day, in_dip, end, first))
            in_dip = None

    if in_dip is not None:
        days = (last - in_dip).days + 1
        if days >= min_dip_days:
            dips.append(_build_dip(km_by_day, in_dip, last, first))
    return dips


def _build_dip(km_by_day: dict[date, float], start: date, end: date,
               first: date) -> Layoff:
    days = (end - start).days + 1
    baseline = _baseline_before(km_by_day, start, first)
    during = _weekly_rate(km_by_day, start, end)
    share = (during / baseline) if baseline else 0.0
    return Layoff(
        start=start, end=end, kind="dip", days=days,
        baseline_weekly_km=baseline, during_weekly_km=during,
        confidence="moderate" if days >= 21 else "weak",
        reason=(f"Volume held at {during:.0f} km/week for {days} days, "
                f"{share:.0%} of the {baseline:.0f} km/week you were running before. "
                f"Running through something at reduced volume looks like this."))
