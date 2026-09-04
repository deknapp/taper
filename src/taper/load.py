"""Turning a day of training into a single number the Banister model can eat.

There is no one right training-load metric, and the ones in use disagree with
each other. So rather than pick one and hide the choice, this module implements
three, in descending order of how much they ask of the runner, and the engine
uses the best one each day's data supports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from taper.athlete import Sex, TrainingDay

# --- Banister TRIMP --------------------------------------------------------
# Banister & Calvert (1980), with the exponential intensity weighting from
# Morton, Fitz-Clarke & Banister, J Appl Physiol 69:1171-1177 (1990). The
# weighting constant differs by sex because the original work fitted the blood
# lactate response separately for men and women.

_TRIMP_B = {Sex.MALE: 1.92, Sex.FEMALE: 1.67, Sex.UNSPECIFIED: 1.80}


def heart_rate_reserve(avg_hr: float, rest_hr: float, max_hr: float) -> float:
    """Fraction of heart rate reserve, clamped to [0, 1]."""
    if max_hr <= rest_hr:
        raise ValueError("max HR must exceed resting HR")
    return max(0.0, min(1.0, (avg_hr - rest_hr) / (max_hr - rest_hr)))


def trimp_heart_rate(duration_min: float, avg_hr: float, rest_hr: float,
                     max_hr: float, sex: Sex = Sex.UNSPECIFIED) -> float:
    """Banister's TRIMP: minutes weighted by an exponential intensity term.

    The exponential is what stops a three-hour plod from outscoring a hard
    interval session purely on duration.
    """
    if duration_min <= 0:
        return 0.0
    fraction = heart_rate_reserve(avg_hr, rest_hr, max_hr)
    b = _TRIMP_B.get(sex, 1.80)
    return duration_min * fraction * 0.64 * math.exp(b * fraction)


# --- Session RPE -----------------------------------------------------------
# Foster et al., "A new approach to monitoring exercise training",
# J Strength Cond Res 15:109-115 (2001). Duration times perceived effort.
# Cruder than TRIMP, but it needs no monitor and tracks it surprisingly well.


def session_rpe(duration_min: float, rpe: float) -> float:
    """Foster's sRPE, in arbitrary units, rescaled to sit near TRIMP's range."""
    if duration_min <= 0 or rpe <= 0:
        return 0.0
    return duration_min * max(0.0, min(10.0, rpe)) * 0.1


# --- Pace-based fallback ---------------------------------------------------
# When there is neither a monitor nor an RPE, intensity can be inferred from how
# fast the running was relative to the runner's own threshold velocity. This is
# our own construction, not a published metric -- it is calibrated to land in
# the same range as the two above so the model does not lurch when the data
# source changes mid-history.


def _threshold_velocity_m_per_min(vdot: float) -> float:
    """Velocity at roughly lactate threshold: about 88% of vVO2max in Daniels."""
    # Invert the Daniels-Gilbert cost curve for the velocity at this VDOT, then
    # take the threshold fraction of it.
    from taper.physiology import vo2_cost

    lo, hi = 50.0, 600.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if vo2_cost(mid) < vdot:
            lo = mid
        else:
            hi = mid
    return 0.88 * (lo + hi) / 2.0


def trimp_pace(distance_km: float, duration_min: float, vdot: float,
               gradient_factor: float = 1.0) -> float:
    """Load from distance, time and fitness, with the same shape as TRIMP.

    `gradient_factor` is the Minetti grade-adjustment: a hilly hour costs more
    than a flat one at the same clock pace, and the model should see that.
    """
    if duration_min <= 0 or distance_km <= 0 or vdot <= 0:
        return 0.0
    velocity = (distance_km * 1000.0) / duration_min * gradient_factor
    intensity = max(0.0, min(1.25, velocity / _threshold_velocity_m_per_min(vdot)))
    # Same exponential shape as Banister's TRIMP, on a 0-1.25 intensity scale.
    return duration_min * intensity * 0.64 * math.exp(1.80 * intensity)


# --- Choosing a metric per day ---------------------------------------------

@dataclass
class DayLoad:
    value: float
    method: str  # 'hr' | 'rpe' | 'pace' | 'distance' | 'assumed' | 'rest'


# Typical easy-running pace assumptions, used only to estimate a duration when
# a day records distance but no time.
_ASSUMED_EASY_PACE_MIN_PER_KM = 6.0


def day_load(day: TrainingDay, *, vdot: float | None = None,
             rest_hr: int | None = None, max_hr: int | None = None,
             sex: Sex = Sex.UNSPECIFIED,
             gradient_factor: float = 1.0) -> DayLoad:
    """Best available load for one day, and which metric produced it.

    The method is returned alongside the number so the UI can be honest about
    which days are well-measured and which are estimated from thin data.
    """
    if day.is_rest:
        return DayLoad(0.0, "rest")

    duration_min = (day.duration_s / 60.0) if day.duration_s else None
    if duration_min is None and day.distance_km > 0:
        duration_min = day.distance_km * _ASSUMED_EASY_PACE_MIN_PER_KM

    if duration_min is None or duration_min <= 0:
        return DayLoad(0.0, "rest")

    if day.avg_hr and rest_hr and max_hr and max_hr > rest_hr:
        return DayLoad(
            trimp_heart_rate(duration_min, day.avg_hr, rest_hr, max_hr, sex), "hr")

    if day.rpe:
        return DayLoad(session_rpe(duration_min, day.rpe), "rpe")

    if vdot and day.distance_km > 0 and day.duration_s:
        return DayLoad(
            trimp_pace(day.distance_km, duration_min, vdot, gradient_factor), "pace")

    if vdot and day.distance_km > 0:
        return DayLoad(
            trimp_pace(day.distance_km, duration_min, vdot, gradient_factor), "distance")

    # A real day's work, but nothing to judge its intensity by: no monitor, no
    # effort rating, and no fitness estimate to compare the pace against. Treated
    # as steady easy running, and labelled 'assumed' rather than 'distance' so the
    # provenance report does not report the duration as made up when it is real.
    return DayLoad(session_rpe(duration_min, 4.0), "assumed")
