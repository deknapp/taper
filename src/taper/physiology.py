"""Published running-physiology models.

Everything here is somebody else's equation, cited at the point of use. Game
tuning lives elsewhere; this module stays honest so that when the sim disagrees
with reality we know which half to blame.
"""
from __future__ import annotations

import math

# --- Daniels & Gilbert VDOT ------------------------------------------------
# Jack Daniels & Jimmy Gilbert, "Oxygen Power" (1979); as presented in Daniels'
# Running Formula. Two empirical curves: the oxygen cost of a given velocity,
# and the fraction of VO2max sustainable for a given duration.


def vo2_cost(velocity_m_per_min: float) -> float:
    """Oxygen cost (ml/kg/min) of running at a velocity, per Daniels-Gilbert."""
    v = velocity_m_per_min
    return -4.60 + 0.182258 * v + 0.000104 * v * v


def fractional_utilization(duration_min: float) -> float:
    """Fraction of VO2max sustainable for `duration_min`, per Daniels-Gilbert."""
    t = duration_min
    return 0.8 + 0.1894393 * math.exp(-0.012778 * t) + 0.2989558 * math.exp(-0.1932605 * t)


def vdot_from_race(distance_m: float, time_s: float) -> float:
    """VDOT implied by a race result: a fitness index in VO2max units.

    It is not a measured VO2max. It is the VO2max that *would* explain this
    performance under Daniels' assumptions about economy, which is exactly what
    we want -- a single number tracking race fitness across distances.
    """
    if distance_m <= 0 or time_s <= 0:
        raise ValueError("distance and time must be positive")
    duration_min = time_s / 60.0
    velocity = distance_m / duration_min
    return vo2_cost(velocity) / fractional_utilization(duration_min)


def predict_time(distance_m: float, vdot: float) -> float:
    """Invert VDOT: the time this fitness predicts at a distance, in seconds.

    No closed form, so bisect on duration. Monotone in time, so this is safe.
    """
    if distance_m <= 0 or vdot <= 0:
        raise ValueError("distance and VDOT must be positive")

    def implied_vdot(time_s: float) -> float:
        return vdot_from_race(distance_m, time_s)

    lo, hi = 1.0, 60.0 * 60.0 * 30.0  # 1 second .. 30 hours
    for _ in range(200):
        mid = (lo + hi) / 2.0
        # Slower time -> lower implied VDOT, so the function is decreasing.
        if implied_vdot(mid) > vdot:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- Riegel endurance model ------------------------------------------------
# Pete Riegel, "Athletic Records and Human Endurance" (American Scientist, 1981).
# Simpler and less accurate than VDOT across wide distance jumps, but it is the
# model most runners already reason with, so we keep it for cross-checking.

RIEGEL_EXPONENT = 1.06


def riegel_predict(known_distance_m: float, known_time_s: float,
                   target_distance_m: float, exponent: float = RIEGEL_EXPONENT) -> float:
    """T2 = T1 * (D2/D1)^k. Degrades badly beyond roughly a 3x extrapolation."""
    if min(known_distance_m, known_time_s, target_distance_m) <= 0:
        raise ValueError("inputs must be positive")
    return known_time_s * (target_distance_m / known_distance_m) ** exponent


# --- Minetti gradient cost -------------------------------------------------
# Minetti et al., "Energy cost of walking and running at extreme uphill and
# downhill slopes", J Appl Physiol 93:1039-1046 (2002). Polynomial fit for the
# energy cost of running, J/kg/m, over gradients of -0.45 to +0.45.

_MINETTI_VALID = (-0.45, 0.45)
COST_FLAT_J_PER_KG_M = 3.6


def gradient_cost(gradient: float) -> float:
    """Energy cost of running (J/kg/m) at a gradient (rise/run, so 0.05 = 5%).

    Note the curve has a minimum near -0.10: gently downhill running is
    genuinely cheaper than flat. It is also where eccentric muscle damage comes
    from, which the injury model handles separately -- cheap is not free.
    """
    lo, hi = _MINETTI_VALID
    i = max(lo, min(hi, gradient))
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


def grade_adjusted_factor(gradient: float) -> float:
    """Cost at this gradient relative to flat. 1.0 on the flat, ~2.6 at +20%."""
    return gradient_cost(gradient) / COST_FLAT_J_PER_KG_M


# --- Odds and ends ---------------------------------------------------------


def max_hr_estimate(age_years: float) -> float:
    """Tanaka, Monahan & Seals, JACC 37:153-156 (2001). 208 - 0.7*age.

    Meaningfully better than the folk '220 - age', though the population
    standard deviation is still around 7 bpm -- treat it as a placeholder until
    the runner supplies a measured value.
    """
    return 208.0 - 0.7 * age_years
