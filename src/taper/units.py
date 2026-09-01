"""Unit helpers. The engine is metric internally; the UI may be either."""
from __future__ import annotations

KM_PER_MILE = 1.609344
M_PER_MILE = 1609.344


def miles_to_km(mi: float) -> float:
    return mi * KM_PER_MILE


def km_to_miles(km: float) -> float:
    return km / KM_PER_MILE


def mps_to_min_per_km(mps: float) -> float:
    """Speed in m/s -> pace in minutes per km."""
    if mps <= 0:
        raise ValueError("speed must be positive")
    return 1000.0 / mps / 60.0


def mps_to_min_per_mile(mps: float) -> float:
    if mps <= 0:
        raise ValueError("speed must be positive")
    return M_PER_MILE / mps / 60.0


def format_duration(seconds: float) -> str:
    """3735.4 -> '1:02:15'. Sub-hour drops the leading zero: '24:15'."""
    if seconds < 0:
        raise ValueError("duration must be non-negative")
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_pace(seconds_per_unit: float) -> str:
    """Pace in seconds per km/mile -> 'M:SS'."""
    total = int(round(seconds_per_unit))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"
