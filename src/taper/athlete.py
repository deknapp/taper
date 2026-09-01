"""The athlete profile: everything the simulation needs to know about a runner.

This is the schema the intake form fills. Fields are optional wherever the sim
can fall back to a population default, and each such field notes what it buys us
in fidelity, so the form can tell the player why it is worth answering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class Sex(str, Enum):
    """Used only for physiological modelling: VO2max trajectory with age, bone
    mineral density baselines, and RED-S / bone-stress risk terms."""

    FEMALE = "female"
    MALE = "male"
    UNSPECIFIED = "unspecified"


class Surface(str, Enum):
    ROAD = "road"
    TRAIL = "trail"
    TRACK = "track"
    TREADMILL = "treadmill"
    GRASS = "grass"


class Tissue(str, Enum):
    """Injuries are grouped by tissue because the tissues adapt and heal on very
    different timescales: tendon and bone remodel over months, muscle over days.
    The sim tracks load per tissue rather than one global 'fatigue' number."""

    BONE = "bone"
    TENDON = "tendon"
    MUSCLE = "muscle"
    JOINT = "joint"
    FASCIA = "fascia"
    OTHER = "other"


class Occupation(str, Enum):
    """Non-training load. A nurse on their feet 12 hours recovers differently
    from a desk worker doing the identical training week."""

    SEDENTARY = "sedentary"
    LIGHT = "light"
    ON_FEET = "on_feet"
    PHYSICAL = "physical"


@dataclass
class RaceResult:
    """One race. The spine of the profile: a series of these over years gives a
    fitness trajectory, an implied training age, and gaps that hint at time lost."""

    distance_m: float
    finish_time_s: float
    race_date: date | None = None
    name: str = ""
    place_overall: int | None = None
    field_size: int | None = None
    surface: Surface = Surface.ROAD
    elevation_gain_m: float | None = None
    source: str = "manual"  # 'manual', 'paste', or the host it was fetched from

    @property
    def speed_mps(self) -> float:
        return self.distance_m / self.finish_time_s

    @property
    def pace_s_per_km(self) -> float:
        return self.finish_time_s / (self.distance_m / 1000.0)


@dataclass
class Injury:
    """Prior injury is the strongest single predictor of future injury in the
    literature, so history here materially changes the hazard model."""

    tissue: Tissue
    body_part: str
    start_date: date | None = None
    weeks_out: float | None = None
    recurrences: int = 0
    notes: str = ""


@dataclass
class TrainingBackground:
    """Recent and lifetime training load. `current_weekly_km` is the single most
    important number in the form: it seeds chronic load, which sets how much the
    runner can absorb in week one without breaking."""

    current_weekly_km: float | None = None       # mean of the last 4 weeks
    peak_weekly_km_ever: float | None = None     # ceiling the body has seen
    longest_recent_run_km: float | None = None
    runs_per_week: float | None = None
    years_running: float | None = None           # training age
    strength_days_per_week: float = 0.0          # protective in the literature
    cross_training_hours_per_week: float = 0.0
    primary_surface: Surface = Surface.ROAD
    hilliness_m_per_km: float | None = None      # typical climb on home routes


@dataclass
class Physiology:
    """Optional measurements. Absent, the sim estimates these from race results
    and age; present, they tighten the zone model and the HR-based load metric."""

    resting_hr: int | None = None
    max_hr: int | None = None
    vo2max_tested: float | None = None
    height_cm: float | None = None
    body_mass_kg: float | None = None


@dataclass
class LifeLoad:
    """Stress and sleep are recovery multipliers, not decoration: both have real
    associations with injury and with blunted training response."""

    sleep_hours: float | None = None
    occupation: Occupation = Occupation.SEDENTARY
    life_stress: int = 3  # 1 = calm, 5 = maximal


@dataclass
class GoalRace:
    distance_m: float
    race_date: date
    name: str = ""
    target_time_s: float | None = None
    elevation_gain_m: float | None = None
    surface: Surface = Surface.ROAD


@dataclass
class AthleteProfile:
    name: str = ""
    birth_date: date | None = None
    sex: Sex = Sex.UNSPECIFIED
    physiology: Physiology = field(default_factory=Physiology)
    training: TrainingBackground = field(default_factory=TrainingBackground)
    life: LifeLoad = field(default_factory=LifeLoad)
    races: list[RaceResult] = field(default_factory=list)
    injuries: list[Injury] = field(default_factory=list)
    goal: GoalRace | None = None

    def age_on(self, when: date) -> float | None:
        if self.birth_date is None:
            return None
        return (when - self.birth_date).days / 365.2425

    def best_races(self) -> list[RaceResult]:
        """Fastest result at each distance, oldest-first, for trajectory work."""
        best: dict[int, RaceResult] = {}
        for r in self.races:
            key = int(round(r.distance_m))
            if key not in best or r.finish_time_s < best[key].finish_time_s:
                best[key] = r
        return sorted(best.values(), key=lambda r: r.distance_m)
