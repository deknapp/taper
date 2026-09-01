"""The Banister fitness-fatigue model: the simulation's core.

Banister et al., "A systems model of the effects of training on physical
performance", IEEE Trans Syst Man Cybern SMC-5:94-102 (1975); refined in Morton,
Fitz-Clarke & Banister, J Appl Physiol 69:1171-1177 (1990).

One training impulse raises two things at once, and they fade at different
rates: fitness slowly, fatigue quickly. Performance is what is left after the
fatigue is subtracted from the fitness. That single asymmetry is where the whole
game comes from -- it is why hard training makes you slower before it makes you
faster, and why backing off before a race makes you faster without making you
fitter.

Known limitation, stated up front: the two-component model has no notion of
intensity as distinct from load, so it concludes that *complete* rest tapers
better than a reduced-volume one. Real tapers cut volume by roughly half while
holding intensity, and runners who stop entirely race worse. The model cannot
express that, because one impulse number cannot carry both how much and how
hard. Until the sim adds a sharpness term, treat its taper recommendations as
directionally right and quantitatively too generous about rest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta

# Defaults from the classic literature. Individual runners vary widely, which is
# what `fit_from_history` is for -- these are only the starting prior.
DEFAULT_TAU_FITNESS = 42.0   # days
DEFAULT_TAU_FATIGUE = 7.0    # days
DEFAULT_K_FITNESS = 1.0
DEFAULT_K_FATIGUE = 2.0      # > k_fitness, or tapering would do nothing


@dataclass(frozen=True)
class BanisterParams:
    """Per-runner model constants.

    `baseline` is performance with no training at all, in whatever units the
    performances are measured in. We use VDOT throughout, so the fitted model
    speaks directly in race-predicting units.
    """

    tau_fitness: float = DEFAULT_TAU_FITNESS
    tau_fatigue: float = DEFAULT_TAU_FATIGUE
    k_fitness: float = DEFAULT_K_FITNESS
    k_fatigue: float = DEFAULT_K_FATIGUE
    baseline: float = 0.0

    def __post_init__(self) -> None:
        if self.tau_fitness <= 0 or self.tau_fatigue <= 0:
            raise ValueError("time constants must be positive")
        if self.tau_fatigue >= self.tau_fitness:
            raise ValueError(
                "fatigue must decay faster than fitness, or the model has no taper in it")

    @property
    def days_to_peak(self) -> float:
        """Days after a single hard session at which performance peaks.

        Closed form: differentiate k1*e^(-t/T1) - k2*e^(-t/T2) and solve for
        zero. This is the taper, falling out of the arithmetic rather than being
        put in by hand.
        """
        ratio = (self.k_fatigue * self.tau_fitness) / (self.k_fitness * self.tau_fatigue)
        if ratio <= 1:
            return 0.0
        return math.log(ratio) / (1.0 / self.tau_fatigue - 1.0 / self.tau_fitness)


@dataclass(frozen=True)
class DayState:
    day: date
    load: float
    fitness: float
    fatigue: float

    def performance(self, params: BanisterParams) -> float:
        return params.baseline + params.k_fitness * self.fitness - params.k_fatigue * self.fatigue

    def form(self, params: BanisterParams) -> float:
        """Fitness minus fatigue, weighted -- 'freshness'. Negative means buried."""
        return params.k_fitness * self.fitness - params.k_fatigue * self.fatigue


def simulate(loads: dict[date, float], params: BanisterParams | None = None, *,
             start: date | None = None, end: date | None = None,
             initial_fitness: float = 0.0, initial_fatigue: float = 0.0) -> list[DayState]:
    """Run the model day by day over a date range.

    Missing days are rest, and rest is not nothing -- it is where fatigue decays
    faster than fitness, so the gaps are load-bearing. The range is filled in
    densely rather than iterating only the days that have training on them.
    """
    params = params or BanisterParams()
    if not loads and (start is None or end is None):
        return []

    start = start or min(loads)
    end = end or max(loads)
    if end < start:
        raise ValueError("end must not precede start")

    decay_fitness = math.exp(-1.0 / params.tau_fitness)
    decay_fatigue = math.exp(-1.0 / params.tau_fatigue)

    fitness, fatigue = initial_fitness, initial_fatigue
    states: list[DayState] = []
    day = start
    while day <= end:
        load = loads.get(day, 0.0)
        fitness = fitness * decay_fitness + load
        fatigue = fatigue * decay_fatigue + load
        states.append(DayState(day=day, load=load, fitness=fitness, fatigue=fatigue))
        day += timedelta(days=1)
    return states


# --- fitting ---------------------------------------------------------------

def _solve_symmetric_3x3(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    x = [0.0] * n
    for r in reversed(range(n)):
        total = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = total / m[r][r]
    return x


@dataclass
class FitResult:
    params: BanisterParams
    rmse: float
    n_observations: int
    note: str


def fit_from_history(loads: dict[date, float],
                     performances: dict[date, float]) -> FitResult | None:
    """Fit a runner's own parameters to their real race history.

    Given the two time constants, the model is *linear* in the three remaining
    parameters, so we grid-search the taus and solve exactly for baseline,
    k_fitness and k_fatigue at each candidate. That avoids a nonlinear optimiser
    and a scipy dependency, and the grid is small enough to be instant.

    Fitting three parameters to a handful of races overfits happily, so the
    result carries a note about how far it should be trusted.
    """
    usable = {d: p for d, p in performances.items() if d in loads or d >= min(loads, default=d)}
    if len(usable) < 3 or not loads:
        return None

    start, end = min(loads), max(max(loads), max(usable))
    best: tuple[float, BanisterParams] | None = None

    for tau_fitness in (25, 30, 35, 40, 42, 45, 50, 55, 60):
        for tau_fatigue in (5, 7, 9, 11, 13, 15):
            if tau_fatigue >= tau_fitness:
                continue
            states = {s.day: s for s in simulate(
                loads, BanisterParams(tau_fitness=tau_fitness, tau_fatigue=tau_fatigue),
                start=start, end=end)}

            rows = [(states[d].fitness, states[d].fatigue, perf)
                    for d, perf in usable.items() if d in states]
            if len(rows) < 3:
                continue

            # Normal equations for perf = c0 + c1*fitness + c2*fatigue.
            design = [[1.0, g, h] for g, h, _ in rows]
            targets = [p for _, _, p in rows]
            ata = [[sum(r[i] * r[j] for r in design) for j in range(3)] for i in range(3)]
            atb = [sum(design[k][i] * targets[k] for k in range(len(rows))) for i in range(3)]

            coeffs = _solve_symmetric_3x3(ata, atb)
            if coeffs is None:
                continue
            baseline, k_fitness, k_fatigue = coeffs[0], coeffs[1], -coeffs[2]

            # Reject fits that invert the model's meaning: training must help,
            # fatigue must hurt, and fatigue must outweigh fitness per unit or
            # there is no taper.
            if k_fitness <= 0 or k_fatigue <= k_fitness:
                continue

            residuals = [(baseline + k_fitness * g - k_fatigue * h) - p for g, h, p in rows]
            rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))

            if best is None or rmse < best[0]:
                best = (rmse, BanisterParams(
                    tau_fitness=float(tau_fitness), tau_fatigue=float(tau_fatigue),
                    k_fitness=k_fitness, k_fatigue=k_fatigue, baseline=baseline))

    if best is None:
        return None

    rmse, params = best
    n = len(usable)
    if n < 6:
        note = (f"Fitted to {n} races. That is too few to pin down five parameters -- "
                f"treat this as a nudge away from the population defaults, not as "
                f"your true physiology.")
    elif rmse > 1.5:
        note = (f"Fitted to {n} races, but the residual is {rmse:.2f} VDOT. Your results "
                f"are noisier than the model can explain, which usually means missing "
                f"training history rather than a bad model.")
    else:
        note = f"Fitted to {n} races, residual {rmse:.2f} VDOT."
    return FitResult(params=params, rmse=rmse, n_observations=n, note=note)


def default_params_for(baseline_vdot: float) -> BanisterParams:
    """Population-default parameters anchored so an untrained runner sits at
    `baseline_vdot`. Used until there is enough history to fit."""
    return BanisterParams(baseline=baseline_vdot)


# --- planning helpers ------------------------------------------------------

def project(states: list[DayState], params: BanisterParams,
            future_loads: dict[date, float]) -> list[DayState]:
    """Carry a simulated history forward through a planned block."""
    if not future_loads:
        return []
    last = states[-1] if states else None
    return simulate(
        future_loads, params,
        start=min(future_loads), end=max(future_loads),
        initial_fitness=(last.fitness if last else 0.0),
        initial_fatigue=(last.fatigue if last else 0.0),
    )


def best_race_day(states: list[DayState], params: BanisterParams) -> DayState | None:
    """The day in this window on which the runner is predicted to be fastest."""
    return max(states, key=lambda s: s.performance(params)) if states else None


def taper_curve(params: BanisterParams, days: int = 42,
                impulse: float = 100.0) -> list[tuple[int, float]]:
    """Performance response to one hard session, day by day after it.

    Dips first, then rises above where it started, then decays. Plotting this is
    the clearest way to show a player what tapering actually is.
    """
    decay_fitness = math.exp(-1.0 / params.tau_fitness)
    decay_fatigue = math.exp(-1.0 / params.tau_fatigue)
    fitness = fatigue = impulse
    out: list[tuple[int, float]] = []
    for n in range(days + 1):
        out.append((n, params.k_fitness * fitness - params.k_fatigue * fatigue))
        fitness *= decay_fitness
        fatigue *= decay_fatigue
    return out
