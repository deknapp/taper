from datetime import date, timedelta

import pytest

from taper.banister import (
    BanisterParams, best_race_day, fit_from_history, project, simulate, taper_curve,
)

START = date(2026, 1, 1)


def days(n: int) -> list[date]:
    return [START + timedelta(days=i) for i in range(n)]


def constant_block(n: int, load: float, start: date = START) -> dict[date, float]:
    return {start + timedelta(days=i): load for i in range(n)}


def test_fatigue_must_decay_faster_than_fitness():
    with pytest.raises(ValueError, match="no taper"):
        BanisterParams(tau_fitness=10, tau_fatigue=20)


def test_single_impulse_dips_then_peaks_then_fades():
    """The shape the whole game rests on: one hard session makes you slower
    first, faster later, then the gain fades."""
    curve = dict(taper_curve(BanisterParams(), days=60))
    assert curve[0] < 0, "immediately after a hard session you are net worse off"
    peak_day = max(curve, key=lambda d: curve[d])
    assert curve[peak_day] > 0, "the gain eventually outlasts the fatigue"
    assert 10 < peak_day < 35
    assert curve[60] < curve[peak_day], "and then it decays away"


def test_days_to_peak_matches_the_numeric_argmax():
    """The closed form must agree with actually running the model forward."""
    for params in (BanisterParams(),
                   BanisterParams(tau_fitness=50, tau_fatigue=9, k_fatigue=2.5),
                   BanisterParams(tau_fitness=30, tau_fatigue=5, k_fatigue=1.8)):
        curve = taper_curve(params, days=90)
        numeric_peak = max(curve, key=lambda pair: pair[1])[0]
        assert abs(params.days_to_peak - numeric_peak) <= 1.0


def test_steady_training_approaches_an_asymptote():
    """Constant load cannot raise fitness forever; it converges on load*tau."""
    params = BanisterParams()
    states = simulate(constant_block(400, 10.0), params)
    final = states[-1].fitness
    # Geometric series limit: w / (1 - e^(-1/tau)).
    import math
    expected = 10.0 / (1 - math.exp(-1 / params.tau_fitness))
    assert final == pytest.approx(expected, rel=0.01)


def test_rest_days_are_simulated_not_skipped():
    """A gap in the training log is rest, and rest is where the model does its
    most important work -- it must not be treated as a missing day."""
    sparse = {START: 100.0, START + timedelta(days=10): 100.0}
    states = simulate(sparse, BanisterParams())
    assert len(states) == 11
    assert [s.load for s in states[1:10]] == [0.0] * 9
    assert states[9].fatigue < states[0].fatigue


def test_tapering_beats_training_through():
    """The thesis: two runners do the same hard block, then one keeps hammering
    and the other backs off. The one who backs off races faster."""
    params = BanisterParams()
    block = constant_block(70, 12.0)
    base = simulate(block, params)

    race_day = max(block) + timedelta(days=14)
    keep_going = {max(block) + timedelta(days=i + 1): 12.0 for i in range(14)}
    taper = {max(block) + timedelta(days=i + 1): 3.0 for i in range(14)}

    through = project(base, params, keep_going)[-1].performance(params)
    tapered = project(base, params, taper)[-1].performance(params)
    assert tapered > through
    assert keep_going.keys() == taper.keys() == {race_day - timedelta(days=13 - i)
                                                 for i in range(14)}


def test_detraining_loses_fitness_slower_than_fatigue():
    params = BanisterParams()
    states = simulate(constant_block(60, 12.0), params)
    end = states[-1]
    rest = project(states, params, {max(constant_block(60, 12.0)) + timedelta(days=i + 1): 0.0
                                    for i in range(21)})
    assert rest[-1].fatigue / end.fatigue < rest[-1].fitness / end.fitness


def test_best_race_day_finds_the_peak():
    params = BanisterParams()
    base = simulate(constant_block(60, 12.0), params)
    window = {max(constant_block(60, 12.0)) + timedelta(days=i + 1): 2.0 for i in range(40)}
    projected = project(base, params, window)
    peak = best_race_day(projected, params)
    assert peak is not None
    # Peaks somewhere in the middle of the window, not on the first or last day.
    assert projected[0].day < peak.day < projected[-1].day


def test_fit_recovers_known_parameters_from_synthetic_history():
    """Generate history from known parameters, then check the fitter finds them."""
    truth = BanisterParams(tau_fitness=45.0, tau_fatigue=9.0,
                           k_fitness=0.9, k_fatigue=2.1, baseline=38.0)
    loads = {START + timedelta(days=i): (14.0 if i % 7 else 0.0) for i in range(400)}
    states = {s.day: s for s in simulate(loads, truth)}

    race_days = [START + timedelta(days=d) for d in (120, 165, 210, 255, 300, 345, 390)]
    performances = {d: states[d].performance(truth) for d in race_days}

    result = fit_from_history(loads, performances)
    assert result is not None
    assert result.rmse < 0.3
    # The recovered model must predict the same performances, which matters more
    # than recovering the exact constants -- the parameters trade off.
    for d in race_days:
        refit = {s.day: s for s in simulate(loads, result.params)}
        assert refit[d].performance(result.params) == pytest.approx(performances[d], abs=0.5)


def test_fit_declines_when_there_is_too_little_to_go_on():
    loads = constant_block(60, 10.0)
    assert fit_from_history(loads, {START + timedelta(days=30): 50.0}) is None
    assert fit_from_history({}, {}) is None


def test_fit_warns_when_the_sample_is_thin():
    truth = BanisterParams(baseline=40.0)
    loads = {START + timedelta(days=i): (12.0 if i % 7 else 0.0) for i in range(200)}
    states = {s.day: s for s in simulate(loads, truth)}
    races = [START + timedelta(days=d) for d in (100, 140, 180)]
    result = fit_from_history(loads, {d: states[d].performance(truth) for d in races})
    assert result is not None
    assert "too few" in result.note


def test_the_model_overvalues_complete_rest():
    """A guard on a known limitation rather than a feature.

    Pure Banister has no intensity term, so it prefers total rest to a
    reduced-volume taper -- which is wrong about real runners. The behaviour is
    pinned here so that if a later sharpness term changes it, this test fails
    loudly and someone updates the docstring that admits to it.
    """
    params = BanisterParams()
    block = constant_block(70, 12.0)
    base = simulate(block, params)
    last = max(block)

    def after(load: float) -> float:
        future = {last + timedelta(days=i + 1): load for i in range(14)}
        return project(base, params, future)[-1].performance(params)

    assert after(0.0) > after(3.0) > after(12.0)
