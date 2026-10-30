'''Demo D3: randomized stress sweep - legacy chunked path vs windowed core
vs no-window permutation reference, across change-point layouts and magnitudes.

Configs (N=250, window_len=60, max_pvalue=0.001, 30 seeds each):
  - single step at 100, delta/sigma in {1.0, 1.5, 2.0, 3.0}
  - twin steps at 100 and 100+sep, sep in {20, 40, 80}, delta = 2 sigma
  - triple steps at 70/140/210, delta = 2 sigma
'''
import sys, copy, time
sys.path.insert(0, '.')

import numpy as np
from functools import partial

from otava.analysis import TTestSignificanceTester, merge, compute_change_points, compute_change_points_orig
from otava.change_point_divisive.calculator import PairDistanceCalculator
from otava.change_point_divisive.detector import ChangePointDetector
from otava.change_point_divisive.significance_test import PermutationsSignificanceTester

WINDOW_LEN = 60
MAX_PVALUE = 0.001
N, SIGMA = 250, 1.0
TOL = 5
SEEDS = 30


def make_series(seed, cps, deltas):
    rng = np.random.default_rng(seed)
    s = np.full(N, 20.0)
    level = 20.0
    events = sorted(zip(cps, deltas))
    last = 0
    for c, d in events:
        s[last:c] = level
        level += d * SIGMA
        last = c
    s[last:] = level
    return s + rng.normal(0, SIGMA, N)


def legacy_answer(series):
    first_pass = MAX_PVALUE * 10
    tester = TTestSignificanceTester(first_pass)
    start, step = 0, WINDOW_LEN // 2
    change_points = []
    while start < len(series):
        end = min(start + WINDOW_LEN, len(series))
        algo = ChangePointDetector(significance_tester=tester, calculator=PairDistanceCalculator)
        new_cps = algo.get_change_points(list(series), start, end)
        last = new_cps[-1].index if new_cps else 0
        start = max(last, start + step)
        idxs = [cp.index for cp in change_points]
        for cp in new_cps:
            if cp.index not in idxs:
                change_points.append(cp)
                idxs.append(cp.index)
    merged = merge(copy.copy(change_points), list(series), MAX_PVALUE, 0.0)
    return [int(cp.index) for cp in merged]


def core_answer(series):
    cps, _ = compute_change_points(series, window_len=WINDOW_LEN, max_pvalue=MAX_PVALUE,
                                   min_magnitude=0.0, core_window=True)
    return [int(cp.index) for cp in cps]


def core_perm_answer(series, seed):
    # windowed calculator composed with the permutation significance test:
    # same core windowing, but the null distribution of qhats is calibrated
    # per-dataset by permuting (statistically consistent, no relaxed threshold)
    calc = partial(PairDistanceCalculator, window_len=WINDOW_LEN)
    tester = PermutationsSignificanceTester(max_pvalue=MAX_PVALUE, permutations=100,
                                            calculator=calc, seed=seed)
    detector = ChangePointDetector(significance_tester=tester, calculator=calc)
    return [int(cp.index) for cp in detector.get_change_points(series=series)]


def orig_answer(series, seed):
    try:
        cps, _ = compute_change_points_orig(list(series), max_pvalue=MAX_PVALUE, seed=seed)
        return [int(cp.index) for cp in cps]
    except Exception as e:
        return [f"FAILED:{type(e).__name__}"]


def score(detected, truths):
    detected = [d for d in detected if isinstance(d, int)]
    found, errs, spurious = 0, [], 0
    for t in truths:
        d = [abs(x - t) for x in detected]
        if d and min(d) <= TOL:
            found += 1
            errs.append(min(d))
    for x in detected:
        if all(abs(x - t) > TOL for t in truths):
            spurious += 1
    return found, errs, spurious


def run_config(name, cps, deltas):
    res = {"legacy": [0, [], 0], "core": [0, [], 0], "core+perm": [0, [], 0], "orig": [0, [], 0]}
    times = {k: 0.0 for k in res}
    for seed in range(SEEDS):
        s = make_series(seed, cps, deltas)
        for key, fn in (("legacy", lambda: legacy_answer(s)),
                        ("core", lambda: core_answer(s)),
                        ("core+perm", lambda: core_perm_answer(s, seed)),
                        ("orig", lambda: orig_answer(s, seed))):
            t0 = time.perf_counter(); ans = fn(); times[key] += time.perf_counter() - t0
            f, errs, sp = score(ans, cps)
            res[key][0] += f; res[key][1] += errs; res[key][2] += sp
    n_true = len(cps)
    print(f"\n{name}  (truth: {cps}, {SEEDS} seeds, {n_true * SEEDS} change points total)")
    for key, label in (("legacy", "legacy split+merge"), ("core", "windowed core(ttest)"),
                       ("core+perm", "windowed core(perm)"), ("orig", "no-window perm ref")):
        found, errs, sp = res[key]
        med = int(np.median(errs)) if errs else -1
        print(f"  {label}: found {found:3d}/{n_true * SEEDS}  med|err| {med}  "
              f"spurious {sp:3d}  ({times[key]:6.1f} s total)")


if __name__ == "__main__":
    print(__doc__)
    run_config("single, delta=1.0 sigma", [100], [1.0])
    run_config("single, delta=1.5 sigma", [100], [1.5])
    run_config("single, delta=2.0 sigma", [100], [2.0])
    run_config("single, delta=3.0 sigma", [100], [3.0])
    run_config("twins, 20 apart (< window_len)", [100, 120], [2.0, -2.0])
    run_config("twins, 40 apart (< window_len)", [100, 140], [2.0, -2.0])
    run_config("twins, 80 apart (> window_len)", [100, 180], [2.0, -2.0])
    run_config("triple, 70/140/210", [70, 140, 210], [2.0, -1.5, 2.0])
