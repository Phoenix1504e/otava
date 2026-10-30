'''Demo D4: wall-clock cost of the detection paths at production scale.

Series: N points, change points every ~N/4, gaussian noise, window_len=60.
The permutation reference is skipped at N=8000 (its O(N^2) per-permutation
cost makes it impractical there - that is exactly the gap windowing closes).
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


def make_series(n, seed=0):
    rng = np.random.default_rng(seed)
    s = np.full(n, 20.0)
    for k in range(1, 4):
        s[n * k // 4:] += 0.8 if k % 2 else -0.8
    return s + rng.normal(0, 0.5, n)


def timeit(fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t0) / reps, out


def legacy(series):
    first_pass = 0.01
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
    merged = merge(copy.copy(change_points), list(series), 0.001, 0.0)
    return merged


def core(series):
    cps, _ = compute_change_points(series, window_len=WINDOW_LEN, max_pvalue=0.001,
                                   min_magnitude=0.0, core_window=True)
    return cps


def core_perm(series, seed=0):
    calc = partial(PairDistanceCalculator, window_len=WINDOW_LEN)
    tester = PermutationsSignificanceTester(max_pvalue=0.001, permutations=100, calculator=calc, seed=seed)
    detector = ChangePointDetector(significance_tester=tester, calculator=calc)
    return detector.get_change_points(series=series)


def orig(series, seed=0):
    cps, _ = compute_change_points_orig(list(series), max_pvalue=0.001, seed=seed)
    return cps


if __name__ == "__main__":
    print(__doc__)
    print(f"{'N':>6} | {'legacy split+merge':>19} | {'windowed core(ttest)':>21} | "
          f"{'windowed core(perm)':>20} | {'no-window perm ref':>19}")
    print("-" * 100)
    for n in (500, 1000, 2000, 4000, 8000):
        s = make_series(n)
        t1, a = timeit(lambda: legacy(s))
        t2, b = timeit(lambda: core(s))
        row = (f"{n:>6} | {t1*1000:>16.0f} ms | {t2*1000:>18.0f} ms |")
        if n <= 2000:
            t3, c = timeit(lambda: core_perm(s))
            t4, d = timeit(lambda: orig(s))
            row += f" {t3*1000:>17.0f} ms | {t4*1000:>16.0f} ms |"
        else:
            # permutation paths cost O(N^2) per permutation (both variants:
            # the distance/cumsum setup dominates) - only run them at small N
            row += f" {'-'*19} | {'-'*19} |"
        print(row)
    n = 8000
    s = make_series(n)
    print(f"\nsanity at N=8000: legacy found {len(legacy(s))} cps, windowed core found {len(core(s))} cps "
          f"(4 true cps)")
