'''Demo D5: the ndarray trap in the current chunked machinery, and the phantom
change points it produces - fixed as a side effect of pushing window_len into
the core.

ChangePointDetector.get_change_points(series, start, end) only slices
series[start:end] when `series` is a list. For ndarray input (the most natural
input for a numpy-based detector) the detection silently runs over the WHOLE
series and the found indexes are then shifted by `start`. split() feeds it
ndarray... via list() in otava's pipeline, but any direct ndarray caller gets:

  - every window re-detects the whole series and shifts the indexes by `start`
  - change points BEYOND the end of the series (480, 510, 660 on a 450-point
    series below)
  - a ValueError crash in split()'s final stats recomputation
'''
import sys
sys.path.insert(0, '.')

import numpy as np
from functools import partial

from otava.analysis import TTestSignificanceTester, compute_change_points
from otava.change_point_divisive.calculator import PairDistanceCalculator
from otava.change_point_divisive.detector import ChangePointDetector


def legacy_split_ndarray(series, window_len=60, first_pass=0.01):
    """otava.analysis.split() verbatim (ndarray goes straight to the detector)."""
    tester = TTestSignificanceTester(first_pass)
    start, step = 0, int(window_len / 2)
    change_points = []
    while start < len(series):
        end = min(start + window_len, len(series))
        algo = ChangePointDetector(significance_tester=tester, calculator=PairDistanceCalculator)
        new_change_points = algo.get_change_points(series, start, end)   # ndarray!
        last = new_change_points[-1].index if new_change_points else 0
        start = max(last, start + step)
        cpindexes = [cp.index for cp in change_points]
        for cp in new_change_points:
            if cp.index not in cpindexes:
                change_points.append(cp)
                cpindexes.append(cp.index)
    change_points.sort(key=lambda cp: cp.index)
    return [int(cp.index) for cp in change_points]


if __name__ == "__main__":
    print(__doc__)
    rng = np.random.default_rng(5)
    series = np.concatenate([
        np.full(150, 20.0), np.full(30, 22.0), np.full(150, 20.0), np.full(120, 19.0),
    ]) + rng.normal(0, 0.4, 450)
    print(f"series length: {len(series)}, true change points: [150, 180, 330]\n")

    print("legacy split() with ndarray input:")
    try:
        print("  weak cps:", legacy_split_ndarray(series), "  <- 480/510/660 are beyond the series end")
    except ValueError as e:
        print("  CRASHED with ValueError (empty side in the final stats recomputation)")

    print("\nlegacy compute_change_points(series as ndarray, core_window=False):")
    try:
        cps, _ = compute_change_points(series, 60, 0.001, 0.0, core_window=False)
        print("  ", [int(c.index) for c in cps])
    except ValueError:
        print("   CRASHED with ValueError")

    print("\nwindowed core (core_window=True, same ndarray input):")
    cps, weak = compute_change_points(series, 60, 0.001, 0.0, core_window=True)
    print("   final:", [int(c.index) for c in cps])
