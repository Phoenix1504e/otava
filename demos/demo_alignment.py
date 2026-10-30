'''Demo D2: what p-value does a change point get? It depends on where the
window grid happens to fall (legacy chunked path) - vs a stable answer with
the window pushed into the core (issue #84 prototype).

Setup: one clean step change (+2 sigma) at index c=177 in a 250-point series,
window_len=60 (chunked path scans windows of 60 with step 30).
The first chunked window that contains the change point sees it at local
position 57, i.e. with only 3 points of right context. The window-local
t-test p-value that split()'s index-dedup keeps (and that merge() judges) is
a function of that accidental geometry - while the same change point scored
with full context has p ~ 1e-15.

We repeat the SAME signal with 0 / 15 / 30 flat noise points prepended
(drawn from a separate fixed generator, so the data around the change point
is bit-identical in all three cases; only the window grid alignment shifts).
'''
import sys, copy
sys.path.insert(0, '.')

import numpy as np
from scipy.stats import ttest_ind_from_stats

from otava.analysis import TTestSignificanceTester, merge, compute_change_points, compute_change_points_orig
from otava.change_point_divisive.calculator import PairDistanceCalculator
from otava.change_point_divisive.detector import ChangePointDetector

WINDOW_LEN = 60
MAX_PVALUE = 0.001
N, C, DELTA, SIGMA = 250, 177, 2.0, 1.0


def make_series(seed, prepend=0):
    signal_rng = np.random.default_rng(seed)          # fixes the data around the CP
    prefix_rng = np.random.default_rng(999)           # fixes the (flat) history prefix
    body = np.full(N, 20.0)
    body[C:] = 20.0 + DELTA * SIGMA
    body += signal_rng.normal(0, SIGMA, N)
    prefix = 20.0 + prefix_rng.normal(0, SIGMA, prepend)
    return np.concatenate([prefix, body])


def instrumented_split(series):
    '''Faithful copy of otava.analysis.split() that records every per-window
    detection together with the window-local p-value the index-dedup keeps.'''
    tester = TTestSignificanceTester(MAX_PVALUE * 10)      # first-pass threshold 0.01
    start, step = 0, WINDOW_LEN // 2
    change_points, trace = [], []
    while start < len(series):
        end = min(start + WINDOW_LEN, len(series))
        algo = ChangePointDetector(significance_tester=tester, calculator=PairDistanceCalculator)
        new_cps = algo.get_change_points(list(series), start, end)   # list input = real windowing
        last = new_cps[-1].index if new_cps else 0
        start = max(last, start + step)
        idxs = [cp.index for cp in change_points]
        for cp in new_cps:
            kept = cp.index not in idxs
            trace.append((int(cp.index), float(cp.stats.pvalue), kept))
            if kept:
                change_points.append(cp)
                idxs.append(cp.index)
    return change_points, trace


def full_context_pvalue(series, index):
    left, right = series[:index], series[index:]
    t, p = ttest_ind_from_stats(left.mean(), left.std(ddof=1), len(left),
                                right.mean(), right.std(ddof=1), len(right),
                                alternative="two-sided")
    return float(p)


def safe_orig(series, seed):
    try:
        cps, _ = compute_change_points_orig(list(series), max_pvalue=MAX_PVALUE, seed=seed)
        return [int(cp.index) for cp in cps]
    except Exception as e:
        return f"FAILED({type(e).__name__})"


def run_case(seed, prepend):
    series = make_series(seed, prepend)
    c = C + prepend

    weak, trace = instrumented_split(series)
    legacy = merge(copy.copy(weak), list(series), MAX_PVALUE, 0.0)
    core, _ = compute_change_points(series, window_len=WINDOW_LEN, max_pvalue=MAX_PVALUE,
                                    min_magnitude=0.0, core_window=True)
    return {
        "series": series, "c": c,
        "trace": trace,
        "weak": [(int(cp.index), float(cp.stats.pvalue)) for cp in weak],
        "legacy": [int(cp.index) for cp in legacy],
        "core": [int(cp.index) for cp in core],
        "orig": safe_orig(series, seed),
        "full_p": full_context_pvalue(series, c),
    }


def show(res):
    print(f"\n=== prepend={res['c'] - C:3d} -> change point at index {res['c']} "
          f"(full-context p = {res['full_p']:.2e}) ===")
    print("  split() per-window detections (index, window-local p, kept by index-dedup?):")
    for idx, pval, kept in res["trace"]:
        tag = "KEPT" if kept else "dropped (duplicate index)"
        print(f"    idx={idx:4d}  p={pval:.2e}  {tag}")
    tp = dict(res["weak"]).get(res["c"])
    if tp is not None:
        ratio = tp / res["full_p"]
        print(f"  -> weak-list p of the true CP: {tp:.2e}  "
              f"({ratio:.0f}x the full-context p it deserves)")
    else:
        print(f"  -> true CP {res['c']} is NOT in the weak list at all")
    print(f"  legacy split+merge answer : {res['legacy'] or 'NO CHANGE POINTS'}")
    print(f"  windowed core answer      : {res['core'] or 'NO CHANGE POINTS'}")
    print(f"  no-window perm. reference : {res['orig']}")


if __name__ == "__main__":
    print(__doc__)
    for prepend in (0, 15, 30):
        show(run_case(seed=2024, prepend=prepend))

    print("\n=== noiseless sanity check (no bias allowed) ===")
    clean = np.concatenate([np.full(177, 20.0), np.full(73, 21.0)])
    from otava.change_point_divisive.calculator import PairDistanceCalculator as PDC
    for W in (None, 60):
        calc = PDC(clean, window_len=W)
        cand = calc.get_candidate_change_point(slice(0, len(clean)))
        print(f"  window_len={W}: candidate index = {cand.index} (exact: {cand.index == 177})")

    print("\n=== 30-seed robustness (found = detection within +-5 of truth) ===")
    TOL = 5
    stats = {p: {"legacy": 0, "core": 0, "orig": 0, "unfair": 0, "in_weak": 0,
                 "err_legacy": [], "err_core": [], "err_orig": []} for p in (0, 15, 30)}

    def nearest(det_list, truth):
        d = [abs(int(x) - truth) for x in det_list]
        return min(d) if d else None

    for seed in range(30):
        for prepend in (0, 15, 30):
            r = run_case(seed, prepend)
            s = stats[prepend]
            nl = nearest(r["legacy"], r["c"])
            nc = nearest(r["core"], r["c"])
            no = nearest(r["orig"], r["c"]) if isinstance(r["orig"], list) else None
            s["legacy"] += int(nl is not None and nl <= TOL)
            s["core"] += int(nc is not None and nc <= TOL)
            s["orig"] += int(no is not None and no <= TOL)
            if nl is not None and nl <= TOL: s["err_legacy"].append(nl)
            if nc is not None and nc <= TOL: s["err_core"].append(nc)
            if no is not None and no <= TOL: s["err_orig"].append(no)
            weak_p = dict(r["weak"]).get(r["c"])
            if weak_p is not None:
                s["in_weak"] += 1
                s["unfair"] += int(weak_p > 10 * r["full_p"])
    for prepend in (0, 15, 30):
        s = stats[prepend]
        med = lambda xs: int(np.median(xs)) if xs else -1
        print(f"  prepend={prepend:3d}: legacy found {s['legacy']}/30 (med err {med(s['err_legacy'])}) | "
              f"windowed core {s['core']}/30 (med err {med(s['err_core'])}) | "
              f"no-window ref {s['orig']}/30 (med err {med(s['err_orig'])}) | "
              f"weak-p >10x unfair {s['unfair']}/{s['in_weak']}")
