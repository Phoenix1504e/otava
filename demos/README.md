# Windowed e-divisive prototype for apache/otava #84

**Branch:** `windowed-qvals` (patch in `windowed-qvals.patch`, based on master `24ef337`)
**Status:** all 46 existing + new tests pass; correctness demos included.

## What this does

Pushes `window_len` down into the core e-divisive computation instead of
chunking the series into overlapping windows:

- `PairDistanceCalculator(series, window_len=W)` evaluates
  `Q_w(tau, kappa) = QQ(series[max(start, tau - W//2) : tau], series[tau : min(end, tau + W//2)])`
  — the exact `Q` function of the window-local slice, with window-local
  `A/B/C` normalizations. `window_len=None` keeps the current behavior bit
  for bit.
- The candidate scan sweeps the `kappa - tau <= W//2` band along its short
  side (vectorized over `tau`): `O(N*W)` time per candidate search, no
  `(tau, kappa)` matrix.
- Windowed candidates carry their **evidence window**
  (`WindowedCandidateChangePoint.window`); the first-pass significance test
  judges the candidate on exactly that window
  (`SignificanceTester.get_sides` clips to it). This is required for
  consistency: testing a locally-found candidate on full-interval sides
  dilutes short excursions and they can never pass.
- `compute_change_points(..., core_window=True)` (default) runs a single
  detection pass with the windowed calculator and the unchanged merge step.
  The legacy chunked path stays available via `core_window=False`
  (and is used automatically for incremental append, which the prototype
  does not cover yet).
- Windowed calculator composes with the permutation test as well
  (`PermutationsSignificanceTester(calculator=partial(PairDistanceCalculator,
  window_len=W))`).

## Validation

`tests/windowed_calculator_test.py`:

1. **Brute-force agreement** — for 5 seeds x 5 window sizes and for
   sub-intervals (`start > 0`): the windowed matrix, the production scan and
   a pure-definition brute force all agree to 1e-8, and the argmax
   (index + qhat) matches.
2. **Large-window equivalence** — `window_len >= 2 * len(series)` reproduces
   the unwindowed candidate (index and qhat) exactly.
3. **Close change points** — two CPs 40 apart with `window_len=60` (the
   motivating case of #84) are both found.
4. **End-to-end** — `compute_change_points(core_window=True)` on a 3-CP
   series finds all of them; the legacy path still passes its own tests.

## Correctness experiments (the "examples of data sets" the maintainer asked for)

All scripts standalone; results in `*_results.txt`, figures in `figures/`.

### 1. Alignment stability / p-value fairness (demo_alignment.py, fig1, fig2)

One clean +2 sigma step at index 177 in a 250-point series, `window_len=60`.
The data around the change point is bit-identical in all cases; only the
window-grid alignment shifts (0 / 15 / 30 unrelated flat points prepended).

| metric (30 seeds) | legacy split+merge | windowed core | no-window perm ref |
|---|---|---|---|
| found, prepend=0 | 23/30 | **24/30** | 29/30 |
| found, prepend=15 | 29/30 | **24/30** | 29/30 |
| found, prepend=30 | 24/30 | **24/30** | 29/30 |
| weak-list p > 10x unfair | **32/32** | 0 | - |

The legacy path's *answer* for the same regression depends on how much
unrelated history precedes it, and whenever its weak list contains the true
CP, the p-value kept by the index-dedup is **always** (>10x in 32/32 cases)
worse than the full-context p-value of the same CP - in the shown seed the
kept p-values are 6e-3, 2e-8 and 5e-3 (spanning 6 orders of magnitude; two of
them above `max_pvalue=0.001`, so the CP is dropped) while the windowed core
attaches 3e-25 at every alignment (fig2).

Root cause: split() judges a CP with window-local stats whose context depends
on where the chunk grid falls, keeps the first window's stats per index, and
lets merge() prune with those. The core path anchors the window at the
*candidate*, so the evidence is a deterministic function of the data.

### 2. Randomized stress sweep (demo_stress.py)

250-point series, `window_len=60`, 30 seeds per config. "found" = detection
within +-5 points; "spurious" = final-answer CPs further than 5 points from
any true CP.

| config | legacy | core (t-test) | core (permutation) | no-window perm ref |
|---|---|---|---|---|
| single, delta=1.0 sigma | 18/30 (sp 12) | 22/30 (sp 46) | 8/30 (sp 1) | 21/30 (sp 9) |
| single, delta=1.5 sigma | 28/30 (sp 8) | **29/30** (sp 36) | 24/30 (sp 2) | 29/30 (sp 1) |
| single, delta=2.0 sigma | 30/30 (sp 6) | 30/30 (sp 34) | 29/30 (sp 0) | 30/30 (sp 0) |
| single, delta=3.0 sigma | 30/30 (sp 6) | 30/30 (sp 32) | 30/30 (sp 0) | 30/30 (sp 0) |
| twins 20 apart | 60/60 (sp 6) | **60/60** (sp 34) | 56/60 (sp 3) | 60/60 (sp 0) |
| twins 40 apart | 59/60 (sp 8) | 59/60 (sp 31) | 59/60 (sp 1) | 59/60 (sp 1) |
| twins 80 apart | 60/60 (sp 7) | 60/60 (sp 35) | 60/60 (sp 0) | 60/60 (sp 0) |
| triple 70/140/210 | 87/90 (sp 9) | **88/90** (sp 40) | 82/90 (sp 4) | 88/90 (sp 2) |

Learnings:
- **Detection recall: parity or better** for the t-test core at every config,
  including the twins closer than the window (#84's motivating case).
- **Purity:** the t-test core's relaxed first pass over many candidate
  positions admits ~1 spurious weak CP per run that survives merge when
  `min_magnitude=0` (legacy: ~0.2/run). The permutation null fixes it
  (spurious 0-4 per 30 runs) at the cost of some recall on subtle shifts.
- **The power trade-off:** windowing caps the evidence at `W` points, so
  subtle shifts (~1.5 sigma at W=60) lose power vs the no-window statistic
  (8/30 vs 21/30 at delta=1.0). W is the recall/cost knob.
- **Stop-at-first-insignificant:** with local (noisy) qhats, the detector
  loop can stop before reaching a real CP when a noise candidate outranks it
  and fails its local test (costs a few seeds at delta ~2 sigma). Candidate
  ordering / termination policy is an open design question.

### 3. The ndarray trap (demo_phantom.py)

`ChangePointDetector.get_change_points(series, start, end)` only slices
`series[start:end]` for **list** input. For ndarray input it detects over the
whole series and shifts the found indexes by `start`. With split()'s window
loop this produces change points *beyond the end of the series* (480, 510,
660 on a 450-point series) and a `ValueError` crash. The windowed core path
works with ndarray input as-is.

### 4. Cost (demo_perf.py, fig3)

End-to-end wall clock, T-test paths (production default):

| N | legacy split+merge | windowed core |
|---|---|---|
| 1000 | 42 ms | 94 ms |
| 2000 | 89 ms | 232 ms |
| 8000 | 361 ms | 3579 ms |

Honest reading: as implemented, the core path is *slower* than chunking for
the T-test path. Three reasons, all addressable:

1. a one-time `O(N^2)` cumsum build per calculator (2 GB at N=16000 - both
   paths are `O(N^2)`-memory today); a **banded cumsum** (all accesses are
   within `W/2` of the diagonal) makes it `O(N*W)` time and memory;
2. the scan uses fancy-indexing gathers against the full cumsum; contiguous
   diagonal extraction would cut the constant factor;
3. the weak-candidate flood multiplies rescan work (see purity above).

The windowed scan itself is cheap and linear (isolated: 48 ms per candidate
eval at N=10000). The permutation paths pay `O(N^2)` per permutation for the
distance/cumsum setup regardless of variant (core+perm is 1.6x faster than
the no-window reference at N=2000); with a banded cumsum the whole
permutation path would drop to `O(perms * N * W)`.

## How to reproduce

    git clone https://github.com/apache/otava && cd otava
    git apply windowed-qvals.patch
    pip install numpy scipy pytest
    python -m pytest tests/windowed_calculator_test.py tests/change_point_divisive_test.py tests/analysis_test.py
    python demo_alignment.py && python demo_stress.py && python demo_perf.py && python demo_phantom.py

## Open design questions

1. Should the first-pass significance test stay at Hunter's 10x-relaxed
   threshold? The relaxation compensated for chunk-grid truncation; with
   candidate-anchored windows there is no truncation, so the strict
   `max_pvalue` (or the permutation null) may be the better-calibrated
   choice.
2. Termination policy of the detection loop (stop at first insignificant
   candidate vs collect all candidates passing their local test).
3. Odd `window_len` handling (`W // 2` on both sides here; `ceil` on the
   right would give exactly W points).
4. Incremental append (`new_data` / `old_weak_cp`): the prototype falls back
   to the chunked path for it; rerunning detection from the last accepted CP
   (as in #154's approach) should compose naturally with the windowed core.
5. Should `split()` be kept as an opt-in legacy path, or removed once the
   core windowing is validated?
