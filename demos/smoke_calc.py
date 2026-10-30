import sys
sys.path.insert(0, '.')
import numpy as np
from otava.change_point_divisive.calculator import PairDistanceCalculator


def brute_windowed_Q(seq, start, end, half):
    """Pure-definition windowed Q matrix (values only, no cumsum tricks)."""
    size = end - start - 1
    Q = np.full((size, size), -np.inf)
    for tau in range(start + 1, end):
        left = max(start, tau - half)
        for kappa in range(tau + 1, min(end, tau + half) + 1):
            X, Y = seq[left:tau], seq[tau:kappa]
            n, m = len(X), len(Y)
            A = sum(abs(x - y) for x in X for y in Y) * 2 / (m + n)
            B = sum(abs(X[i] - X[k]) for i in range(n - 1) for k in range(i + 1, n))
            B = B * 2 * m / (m + n) / (n - 1) if n > 1 else 0.0
            C = sum(abs(Y[j] - Y[k]) for j in range(m - 1) for k in range(j + 1, m))
            C = C * 2 * n / (m + n) / (m - 1) if m > 1 else 0.0
            Q[tau - start - 1, kappa - start - 2] = A - B - C
    return Q


rng = np.random.default_rng(7)
seq = np.round(rng.normal(0, 1, 30), 3)
seq[15:20] += 4.0

for W in (4, 6, 10, 16, 60):
    calc = PairDistanceCalculator(seq, window_len=W)
    half = W // 2
    Qm = calc._get_Q_vals_windowed(0, len(seq))
    Qb = brute_windowed_Q(seq, 0, len(seq), half)
    mask = np.isfinite(Qb)
    ok_matrix = np.allclose(Qm[mask], Qb[mask], atol=1e-9)
    cand = calc.get_candidate_change_point(slice(0, len(seq)))
    i, j = np.unravel_index(np.argmax(Qb), Qb.shape)
    ok_scan = (cand.index == i + 1) and np.isclose(cand.qhat, Qb[i, j], atol=1e-9)
    print(f"W={W:3d} matrix==brute:{ok_matrix}  scan==brute-argmax:{ok_scan} "
          f"(tau={cand.index}, qhat={cand.qhat:.4f})")

# sub-interval check (start > 0)
W = 8
calc = PairDistanceCalculator(seq, window_len=W)
Qm = calc._get_Q_vals_windowed(5, 25)
Qb = brute_windowed_Q(seq, 5, 25, W // 2)
mask = np.isfinite(Qb)
print("sub-interval matrix==brute:", np.allclose(Qm[mask], Qb[mask], atol=1e-9))

# large-window equivalence vs unwindowed
big = np.round(rng.normal(0, 1, 40), 3)
big[20:26] += 3.0
calc_u = PairDistanceCalculator(big)                      # unwindowed
Qu = calc_u._get_Q_vals(0, len(big))
cand_u = calc_u.get_candidate_change_point(slice(0, len(big)))
calc_w = PairDistanceCalculator(big, window_len=2 * len(big))
cand_w = calc_w.get_candidate_change_point(slice(0, len(big)))
print(f"equivalence W=2N: index {cand_w.index} vs {cand_u.index}, "
      f"qhat {cand_w.qhat:.6f} vs {cand_u.qhat:.6f}, "
      f"match: {cand_w.index == cand_u.index and np.isclose(cand_w.qhat, cand_u.qhat)}")
