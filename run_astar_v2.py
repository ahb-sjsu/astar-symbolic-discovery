#!/usr/bin/env python3
"""
A* formula search v2: Fix the failed theoretical results.

Key insight from v1 failures:
- "Monotone" transforms are not monotone on mixed-sign data
- KNN on raw features doesn't bound transformed-feature F1
- Need to bound F1 of the OPERATION OUTPUT, not the raw features

New approach: ORDER-STATISTIC BOUND
- For any 1D thresholded predictor on N samples with P positives,
  the maximum achievable F1 is bounded by the AUROC of the values
- AUROC is computable in O(N log N) and is invariant under monotone
  transforms (this IS provable — AUROC depends only on rank order)
- For truly monotone transforms: AUROC(f) = AUROC(g(f))
- Therefore: F1_max(g(f)) ≤ f(AUROC(f)) where f is the
  AUROC-to-F1 upper bound function

This gives us a CORRECT admissible heuristic for monotone-only search.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles, make_moons, load_breast_cancer
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, "src")
from symbolic_search._search import _f1_threshold_sweep
from symbolic_search._ops import BINARY_OPS_MINIMAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def auroc_f1_bound(auroc: float, prevalence: float) -> float:
    """RETIRED 2026-09-03. Not a valid upper bound on F1 (see ERRATA.md).

    Kept only so old call sites fail loudly rather than silently prune.
    """
    raise RuntimeError("auroc_f1_bound is not a valid bound; use youden_f1_bound with the measured index")


def max_youden_index(values: np.ndarray, actual: np.ndarray) -> float:
    """Measured maximum Youden index over thresholds and both directions; O(N log N), no assumptions."""
    values = np.asarray(values, dtype=float)
    y = np.asarray(actual).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-values, kind="stable")
    v, ys = values[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    last = np.ones(len(v), dtype=bool)
    last[:-1] = v[:-1] != v[1:]
    j_desc = tp[last] / n_pos - fp[last] / n_neg
    j_asc = fp[last] / n_neg - tp[last] / n_pos  # the other threshold direction
    return float(max(0.0, j_desc.max(), j_asc.max()))


def youden_f1_bound(youden: float, prevalence: float, n_points: int = 200) -> float:
    """Upper bound on thresholded F1 from the maximum Youden index J.

    FPR >= TPR - J at every operating point, so F1 <= 2 t pi / (t pi + pi + (t - J)(1 - pi))
    over t in [J, 1]. Holds for every score. This replaces the AUROC form, which held
    only for concave ROC curves (ERRATA.md, 2026-09-03).
    """
    J = float(min(max(youden, 0.0), 1.0))
    best = 0.0
    for tpr in np.linspace(max(0.01, J), 1.0, n_points):
        fpr = tpr - J
        if fpr < 0 or fpr > 1:
            continue
        prec = tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence) + 1e-30)
        if prec + tpr > 0:
            best = max(best, 2 * prec * tpr / (prec + tpr))
    return best


def compute_auroc(values: np.ndarray, actual: np.ndarray) -> float:
    """Compute AUROC, handling edge cases."""
    try:
        vals = np.nan_to_num(values, nan=0, posinf=1e10, neginf=-1e10)
        if len(np.unique(actual)) < 2:
            return 0.5
        auc = roc_auc_score(actual, vals)
        return max(auc, 1 - auc)  # direction-invariant
    except Exception:
        return 0.5


def main():
    log.info("=" * 70)
    log.info("A* FORMULA SEARCH v2: Correct Admissible Heuristic")
    log.info("=" * 70)

    # ============================================================
    # Key theoretical result: AUROC is rank-order invariant
    # ============================================================
    log.info("\n--- AUROC Rank-Order Invariance ---")
    log.info("Theorem: For any strictly monotone g, AUROC(g(f)) = AUROC(f)")
    log.info("Proof: AUROC depends only on the rank ordering of values.")
    log.info("       Monotone transforms preserve rank order. QED.")

    rng = np.random.RandomState(42)
    # Verify empirically
    N = 1000
    values = rng.randn(N)
    actual = (values + rng.randn(N) * 0.5 > 0).astype(bool)

    auroc_base = compute_auroc(values, actual)
    truly_monotone = {
        "2x+3": lambda x: 2 * x + 3,
        "x^3": lambda x: x**3,
        "exp": lambda x: np.exp(np.clip(x, -500, 500)),
        "neg": lambda x: -x,
    }
    NOT_monotone = {
        "x^2": lambda x: x**2,
        "abs": lambda x: np.abs(x),
        "sin": lambda x: np.sin(x),
    }

    log.info("\n  Truly monotone transforms (should preserve AUROC = %.4f):", auroc_base)
    for name, fn in truly_monotone.items():
        auroc_t = compute_auroc(fn(values), actual)
        diff = abs(auroc_t - auroc_base)
        log.info(
            "    %10s: AUROC = %.4f (diff = %.6f) %s",
            name,
            auroc_t,
            diff,
            "OK" if diff < 0.001 else "VIOLATION",
        )

    log.info("\n  Non-monotone transforms (may change AUROC):")
    for name, fn in NOT_monotone.items():
        auroc_t = compute_auroc(fn(values), actual)
        diff = abs(auroc_t - auroc_base)
        log.info(
            "    %10s: AUROC = %.4f (diff = %.6f) %s",
            name,
            auroc_t,
            diff,
            "CHANGED" if diff > 0.01 else "same",
        )

    # ============================================================
    # AUROC-based pruning for Phase 2
    # ============================================================
    log.info("\n" + "=" * 70)
    log.info("AUROC-BASED PRUNING FOR PAIRWISE SEARCH")
    log.info("=" * 70)
    log.info("Strategy: For each operation b and pair (i,j), compute")
    log.info("AUROC(b(f_i, f_j)) in O(N log N). If the AUROC-to-F1")
    log.info("upper bound is below current best F1, skip the full")
    log.info("threshold sweep.")

    datasets = {}
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=42)
    datasets["Circles"] = (X, y, ["x1", "x2"])
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    datasets["Moons"] = (X, y, ["x1", "x2"])
    bc = load_breast_cancer()
    X = StandardScaler().fit_transform(bc.data[:, :15])
    datasets["Breast Cancer (15D)"] = (X, bc.target, [f"f{i}" for i in range(15)])
    rng = np.random.RandomState(42)
    X = rng.randn(2000, 20)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 2).astype(int)
    datasets["Synthetic (20D)"] = (X, y, [f"x{i}" for i in range(20)])

    for name, (X, y, features) in datasets.items():
        N, d = X.shape
        actual = y.astype(bool)
        prevalence = actual.mean()

        # Exhaustive search
        t0 = time.time()
        best_f1_exhaustive = 0
        best_formula_exhaustive = ""
        formulas_exhaustive = 0

        for i in range(d):
            f1, _, _ = _f1_threshold_sweep(X[:, i], actual)
            formulas_exhaustive += 1
            if f1 > best_f1_exhaustive:
                best_f1_exhaustive = f1
                best_formula_exhaustive = features[i]

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for op_name, op_fn in BINARY_OPS_MINIMAL.items():
                    try:
                        vals = op_fn(X[:, i], X[:, j])
                        f1, _, _ = _f1_threshold_sweep(vals, actual)
                        formulas_exhaustive += 1
                        if f1 > best_f1_exhaustive:
                            best_f1_exhaustive = f1
                            best_formula_exhaustive = f"{features[i]} {op_name} {features[j]}"
                    except Exception:
                        formulas_exhaustive += 1
        time_exhaustive = time.time() - t0

        # AUROC-pruned search
        t0 = time.time()
        best_f1_pruned = 0
        best_formula_pruned = ""
        formulas_evaluated = 0
        formulas_pruned = 0

        for i in range(d):
            f1, _, _ = _f1_threshold_sweep(X[:, i], actual)
            formulas_evaluated += 1
            if f1 > best_f1_pruned:
                best_f1_pruned = f1
                best_formula_pruned = features[i]

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for op_name, op_fn in BINARY_OPS_MINIMAL.items():
                    try:
                        vals = op_fn(X[:, i], X[:, j])
                        vals = np.nan_to_num(vals, nan=0, posinf=1e10, neginf=-1e10)

                        # Youden-based pruning: sound for every score (ERRATA.md 2026-09-03)
                        auroc = compute_auroc(vals, actual)
                        f1_bound = youden_f1_bound(max_youden_index(vals, actual), prevalence)

                        if f1_bound < best_f1_pruned:
                            # Prune: AUROC bound says this can't beat current best
                            formulas_pruned += 1
                            continue

                        # Full threshold sweep (expensive)
                        f1, _, _ = _f1_threshold_sweep(vals, actual)
                        formulas_evaluated += 1
                        if f1 > best_f1_pruned:
                            best_f1_pruned = f1
                            best_formula_pruned = f"{features[i]} {op_name} {features[j]}"
                    except Exception:
                        formulas_evaluated += 1

        time_pruned = time.time() - t0

        f1_match = abs(best_f1_exhaustive - best_f1_pruned) < 0.001
        speedup = time_exhaustive / max(time_pruned, 0.001)
        prune_rate = formulas_pruned / max(formulas_pruned + formulas_evaluated, 1)

        log.info("\n  %s:", name)
        log.info(
            "    Exhaustive: F1=%.3f, formulas=%d, time=%.2fs, formula=%s",
            best_f1_exhaustive,
            formulas_exhaustive,
            time_exhaustive,
            best_formula_exhaustive,
        )
        log.info(
            "    AUROC-pruned: F1=%.3f, formulas=%d, pruned=%d (%.0f%%), time=%.2fs",
            best_f1_pruned,
            formulas_evaluated,
            formulas_pruned,
            100 * prune_rate,
            time_pruned,
        )
        log.info(
            "    Speedup: %.1fx, F1 match: %s",
            speedup,
            "YES" if f1_match else "NO *** ADMISSIBILITY VIOLATION ***",
        )

    log.info("\n" + "=" * 70)
    log.info("CONCLUSION")
    log.info("=" * 70)
    log.info("The Youden-based bound is ADMISSIBLE: it never prunes a formula")
    log.info("that would have been the best. This is because:")
    log.info("1. FPR >= TPR - J at every threshold, J the measured maximum Youden index")
    log.info("2. The bound function youden_f1_bound(J, prevalence) is therefore an")
    log.info("   upper bound on achievable F1 for every score (the earlier AUROC")
    log.info("   form held only for concave ROC curves; see ERRATA.md 2026-09-03)")
    log.info("3. J is O(N log N), one sort, the same cost as AUROC")
    log.info("4. The pruning reduces expensive threshold sweeps while guaranteeing")
    log.info("   the same optimal formula is found")


if __name__ == "__main__":
    main()
