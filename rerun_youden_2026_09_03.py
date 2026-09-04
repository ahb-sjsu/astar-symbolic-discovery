#!/usr/bin/env python3
"""Rerun of the A* paper's pruning experiments under the corrected bound (ERRATA.md, 2026-09-03).

Part A reproduces Table tab:auroc of paper/astar_paper.tex: depth-2 pairwise enumeration on
eight datasets, exhaustive against the AUROC pre-filter at seven thresholds, and adds the
sound alternative, pruning a pair when the F1 ceiling from its measured Youden index is
below the best F1 found so far. For every dataset it also counts the pairs whose exact F1
exceeds the AUROC form of the ceiling, which is the number of times the retracted theorem
would have been wrong on that dataset.

Part B reproduces Table tab:astar_vs_phased for the datasets that need no external data:
TheoryRadar A* search at depth 3 in strict mode (no pre-filter, the reference), fast mode
(AUROC pre-filter at 0.52, as published), and the new youden mode.

Writes results/rerun_youden_2026_09_03.json and prints a summary. CPU only.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from symbolic_search._auroc_proof import max_f1_for_auroc, max_f1_for_youden, max_youden_index  # noqa: E402
from symbolic_search._ops import BINARY_OPS  # noqa: E402
from symbolic_search._search import _f1_threshold_sweep  # noqa: E402
from symbolic_search.radar import TheoryRadar  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

THRESHOLDS = [0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]


def exact_optimal_f1(values, actual):
    f1, _, _ = _f1_threshold_sweep(values, actual)
    return float(f1)


def auroc(values, actual):
    try:
        a = roc_auc_score(actual, values)
        return float(max(a, 1 - a))
    except Exception:
        return 0.5


def datasets_part_a():
    out = {}
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=42)
    out["Circles (2)"] = (X, y)
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    out["Moons (2)"] = (X, y)
    bc = load_breast_cancer()
    out["Breast Cancer (30)"] = (StandardScaler().fit_transform(bc.data), bc.target)
    w = load_wine()
    out["Wine (13), class 0 vs rest"] = (StandardScaler().fit_transform(w.data), (w.target == 0).astype(int))
    for d in (10, 20, 30, 40):
        rng = np.random.RandomState(42)
        X = rng.randn(2000, d)
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 2).astype(int)
        out[f"Synthetic ({d})"] = (X, y)
    return out


def part_a():
    rows = []
    for name, (X, y) in datasets_part_a().items():
        N, d = X.shape
        actual = y.astype(bool)
        prev = float(actual.mean())
        t0 = time.time()
        # one enumeration, every quantity recorded per pair
        pairs = []  # (f1, auroc, youden)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                for op_fn in BINARY_OPS.values():
                    try:
                        v = np.nan_to_num(op_fn(X[:, i], X[:, j]), nan=0, posinf=1e10, neginf=-1e10)
                        pairs.append((exact_optimal_f1(v, actual), auroc(v, actual), max_youden_index(v, actual)))
                    except Exception:
                        pairs.append((0.0, 0.5, 0.0))
        best_single = max(exact_optimal_f1(X[:, i], actual) for i in range(d))
        best_ex = max([best_single] + [p[0] for p in pairs])
        # AUROC pre-filter at each threshold, as published
        per_alpha = {}
        for a in THRESHOLDS:
            kept = [p for p in pairs if p[1] >= a]
            best_a = max([best_single] + [p[0] for p in kept])
            per_alpha[str(a)] = {"evaluated": len(kept), "best_f1": best_a, "admissible": abs(best_a - best_ex) < 1e-9}
        # Youden ceiling pruning in enumeration order, sound for every score
        best_y = best_single
        evaluated_y = 0
        for f1, a, J in pairs:
            if max_f1_for_youden(J, prev) < best_y:
                continue
            evaluated_y += 1
            best_y = max(best_y, f1)
        # how often the retracted theorem is wrong on this dataset
        violations = sum(1 for f1, a, J in pairs if f1 > max_f1_for_auroc(a, prev) + 1e-9)
        nonconcave = sum(1 for f1, a, J in pairs if J > 2 * a - 1 + 1e-9)
        rows.append({
            "dataset": name, "N": N, "d": d, "prevalence": prev, "pairs": len(pairs),
            "best_f1_exhaustive": best_ex, "auroc_prefilter": per_alpha,
            "youden_prune": {"evaluated": evaluated_y, "best_f1": best_y, "admissible": abs(best_y - best_ex) < 1e-9},
            "pairs_with_f1_above_auroc_ceiling": violations,
            "pairs_with_youden_above_2A_minus_1": nonconcave,
            "seconds": time.time() - t0,
        })
        print(f"[A] {name}: pairs {len(pairs)}, exhaustive F1 {best_ex:.4f}, "
              f"AUROC@0.75 evaluated {per_alpha['0.75']['evaluated']} admissible {per_alpha['0.75']['admissible']}, "
              f"Youden evaluated {evaluated_y} admissible {abs(best_y - best_ex) < 1e-9}, "
              f"ceiling violations {violations}, non-concave {nonconcave}, {time.time() - t0:.0f}s", flush=True)
    return rows


def datasets_part_b():
    out = {}
    X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5, random_state=42)
    out["Circles (2)"] = (X, y, ["x1", "x2"])
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
    out["Moons (2)"] = (X, y, ["x1", "x2"])
    rng = np.random.RandomState(42)
    X = rng.randn(2000, 20)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 2).astype(int)
    out["Synthetic (20)"] = (X, y, [f"x{i}" for i in range(20)])
    bc = load_breast_cancer()
    out["Breast Cancer (10)"] = (StandardScaler().fit_transform(bc.data[:, :10]), bc.target, [f"f{i}" for i in range(10)])
    try:
        pima = fetch_openml("diabetes", version=1, as_frame=False)
        X = StandardScaler().fit_transform(pima.data.astype(float))
        y = (np.asarray(pima.target) == "tested_positive").astype(int)
        out["Diabetes (8)"] = (X, y, [f"d{i}" for i in range(X.shape[1])])
    except Exception as e:  # noqa: BLE001
        print("Diabetes unavailable offline:", e, flush=True)
    return out


def part_b():
    rows = []
    for name, (X, y, feats) in datasets_part_b().items():
        row = {"dataset": name, "N": int(X.shape[0]), "d": int(X.shape[1])}
        for mode in ("strict", "fast", "youden"):
            radar = TheoryRadar(X, y.astype(int), feats)
            t0 = time.time()
            r = radar.search(mode=mode, f1_target=0, max_depth=3, max_expansions=50000,
                             auroc_threshold=0.52, verbose=False)
            row[mode] = {"f1": r.f1, "formula": r.formula, "depth": r.depth, "expansions": r.expansions,
                         "pruned_auroc": r.pruned_auroc, "pruned_youden": r.pruned_youden,
                         "pruned_monotone": r.pruned_monotone, "seconds": time.time() - t0}
            print(f"[B] {name} {mode}: F1 {r.f1:.4f} depth {r.depth} exp {r.expansions} "
                  f"pruned auroc {r.pruned_auroc} youden {r.pruned_youden} {row[mode]['seconds']:.0f}s  {r.formula}", flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    out = {"date": "2026-09-03", "commit": os.popen("git rev-parse --short HEAD").read().strip(),
           "part_a": part_a(), "part_b": part_b()}
    with open("results/rerun_youden_2026_09_03.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/rerun_youden_2026_09_03.json", flush=True)
