# Errata

## 2026-09-03. The AUROC to F1 bound holds only for concave ROC curves

**What was claimed.** Theorem "AUROC-F1 Bound" in `paper/astar_paper.tex`, the derivation in
`src/symbolic_search/_auroc_proof.py`, and the pruning bounds in `run_astar_v2.py` and
`src/symbolic_search/_heuristic_dag.py` all rested on the statement that any ROC curve with
area A has maximum Youden index J = max(TPR − FPR) at most 2A − 1.

**Why it is wrong.** The statement holds for concave ROC curves (equivalently, scores whose
likelihood ratio is monotone), because a concave curve lies above the two chords through its
Youden point and so has area at least (1 + J)/2. It fails for non-concave curves. Counterexample,
found by an external reviewer of the companion textbook: rank three quarters of the positives
above every negative and the remaining quarter below every negative. AUROC is 0.75, the maximum
Youden index is 0.75, and at prevalence one half the best thresholded F1 is 0.857, above the
0.80 the AUROC form of the ceiling allows. The `auroc_f1_bound` formula in `run_astar_v2.py`,
2Aπ/(Aπ + (1 − A)(1 − π)), is not a valid upper bound either: at prevalence 0.1 the same score
reaches F1 0.857 against a "bound" of 0.5.

**Consequence for reported results.** The searches that used these bounds to prune could have
discarded a candidate formula whose ROC curve was not concave and whose true best F1 exceeded the
computed ceiling. The formulas and wins reported by those runs are therefore what a pruned search
found, and are lower bounds on what an unpruned search would find. The Monotone Invariance
Theorem and the results that do not depend on the AUROC bound are unaffected.

**Correction.** The sound form of the ceiling substitutes the measured maximum Youden index of
the candidate's own scores for 2A − 1. It holds for every score, costs the same sort as AUROC,
and is what `max_f1_for_youden` and `youden_f1_bound` now compute. The AUROC form is retained
under its original names with a docstring stating the concavity condition, and the pruning sites
now use the measured index. A rerun of the published searches under the corrected bound was done on
2026-09-03 and 2026-09-04, `rerun_youden_2026_09_03.py`, results in
`results/rerun_youden_2026_09_03.json`, run on the Atlas workstation, CPU only.

**Rerun, part A, the pre-filter admissibility table (`tab:auroc`).** Depth-2 pairwise
enumeration on the eight datasets, exhaustive against the AUROC pre-filter at the seven
published thresholds, with the sound Youden ceiling beside it. The evaluation counts under
the pre-filter reproduce the published table (Breast Cancer 8700 pairs, 2576 evaluated at
0.75; Wine 1560, 413; Circles 20, 4; Moons 20, 2). The pre-filter lost no optimum on any
dataset at any threshold, which is what the paper reported. That was luck of the data, not
the theorem: the retracted ceiling sat below the F1 a pair formula actually reached for 812
of 8700 pairs on Breast Cancer, 412 of 1560 on Wine, 57 of 3800 on Synthetic (20), 18 of 900
on Synthetic (10), 14 of 20 on Circles, and 3 of 20 on Moons. The Youden ceiling, pruning a
pair when its ceiling falls below the best F1 so far, was admissible on all eight datasets
and evaluated 199 pairs on Breast Cancer where the pre-filter evaluated 2576, and 38 on Wine
against 413.

**Rerun, part B, the A* results table (`tab:astar_vs_phased`).** Depth-3 A* search with
50,000 expansions in three modes: strict (no pre-filter, the reference), fast (AUROC
pre-filter at 0.52, as published), and the new `youden` mode.

| Dataset | strict F1, formula | fast F1 | youden F1, formula | expansions strict / fast / youden |
|---|---|---|---|---|
| Circles (2) | 0.9965, x1 hypot x2 | 0.9965 | 0.9965, same | 1626 / 540 / 87 |
| Moons (2) | 0.8927, neg(x1) max x2 | 0.8927 | 0.8927, same | 1626 / 1386 / 86 |
| Synthetic (20) | 1.0000, x0 hypot x1 | 1.0000 | 1.0000, same | 50000 / 50000 / 465 |
| Breast Cancer (10) | 0.9578, (f1 min f2) + f7 | 0.9578 | 0.9516, (f3 + f7) min f6 | 50000 / 50000 / 841 |
| Diabetes (8) | 0.6937, (d5 min d7) + d1 | 0.6937 | 0.6921, (d1 + d7) min d5 | 50000 / 46307 / 192 |

Strict and fast agree on every dataset, and the Breast Cancer and Diabetes optima and
formulas reproduce the published table (0.958 and 0.694). Moons reads 0.893 here against the
published 0.902, a difference in the generated instance rather than in the search. The
3-body row was not rerun, since it needs the tensor landscape. The `youden` mode reaches the
optimum on three datasets in one to two percent of the expansions and falls short by 0.006 and
0.002 on the other two: it prunes a subtree by its root's ceiling, which is sound for the root
and a heuristic for its descendants, exactly as the AUROC pre-filter is, and on those two
datasets the pruned root had a better descendant. Neither mode is a guaranteed search; strict
is.

**What stands after the rerun.** The published A* results are reproduced. The published
admissibility claim is reproduced as an empirical fact about those datasets and withdrawn as
a theorem. The corrected code prunes leaves soundly and subtrees heuristically, and says which.

Reported in *Data Mining as Observation*, chapter 5 section 5.4, which prints the counterexample.
