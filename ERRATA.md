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
now use the measured index. A rerun of the published searches under the corrected bound is an
open item and has not been done as of this entry.

Reported in *Data Mining as Observation*, chapter 5 section 5.4, which prints the counterexample.
