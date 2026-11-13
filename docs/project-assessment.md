# Project assessment

Reviewed November 13, 2026 against the supplied project plan and four course-guideline screenshots.

## Verdict

This is a good, feasible problem-driven optimization term project for a team of three. M1 and M2 provide a sufficient core; M3 and M4 add useful depth. Free SciPy/HiGHS solvers avoid licensing risk. Computational feasibility looks favorable because the principal LP has employee-category counts rather than ticket-level scheduling variables. This is an architectural assessment, not a measured runtime claim.

The supplied guidelines require a practical decision problem, optimization, benchmarks, computational results, and recommendations. The plan covers each. Sensitivity analysis and fairness/cross-training extensions align with the stronger-project criteria. The screenshots do not establish the submission deadline, page limits, video requirements, or AI policy; verify these against the complete syllabus. December 7, 2026 is Monday, not Sunday. Several schedule weekday labels also need correction.

## Evidence and verification boundary

The supplied PDF was read in full. Its appendix explicitly says preliminary numbers must be reproduced. No CSVs have been inspected and no preliminary claims have been independently reproduced.

Publisher: https://data.mendeley.com/datasets/btm76zndnt/3 (DOI 10.17632/btm76zndnt.3). The publisher describes real help-desk records from January 2007 through March 2023, masked identifiers, issue histories and assignee snapshots, and a CC BY 4.0 license. Filenames differ between documentation (`issues_snapshots.csv`) and the plan (`issues_snapshot.csv`); the loader should support both. Terminal DNS failed during attempted retrieval and browser access was unavailable.

## Corrections needed for credible results

1. **Counterfactual identification.** Employee-category averages are observational associations. Dispatchers select tickets using information omitted from this model. Test-period cell averages do not reveal the outcome of assigning the same ticket to another employee. Present results as model-based retrospective comparisons, not measured service improvements or employee ability rankings.
2. **Backlog accounting.** The scoring formula omits unassigned tickets. Report backlog separately and include a common penalized log-cost per arriving ticket. Report resolution-time summaries only for routed tickets, with their coverage. Never exponentiate a large backlog penalty and label it resolution hours. Prefer comparison of backlog first, then service proxy at equal coverage.
3. **Decision-time availability.** Employees observed receiving tickets during a test week form a hindsight roster. Reproduce this only as an explicitly labeled retrospective scenario; include a roster based on prior information or a supplied staffing roster. Weekly category demand is also assumed known in a batch-routing problem, rather than predicted online.
4. **Outcome-time leakage.** A ticket created before a fitting cutoff may resolve after it. Provide strict as-of fitting using resolutions observable by that cutoff, alongside the plan's creation-date split for reproduction. Final snapshot priority/type may also be post-arrival fields; inspect the dictionary and history before describing features as available at routing time.
5. **Capacity estimation.** Specify whether the weekly 90th percentile includes zero-assignment weeks. Nonzero-only weeks inflate inferred capacity. Use an explicit calendar and document the chosen convention. These are routing slots, not measured labor capacity or queue-service capacity.
6. **Historical comparator.** Historical assignments can violate inferred eligibility or capacity. Report those violations rather than claiming all methods have the same feasible set. Capacity-aware baselines should use the same roster and constraints as the LP.
7. **Statistical uncertainty.** The pooled sigma/sqrt(n+k) term is a heuristic uncertainty penalty for a shrinkage estimator, not automatically a calibrated confidence interval. Include common-support diagnostics and uncertainty across paired weeks. Select hyperparameters using validation only; do not choose them to reproduce the preliminary narrative.
8. **Units.** exp(mean(log(1+hours)))-1 is a shifted geometric-mean proxy, not expected resolution hours. A capacity dual is log-objective improvement per extra ticket slot. Use finite differences as a local check; integer one-slot improvements need not equal the infinitesimal dual at a breakpoint.
9. **Cross-training.** Predictions for unobserved employee-category pairs are assumptions. Label savings as scenario results, vary new-pair costs, use training weeks, and do not claim causally identified training benefits. Rounding an MIP relaxation is not automatically feasible.
10. **Backlog penalty.** A merely large M need not produce lexicographic maximum service in every network. Prefer two-stage optimization: minimize backlog, then minimize routing cost at that minimum; retain the plan's penalized formulation as a documented reproduction option.

## Scope and report framing

Finish M1/M2 and the honest evaluation first. Add M3, duals and capacity sensitivity next, then the bounded M4. Keep the workflow map descriptive. Excluding labor-hour optimization and scheduling is sensible because calendar duration is not effort. Report whichever results the data supports, including a robust method that fails to beat historical dispatch. Matching the historical score does not establish equivalence without uncertainty analysis.

Suggested central question: How does uncertainty-aware category-level routing trade estimated service time, coverage, and workload balance under inferred capacity constraints?

## Completed implementation verification

The raw CSVs were found locally after this initial assessment. The implementation now reproduces 11,068 usable outcomes, 7,172 training and 3,896 test outcomes. Both scenario pipelines and both notebooks execute successfully. See results/reproduction/RESULTS.md and results/strict/RESULTS.md for actual findings. These supersede the initial verification boundary above; appendix performance claims were not assumed.
