# Results: strict

Generated from the local real Mendeley v3 CSVs. These are retrospective model-based proxies, not causal treatment effects or measured counterfactual improvements.

Usable Done outcomes: 11,068; train/test outcomes: 7,172/3,896. Evaluated 34 test weeks. Validated k=50, z=0.5.

Historical: 69.81; round-robin: 88.16; naive: 74.10; robust: 73.45 mean weekly shifted geometric hours among routed tickets.

Backlogged/unassigned counts: historical 427, round-robin 5, naive 4, robust 4. Historical missing-assignee counts are unknown routing labels, not evidence of operational backlog.

At equal weekly coverage, robust descriptive change versus historical: nan% across 0 weeks; versus round-robin: 16.86%. Positive means lower proxy hours. Comparisons at different coverage are not service improvements.

Maximum LP fractionality: 0. Common-support cell correlation: 0.518; category-adjusted correlation: 0.379, 100 cells.

Capacity finite-difference check: employee 4kpk, week 2023-02-13; negative dual 1.297; one-slot objective drop 95.127. In backlog-first mode duals are conditional on fixed minimum backlog; a capacity change may change that minimum.

## Interpretation and limits

- Training costs use historical assignment associations and cannot separate employee effects from dispatcher selection.
- Test-cell estimates are an offline evaluation oracle and use outcomes across the test period. They never enter planning but are not prospective causal validation.
- Even strict mode uses final-record type/priority and known batch demand, so it is a stronger audit scenario, not a fully deployable online policy.
- Reproduction mode uses hindsight weekly staffing, completed-ticket demand, and creation-date fitting. Strict mode uses all category arrivals, prior-eight-week observed staffing, calendar-zero capacity and outcomes resolved before cutoff.
- Strict validation scoring uses only outcomes resolved before the test-period start. Missing assignment timestamps are excluded from prior rosters. Historical unknown assignees are not actual operational backlog and prevent fair coverage comparisons.
- Calendar durations include waiting, nights and weekends. Shifted geometric hours are not mean expected hours or labor hours.
- Fairness can add backlog. Compare backlog/coverage first; assigned-only service proxies alone can mislead.
- Cross-training uses assumed new-pair costs and a restricted candidate set, planned on eight representative training-period weeks. Its savings are penalized log-objective units, not hours.
- Cross-training with the stipulated central new-pair costs saves 0.000 objective units at the tested budgets. Zero value is a valid result, not evidence that training never helps.
- The validated uncertainty penalty is z=0.5. If z=0, M2 is exactly the tuned-shrinkage ablation; any improvement over naive M1 must not be attributed to robustness.
- Paired bootstrap intervals resample weeks independently and hold cost estimates fixed. They do not account for time dependence, estimation uncertainty or confounding.

## Artifacts

comparison.csv: all method metrics; weekly_metrics.csv: weeks and seeds; validation.csv: complete tuning grid; metadata.json: settings, source hashes and audit; routing_counts.csv: robust weekly allocation; ticket_assignments.csv: local ticket mapping (gitignored); estimates.csv: shrunk costs/support; cell_stability.csv; capacity_duals.csv; sensitivity.csv; cross_training.csv; workflow_transitions.csv. PNG/SVG figures share descriptive names.

## AI disclosure

This implementation and accompanying documentation were generated with assistance from OpenAI Codex. Team members must independently understand and verify the code, results and assumptions, and check the complete course policy before submission. This is a project implementation, not an assertion that it satisfies an unprovided academic AI policy.
