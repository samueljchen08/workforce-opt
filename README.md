# Robust ticket routing for a help desk

6.C571 term project · Sam Chen, Christina Ding, Leo Sun

Reproducible weekly ticket routing with a transportation LP, interval-robust costs, workload fairness caps and a cross-training MILP. This repository includes actual-data results and explicitly audits the assumptions behind the original plan.

## Start with the results

- [Original plan](docs/PLAN.md), [methodological assessment](docs/project-assessment.md)
- [Plan reproduction results](results/reproduction/RESULTS.md)
- [Stricter historical scenario](results/strict/RESULTS.md)
- [Data notebook](notebooks/01_data.ipynb), [experiment notebook](notebooks/02_results.ipynb)
- `results/index.html`: locally generated visual results overview.

Read backlog and coverage alongside every service-time comparison. The outcome is **log(1 + calendar resolution hours)**. Exponentiated averages are shifted geometric-time proxies, not expected labor or resolution hours. Employee effects and reassignment benefits are not causally identified.

## Install and run

Python 3.11 or newer. No commercial solver or API key required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.cli --mode both
python scripts/execute_notebooks.py
```

In this workspace the verified interpreter is `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`. NumPy, pandas, SciPy and matplotlib are already available there. The lightweight notebook runner executes the included plain-Python cells and records stdout without requiring Jupyter. Install Jupyter separately for interactive editing if desired.

## Dataset

[Help Desk Tickets v3, Mohammad Abdellatif, Mendeley Data](https://data.mendeley.com/datasets/btm76zndnt/3), DOI `10.17632/btm76zndnt.3`, CC BY 4.0. Download these files to `data/raw/`:

- `issues.csv`
- `issues_snapshot.csv` (plural `issues_snapshots.csv` also accepted)
- `issues_change_history.csv`
- `FEATURES.md` and `EXAMPLE.md` are useful reference documents.

Raw data is excluded from Git. Input checksums, package versions, filtering counts and settings are saved in each `metadata.json`. The local data dictionary describes 2016–2023 records; the landing page describes a broader 2007–2023 collection. This project uses 2021–February 2023 only.

## Models and comparisons

- **M1:** count-based transportation LP with employee ticket capacities and backlog. Integer demand/capacity produces integral vertex solutions, checked before mapping counts back to tickets.
- **M2:** replace cost by `shrunk_mean + z * pooled_sigma / sqrt(n+k)`. Tune `k ∈ {5,20,50}` and `z ∈ {0,.5,1,1.5,2}` using validation only. This is an uncertainty heuristic, not a calibrated confidence bound.
- **M3:** cap load at `min(K_i, ceil(alpha * demand / staff_count))` for alpha 1.25, 1.5, 2 and 3, plus uncapped.
- **M4:** binary cross-training decisions shared over eight representative training-period weeks; budgets 0, 1, 2, 3 and 5; three assumed new-pair cost scenarios. Status and optimality gap are saved.
- **Baselines:** historical reference, capacity-aware round-robin, cost greedy and historical proportions over 20 seeds; tuned-shrinkage ablation isolates the value of the uncertainty term.

All E1–E9 experiments are implemented. See `results/<mode>/` for workflow transitions, cost stability, fractionality, planning/held-out scores, comparisons, duals and finite differences, capacity/eligibility sensitivity, fairness and cross-training outputs.

## Two evaluation scenarios

`reproduction` follows the plan's Done-ticket cohort, nonzero-week capacity quantile, hindsight weekly employee roster, creation-date split and fixed-M penalty. This is a retrospective batch experiment.

`strict` uses all category arrivals, training outcomes resolved before fitting cutoff, validation scoring outcomes resolved before July 1, prior-eight-week staffing observed before each week, zero-filled calendar-week capacities, complete calendar weeks and backlog-first optimization. Missing assignment timestamps are excluded from staffing inference. Final-record category/priority and known batch demand remain assumptions: this is **not** a fully prospective online test.

Both use retrospective test-period employee/category averages as a common scoring oracle. They are estimates for observed assignment groups, not outcomes for the same tickets under alternative routing. Common-support and fallback counts are provided. Historical missing first-assignee labels are unknown observations, not verified backlog; historical assignments can violate inferred operational constraints.

Selection minimizes validation backlog first, then a fixed common penalized log-score. Service proxy hours are calculated only on assigned tickets and never include exponentiated backlog penalties. Confidence intervals resample paired weeks conditional on fitted estimates; they omit temporal dependence and cost-estimation uncertainty.

## Repository layout

`src/data.py`: audited loading, categories, cohorts, weeks, workflow transitions. `estimates.py`: shrinkage, eligibility, capacities, evaluation oracle. `models.py`: LP, fairness, ticket mapping and MILP. `baselines.py`: dispatch rules. `evaluate.py`: coverage-aware scoring and summaries. `experiments.py`: validation and E1–E9. `cli.py`: command-line entry point. `tests/`: independent tiny fixtures and regression checks.

`ticket_assignments.csv` is generated locally but excluded from Git. Compact aggregates, plots and checksums are committed. To publish after signing in, run `scripts/publish_github.sh`; it creates a private repository if absent and refuses to overwrite a different origin.

## Course scope and disclosure

The supplied guidelines support this as a problem-driven project. Verify the deadline: December 7, 2026 is Monday, despite the plan calling it Sunday. This repository is the research implementation and results package, not a finished submission-ready report, narrated video or presentation deck.

Code and documentation were generated with OpenAI Codex assistance. Team members must independently review, understand and reproduce them and verify the course's complete AI-use policy before submission. Preserve an accurate disclosure; do not submit preliminary appendix values as reproduced facts.
