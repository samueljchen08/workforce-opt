# Robust ticket routing implementation design

Status: approved by the user on November 13, 2026; implemented and verified locally. GitHub publication is tracked separately.

## Intent

Implement the supplied 6.C571 plan as a reproducible Python research repository for Sam Chen, Christina Ding and Leo Sun. The deliverable is code, notebooks, tests, regenerable results and methodological documentation. Actual report/slides/video creation is outside the immediate coding request.

## Approaches

1. Literal reproduction: closest to the plan but retains ambiguous backlog scoring, hindsight availability and cutoff leakage.
2. Corrected full pipeline (recommended): implement M1–M4 and E1–E9, retain clearly labeled reproduction settings, and add honest evaluation defaults and diagnostics from the assessment.
3. M1/M2 core only: fastest route to a minimum viable project, with extensions deferred.

The user has been asked to choose scope and repository destination. Recommended implementation uses Python, pandas, NumPy, SciPy HiGHS, matplotlib and Jupyter; no proprietary solver or hosted application.

## Data

Read raw Mendeley v3 CSVs from gitignored data/raw. Support singular/plural snapshot filenames. Validate required columns against the real dictionary, identifier uniqueness and turn-1 join cardinality. Parse mixed UTC timestamps. Save filter counts, unmatched IDs, exclusions and input checksums. Never fabricate missing data or present synthetic runs as empirical results.

Implement the four categories, log1p calendar-hour outcome, Monday week labels, prescribed train/validation/test boundaries and minimum weekly volume. Offer strict as-of fitting and explicit retrospective creation-date reproduction. Distinguish known staffing from hindsight rosters. Persist configuration with every run.

Separate arrival workload from the resolved-outcome estimation cohort: estimate staffing/count capacity from all observable arrivals where available, rather than silently treating Done tickets as all demand. Preserve the plan's resolved-only cohort as a labeled reproduction mode. Exclude boundary-crossing partial calendar weeks from primary weekly comparisons and record their excluded counts; retain them only in explicitly labeled sensitivity results.

## Estimation

Fit shrinkage means, pooled standard deviation, uncertainty penalties, eligibility and integer ticket capacities from fitting data only. Include empty weeks in an explicit capacity-calendar convention. Tune k and z on validation, refit once, then evaluate test. Test-cell costs are an evaluation oracle only, never optimizer input. Export pair-support and fallback diagnostics.

Choose validation settings by minimum aggregate backlog first, then common penalized log-score with one fixed evaluation penalty across candidates. Assigned-only hours never determine selection when coverage differs. Strict as-of mode must not use held-out resolutions to infer historical eligibility or roster membership.

## Optimization and baselines

M1/M2 share a sparse transportation LP with backlog and dual extraction. M3 modifies integer capacities. Provide the plan's fixed-M objective and a two-stage maximum-coverage option. Convert integer count solutions to ticket assignments deterministically by creation time and ID; verify constraints and integrality before conversion.

M4 uses scipy.optimize.milp with shared binary training decisions and per-week routing variables. Choose bounded candidate pairs using training utilization and duals only. Apply the stipulated new-pair cost assumption explicitly. Report solver status, incumbent, optimality gap, time limit and candidate restriction; never report a timeout incumbent as proven optimal.

Baselines: actual historical routing, seeded historical proportions, round-robin and cost-greedy. Automated baselines enforce identical operational constraints. Historical routing remains an observational reference with feasibility violations reported. Average randomized performance over 20 seeds without treating fractional mean counts as executable routes.

## Evaluation and outputs

Save weekly and aggregated CSVs, plots, run metadata and actual tuning choices. Separate backlog, penalized log-cost, assigned-only shifted geometric hours, coverage, utilization, workload shares and Gini. Include zero-load roster members in workload metrics. Compare methods on identical weeks; use paired weekly uncertainty summaries. Implement E1–E9 with data-dependent outputs, and never hardcode the preliminary 33 weeks or appendix performance values.

Notebook 01 inspects real data, filtering and workflow structure. Notebook 02 runs the experiment pipeline and regenerates figures/tables. Also provide a command-line entry point for reproducibility without notebooks. A tiny synthetic smoke mode verifies mechanics and is conspicuously labeled synthetic.

## Verification

Hand-solvable transportation instance; demand conservation; eligibility and capacity compliance; integral solutions; all-backlog and empty-roster cases; robust costs; fairness rounding; finite-difference dual behavior; M4 tiny-instance exhaustive comparison; data join and date-boundary fixtures; no fitting on post-cutoff outcomes; deterministic seeds; baseline accounting; notebook/CLI smoke run. Raw-data end-to-end verification requires obtaining the actual CSVs.

## Publication

Use an isolated new local repository, never the home directory as a repository. Exclude raw data, credentials, local environments and generated bulk outputs. Default proposed destination is a new private samueljchen08/workforce-opt repository, subject to the user's destination response. Connector authentication works but exposes only unrelated repositories and has no repository-creation tool. Terminal gh authentication failed. Do not request passwords or tokens in chat; use normal local GitHub login if necessary.

## Remaining external dependencies

Fresh builder review: design is buildable; arrival/outcome cohort separation, validation selection and partial-week handling were identified and specified above. No further design decisions were referred to the user.

Actual CSV retrieval and dictionary inspection; an authorized writable GitHub destination; working publication authentication. Code can proceed after design approval independently of GitHub login, but empirical conclusions cannot be verified without raw data.
