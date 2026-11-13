# Robust ticket routing implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce and critically evaluate the full 6.C571 routing plan on the local real dataset.
**Architecture:** Separate ingestion, estimates, solvers, baseline routing, evaluation and experiment orchestration. Preserve a labeled plan-reproduction mode alongside corrected operational scenarios.
**Tech Stack:** Python, NumPy, pandas, SciPy HiGHS, matplotlib, Jupyter; standard-library unittest.
**Spec:** ../specs/2026-11-13-ticket-routing-design.md

## Global constraints

- No raw data or credentials committed. Seed 571. Test outcomes never used to plan routes.
- Report missing support, backlog and hindsight assumptions explicitly.
- Do not hardcode preliminary results; produce real-data outputs and reproducible commands.

## Review focus

- Missing or duplicate turn-1 records: audited joins, duplicate rejection.
- Partial weeks and post-cutoff outcomes: explicit filters and diagnostics.
- Empty rosters and unsupported categories: backlog with conserved demand.
- Different coverage: backlog-first comparisons, no exponentiated penalty.
- MILP time limits: preserve status/gap and never assert optimality without proof.

### Task 1: Data and estimates

Files: src/data.py, src/estimates.py, tests/test_data.py, tests/test_estimates.py.
Interfaces: load_data(raw) -> (arrivals, outcomes, audit); fit(outcomes, arrivals, start, end, k, n_min, strict, capacity_mode) -> Estimates; weeks(frame,start,end,min_tickets,complete) -> weekly frames.
- [ ] Write boundary, join, category, shrinkage and zero-week capacity tests; run them failing.
- [ ] Implement D1–D7, hashes, strict cutoff, shrinkage, eligibility and capacity.
- [ ] Run unittest discovery and commit.

### Task 2: Models and baselines

Files: src/models.py, src/baselines.py, tests/test_models.py, tests/test_baselines.py.
Interfaces: route_lp(cost,demand,capacity,backlog_first) -> Routing; cross_train_mip(costs,demands,capacities,candidates,budget) -> CrossTraining; baseline(tickets,est,roster,method,seed) -> Routing.
- [ ] Write hand-calculated optimality, integer, empty, robust, fairness, dual and exhaustive cross-training tests; run failing.
- [ ] Implement sparse transportation LP, M4 MILP and capacity-aware baselines.
- [ ] Run full tests and commit.

### Task 3: Evaluation and experiments

Files: src/evaluate.py, src/experiments.py, src/cli.py, tests/test_evaluate.py, tests/test_experiments.py.
Interfaces: score(routing,oracle) -> metrics; run(raw,out,mode) -> saved artifacts.
- [ ] Write backlog scoring, deterministic random routing and end-to-end toy tests; run failing.
- [ ] Implement validation grid, paired-week metrics, E1–E9, fairness, support, sensitivity, provenance and representative-week M4.
- [ ] Run full suite and real-data pipeline, inspect numeric outputs and figures.

### Task 4: Delivery

Files: notebooks/01_data.ipynb, notebooks/02_results.ipynb, README.md, report/results.md, requirements.txt, .github/workflows/tests.yml.
- [ ] Provide notebooks that call the same CLI pipeline, setup documentation and generated results explanation.
- [ ] Independently review implementation, correct material defects, rerun tests and experiments as necessary.
- [ ] Commit verified source and compact results; publish to personal GitHub once authentication/network allow it.
