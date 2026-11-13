# 6.C571 Term Project — Full Plan
## Robust Ticket Routing for a Help Desk

**Team:** Sam Chen, Christina Ding, Leo Sun
**Deliverables due:** Sunday, Dec 7, 2026: technical report (5–10 pp), presentation slides, video (≤5 min)
**In-class presentations:** Dec 9–10 · **Second midterm:** Dec 2/3 (plan around it)

---

## Contents
1. The project in one paragraph
2. Requirements checklist (syllabus + interim-report promises)
3. Why the scope changed
4. Data pipeline, step by step
5. Notation and cost estimation
6. Optimization models (M1–M4)
7. Evaluation protocol
8. Experiments (E1–E9)
9. Code layout
10. Schedule and roles
11. Report outline
12. Slides and video
13. Risks and fallbacks
14. Limitations
15. Appendix: preliminary numbers from the Nov 12 feasibility check

---

## 1. The project in one paragraph

A help-desk manager must decide, every week, **which employee each new ticket is routed to first**. Using
about 11,000 real tickets (2021–2023) from an international software company, we estimate how long tickets take to resolve
depending on who receives them. We then route each week's tickets with a linear program that minimizes total
expected resolution time, subject to each employee's capacity. The naive optimizer looks far better than the
real dispatcher on the data it was fit to, but loses to the dispatcher on later weeks: the **optimizer's curse**.
A **robust** version that penalizes uncertain estimates closes that gap, roughly matches the experienced
dispatcher using only ticket category, and clearly beats rule-based routing. Finally, LP duals and a small
integer program tell the manager **where extra capacity helps most, who to cross-train, and what fairness
costs.**

---

## 2. Requirements checklist

### Syllabus requirements
| Requirement | Where it is met |
|---|---|
| Clearly identified decision problem | §1, §6: weekly first-assignee routing |
| Uses optimization methods | M1 LP / transportation problem, total unimodularity, duality; M2 robust optimization; M3 constrained LP; M4 mixed-integer program |
| Reports computational results | E1–E9 on 33 out-of-sample weeks; solve times |
| Benefit of optimization vs benchmarks | Historical, round-robin, greedy, historical-proportions (§7) |
| Recommendations to a decision-maker | Report Section 8: routing table, capacity, cross-training, fairness |
| Slides, technical report, video | §11, §12 |

### Promises made in the interim report (keep them, or explain the change)
| Interim promise | Status in this plan |
|---|---|
| Workflow map of task types and transitions | Kept as one descriptive figure (E1) |
| GAP model minimizing total labor hours | **Changed** → routing LP minimizing expected resolution time (§3) |
| Eligibility constraints e_ij | Kept (eligible pairs E) |
| Capacity H_i | Kept, as tickets per week K_i |
| Baselines: historical proportions, round-robin | Kept, plus historical and greedy |
| Metrics: objective, workload balance, utilization, solve time | Kept (§7.4) |
| Workload-share cap variant | Kept (M3) |
| Sensitivity to processing times and capacity | Kept and strengthened: robust model (M2) and capacity sweep (E7) |
| Where an extra hour of capacity saves the most | Kept: capacity duals (E6) |
| Which employees to cross-train | Kept and made rigorous: M4 integer program |
| Makespan / precedence extension | **Dropped** (data does not support it; §3) |

---

## 3. Why the scope changed

The interim report planned to minimize **total labor hours**. When we examined the data:

- Every time field (`wf_*`, `started`/`ended`) is **calendar time**, not working time. It includes nights,
  weekends, and time waiting on the customer. There is no measure of how long anyone actually worked on a ticket.
- 76% of tickets pass through several people. Many "first assignees" are dispatchers who hand the
  ticket on.

What the data does support is the **end-to-end resolution time of a ticket given who it was first routed to**,
and that relationship is reasonably stable over time (§15). So:

- objective: total labor hours → **total expected (log) resolution time**
- capacity: hours per week → **tickets per week**
- decision: assign individual tasks → **route this week's tickets by category**

State this in the report as a data-driven modeling decision. It shows judgment, not failure.

---

## 4. Data pipeline, step by step

Dataset: Help Desk Tickets v3, Mendeley Data, https://data.mendeley.com/datasets/btm76zndnt/3
Local copy: `data/raw/` (do not commit; about 68 MB).

| File | Used for |
|---|---|
| `issues.csv` | type, priority, created, resolution date, resolution |
| `issues_snapshot.csv` | turn-1 assignee (= routing decision) |
| `issues_change_history.csv` | workflow-map figure only (E1) |

**Parsing tips:** read with `low_memory=False`. Parse timestamps with
`pd.to_datetime(..., utc=True, errors="coerce", format="mixed")`, because the formats are mixed. IDs load as floats;
drop NaN and cast to integers before joining. Every join on `id` matches 100%.

### Steps (with sanity targets to check against)

| # | Step | Expected result |
|---|---|---|
| D1 | Keep issues created 2021-01-01 to 2023-02-28 | about 13,800 issues |
| D2 | Keep `issue_resolution == "Done"` (drop Won't Do, Duplicate, Cannot Reproduce, unresolved) | about 12,500 |
| D3 | Map to 4 categories (table below); drop other types | about 11,100 |
| D4 | Attach first assignee = `issue_assignee` where `turn == 1` in snapshot | about 92% matched |
| D5 | Outcome y = log(1 + hours from created to resolution); drop negatives | about 11,000 usable |
| D6 | Week label = calendar week (Mon–Sun) of `issue_created` | 33 test weeks with ≥50 tickets |
| D7 | Split: **train** 2021-01-01 to 2022-06-30, **test** 2022-07-01 to 2023-02-28 | about 7,000 / about 4,000 |

### Categories
| Category | Rule | Approx. tickets |
|---|---|---|
| T-normal | Ticket with priority Lowest, Low, Medium | 4,800 |
| T-high | Ticket with priority High | 3,000 |
| T-urgent | Ticket with priority Highest, Blocker | 2,000 |
| Service | Service or HD Service, any priority | 1,300 |

Deployment (253 tickets, 3 employees) and non-work types (Project, Vacation, Epic, Story, Sub-task) are dropped.
Low priority and service sub-priorities were too sparse to stand alone.

### Why log hours
Raw resolution hours are extremely skewed (median about 146 h, 99th percentile about 6,600 h). Averaging logs means a
handful of year-long tickets can't dominate. Report results back in hours as **exp(mean y) − 1**, the geometric-mean
resolution time.

---

## 5. Notation and cost estimation

### Sets and data (for one week)
| Symbol | Meaning |
|---|---|
| I | employees available this week: those who received ≥1 first assignment this week and are eligible for ≥1 category |
| C | categories {T-normal, T-high, T-urgent, Service} |
| d_c | number of category-c tickets created this week (integer) |
| K_i | ticket capacity of employee i this week (integer) |
| E ⊆ I × C | eligible pairs: i had ≥ n_min = 5 category-c tickets in training |
| ĉ_ic | estimated expected log-resolution time when i receives a category-c ticket |
| s_ic | standard error of ĉ_ic |

### Cost estimate with shrinkage
Let n_ic = number of training tickets first-assigned to i in category c, Y_ic = the sum of their y values,
ȳ_c = training mean of y in category c, and σ = the training standard deviation of y.

```
ĉ_ic = ( Y_ic + k · ȳ_c ) / ( n_ic + k )
s_ic = σ / sqrt( n_ic + k )
```

Read k as "k imaginary tickets at the category average." Small samples get pulled toward the average; large samples
keep their own mean. k ∈ {5, 20, 50} is tuned on validation (§7.2).

### Capacity
```
K_i = max( 2, floor( 90th percentile of i's weekly first-assignment counts in training ) )
```
K_i must be an integer (see M1). Employees with no training history are not in I.

---

## 6. Optimization models

### M1 — Routing LP (a transportation problem)

Decision variables: x_ic ≥ 0, the number of category-c tickets routed to employee i, and u_c ≥ 0, the category-c
tickets left in the backlog this week.

```
minimize    Σ_{(i,c) ∈ E}  ĉ_ic · x_ic   +   M · Σ_c u_c

subject to  Σ_{i : (i,c) ∈ E} x_ic + u_c  =  d_c        for every c ∈ C     (route every ticket)
            Σ_{c : (i,c) ∈ E} x_ic       ≤  K_i        for every i ∈ I     (capacity)
            x_ic ≥ 0,  u_c ≥ 0
```

- **M** is a large penalty (e.g., 100; every ĉ is below about 10). Backlog is used only when capacity runs out. One
  test week (starting 2022-11-28) needs it.
- **Integrality without an integer program.** This is a transportation problem: every variable appears in exactly one
  demand row and at most one capacity row. Its constraint matrix is totally unimodular, so with integer d_c and K_i every vertex
  of the feasible region is integer, and the LP solution is an integer routing. Check it: max |x − round(x)| should be 0. (In the
  feasibility check, non-integer K_i produced half-integer solutions until K_i was floored. Worth a
  sentence.)
- **Duals.** Let λ_i ≤ 0 be the dual price on employee i's capacity constraint. Then |λ_i| = how much total expected
  log-time falls if i could take one more ticket this week. Average |λ_i| over weeks to rank employees.
- **Turning counts into tickets:** tickets in the same category are interchangeable in the model. Assign them in
  creation order to employees until each x_ic is used up.

### M2 — Robust routing (main result)

The ĉ_ic are estimates. Suppose each true cost lies anywhere in the interval

```
c_ic ∈ [ ĉ_ic − z·s_ic ,  ĉ_ic + z·s_ic ]
```

and route to minimize the **worst-case** total cost:

```
minimize over x   max over c in the box   Σ c_ic · x_ic   (+ backlog term, same constraints as M1)
```

Because x ≥ 0, the inner maximum is reached by setting every c_ic to its upper bound. So the robust counterpart is
just M1 with costs

```
c̃_ic = ĉ_ic + z · s_ic
```

It is still a transportation LP: integer solutions and duals both carry over. z = 0 is the naive model; larger z
favors employees with a long, proven track record. Explain this derivation in your own words in the report.

*Stretch, only if time allows:* the Bertsimas–Sim budget-of-uncertainty version (at most Γ costs at their worst), which
is less conservative. It is still an LP after taking the dual of the inner problem.

### M3 — Fairness cap

Replace K_i by

```
K_i(α) = min( K_i ,  ceil( α · Σ_c d_c / |I| ) )
```

so no one receives more than α times an equal share. Sweep α ∈ {1.25, 1.5, 2, 3, ∞}. Lower α = fairer,
but costlier, and backlog appears if it is too tight. This traces the **cost-of-fairness frontier**.

### M4 — Cross-training (mixed-integer program)

Decide which currently ineligible pairs (i, c) to train, given a budget of B pairs. Plan over a set W of
representative weeks (use the **training-period** weeks, to avoid tuning on test):

```
variables   t_ic ∈ {0,1}   for (i,c) ∉ E      (train i on c)
            x_icw ≥ 0, u_cw ≥ 0   for each week w ∈ W

minimize    Σ_w [ Σ_{i,c} c̃_ic · x_icw  +  M · Σ_c u_cw ]

subject to  Σ_i x_icw + u_cw = d_cw              for every c, w
            Σ_c x_icw ≤ K_i                      for every i, w
            x_icw ≤ K_i · t_ic                   for every (i,c) ∉ E, w   (usable only if trained)
            Σ t_ic ≤ B
```

- **Cost of a newly trained pair (assumption; state it):** c̃_ic = ȳ_c + δ_i + z·σ, where δ_i is employee i's
  average deviation from the category means across the categories they already handle. A good employee is expected to stay good
  in a new category, with uncertainty.
- Solve for B ∈ {0, 1, 2, 3, 5}. Report the objective drop for each extra trained pair (diminishing returns) and which pairs are
  picked.
- Restrict candidates to employees with capacity left over on average (high K_i, small |λ_i|), to keep the program small.
- This is the project's one true integer program. The binary t breaks total unimodularity, so mention that the
  LP relaxation can be fractional here.

---

## 7. Evaluation protocol

### 7.1 The scoring rule (identical for every method)
Fit **test-period** costs with the same shrinkage formula (k = 5, using test data only); call them c^test_ic. For a pair
with no test tickets, use the test category mean. For a week's routing:

```
score(week) = (1 / #tickets) · Σ_tickets c^test_{assignee, category}
hours(week) = exp(score) − 1
```

Never score a method with the costs it planned with. That is exactly the in-sample illusion the project exposes.

### 7.2 Choosing k and z without touching the test set
- Validation: fit costs on 2021-01-01 to 2021-12-31 and score on the weeks of 2022-01-01 to 2022-06-30.
- Grid: k ∈ {5, 20, 50} × z ∈ {0, 0.5, 1, 1.5, 2}. Pick the best pair.
- Refit on the full training period with that (k, z) and run on the test weeks once.

### 7.3 Methods compared
| Method | Rule |
|---|---|
| Historical | the actual turn-1 assignee (what the dispatcher did) |
| Historical proportions | random: route a category-c ticket to i with probability proportional to i's training count in c (average over 20 seeds) |
| Round-robin | cycle through eligible employees within each category |
| Greedy | tickets in creation order; each goes to the lowest-ĉ eligible employee with remaining capacity |
| M1 naive | z = 0, k = 5 |
| M2 robust | validated (k, z) |
| M2 + M3 | robust with fairness cap α |

### 7.4 Metrics per method
- Geometric-mean resolution hours (mean over weeks, plus the 10th–90th percentile range across weeks)
- % improvement vs historical and vs round-robin
- Weeks (out of 33) beating historical
- Workload balance: max employee share and Gini of tickets per employee
- Capacity utilization: Σ x / Σ K
- Backlogged tickets
- Solve time per week

### 7.5 Fairness note on "historical"
The dispatcher sees ticket content and urgency; our model sees only the category. Historical is therefore a
**strong** benchmark, and matching it with category-only information is a meaningful result. Round-robin
and greedy represent what an automated rule would do without optimization.

---

## 8. Experiments

| ID | Question | Output |
|---|---|---|
| E1 | What does the workflow look like? | Figure: status-transition map from change history (top transitions only). Descriptive. |
| E2 | Do employees differ, and is it stable? | Figure: spread of ĉ_ic within each category. Number: train vs test correlation of cell means |
| E3 | Does M1 give integer solutions? | Max fractionality across all weeks (expect 0) |
| E4 | **Optimizer's curse** | Figure: in-sample vs out-of-sample hours for M1, next to historical |
| E5 | **Main comparison** | Table: all methods × metrics (§7.4) on 33 test weeks |
| E6 | Where does capacity matter? | Table: top-10 employees by average capacity dual; check one by re-solving with K_i + 1 |
| E7 | Sensitivity to capacity and eligibility | Hours vs capacity scale ∈ {0.5, 0.75, 1, 1.25, 1.5} × K_i; hours vs n_min ∈ {3, 5, 10} |
| E8 | Cost of fairness | Figure: hours vs max-share (or Gini) as α varies |
| E9 | Cross-training value | Figure: hours saved vs budget B; table of the chosen pairs |
| — | Robustness tuning | Figure: validation hours vs z for each k (§7.2) |

**Minimum viable project = E2, E3, E4, E5 with M1 + M2.** Everything else adds depth.

---

## 9. Code layout

Keep it small and readable. One person owns each file; everyone reads all of them.

```
workforce-opt/
  data/raw/              CSVs (gitignored)
  src/
    data.py              load + filter (D1–D7), categories, first assignee, y, week labels, splits
    estimates.py         shrunk costs ĉ, standard errors s, eligibility E, capacities K
    models.py            route_lp(costs, demand, capacity) → x, duals   (serves M1, M2, M3)
                         cross_train_mip(...) → chosen pairs, objective  (M4)
    baselines.py         historical, historical_proportions, round_robin, greedy
    evaluate.py          score a routing; loop over weeks; collect metrics
  notebooks/
    01_data.ipynb        D1–D7 sanity targets, E1, E2
    02_results.ipynb     E3–E9 and every report table/figure
  results/               saved CSVs + PNGs (regenerable)
  report/                LaTeX/Docs for the report
```

**Tools**

- `scipy.optimize.linprog(method="highs")` for M1–M3: free, fast, and returns capacity duals in
  `res.ineqlin.marginals`.
- `scipy.optimize.milp` or PuLP for M4. Gurobi (free academic license) is optional.
- pandas, numpy, matplotlib. Fix one random seed (e.g., 571).

**Habits**

- Every report number comes from re-running `02_results.ipynb` top to bottom.
- Check the code on a toy instance (2 employees, 2 categories) where you can work out the answer by hand before running it on real weeks.
- Commit to the team repo regularly. Don't commit raw data.

---

## 10. Schedule and roles

**Roles** (assign names at the first meeting):

- **Data & Estimation:** `data.py`, `estimates.py`, notebook 01, report Sections 2–3 and Limitations
- **Models:** `models.py` (M1–M4), report Section 4, the robust-counterpart explanation
- **Evaluation & Communication:** `baselines.py`, `evaluate.py`, notebook 02 figures, report Sections 5–8,
  slides/video lead

| Dates | Tasks | Done when |
|---|---|---|
| **Thu Nov 13 – Sun Nov 16** | Team meeting: agree on scope, assign roles, set up repo. Data: D1–D7. | Notebook 01 reproduces the §4 sanity targets |
| **Mon Nov 17 – Thu Nov 20** | Estimates (ĉ, s, E, K). Baselines + scoring. Models: M1 on toy instance. | Historical and round-robin scored on all test weeks; M1 correct on toy |
| **Fri Nov 21 – Mon Nov 24** | M1 on all weeks; E3, E4, E5 (partial). Validation grid for M2. | **Minimum viable project done**: M1 vs baselines table exists |
| **Tue Nov 25 – Sun Nov 30** (Thanksgiving Nov 26) | M2 final, M3 frontier, M4, E6–E9. Start writing Sections 2–4. | All figures and tables generated |
| **Mon Dec 1** | **Freeze results.** No new experiments after this. | Notebook 02 runs end-to-end |
| **Tue Dec 2 – Wed Dec 3** | Midterm. Light editing only. | — |
| **Thu Dec 4 – Sat Dec 6** | Finish report; slides; record video. Cross-read each other's sections. | Drafts complete by Sat night |
| **Sun Dec 7** | Final proofread; submit report, slides, video. | Submitted |
| **Dec 9–10** | Present in class. Rehearse twice beforehand. | — |

Weekly check-ins: Sun Nov 16, Thu Nov 20, Mon Nov 24, Sun Nov 30, Thu Dec 4.

---

## 11. Report outline (target 8 pages + appendix)

| § | Section | ~Pages | Contents |
|---|---|---|---|
| 1 | Introduction & decision problem | 0.75 | Who the decision-maker is, the weekly decision, why it matters, summary of findings |
| 2 | Data | 1 | Dataset, filters, categories, outcome, split; E1 workflow figure; the scope change (§3) |
| 3 | Cost estimation | 0.75 | Shrinkage formula, standard errors, eligibility, capacity; E2 figure |
| 4 | Models | 1.5 | M1 (+ TU integrality, duals), M2 (robust-counterpart derivation), M3, M4 |
| 5 | Experimental setup | 0.5 | Scoring rule, validation for (k, z), baselines, metrics |
| 6 | Results | 1.5 | E4 optimizer's-curse figure, E5 main table, E3 integrality, solve times |
| 7 | Sensitivity, fairness, capacity | 1 | E6 duals, E7 sensitivity, E8 fairness frontier |
| 8 | Recommendations | 0.5 | Manager-facing: routing table, capacity, cross-training (E9), fairness cost |
| 9 | Limitations & future work | 0.5 | §14; future: ticket-content features, live A/B test, queueing model |
| 10 | AI-use disclosure | short | Required by the syllabus if any AI tools were used |
| App. | Extra tables | — | Full routing table, validation grid, data dictionary |

**Writing rules:** every number traces to a notebook cell; every figure has a one-sentence takeaway caption;
lead each section with its conclusion.

---

## 12. Slides and video

### Slides (about 9 slides, for a 5-minute talk)
1. Title + one-line result
2. The decision: a dispatcher routing about 110 tickets/week to about 21 people
3. Data and what it can and can't tell us (the scope change in one picture)
4. The model: the routing LP in one box; integrality + duals in two bullets
5. **The optimizer's curse:** in-sample vs out-of-sample bars (the hook)
6. Robust fix: the cost + z·se idea; results table vs baselines
7. Fairness frontier + capacity duals
8. Cross-training: hours saved vs B
9. Recommendations for the manager + limitations

### Video (≤ 5 minutes)
Narrate the slides; each teammate presents their own part (about 1.5 min each). Script it, time it, record it in one
session. Leave 15 s of buffer under the limit.

---

## 13. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Robust model still loses to historical | That's fine: the main claims are (a) the optimizer's-curse lesson, (b) it beats round-robin/greedy, (c) recommendations. Frame it as "matches an expert who has more information." |
| Results differ from the preliminary numbers | Expected. Report your own numbers; check data filters against the §4 targets first |
| M4 too slow or too large | Fewer weeks in W (e.g., 8 representative weeks), fewer candidate pairs, or solve the LP relaxation and round |
| Some weeks infeasible | The backlog variable u_c guarantees feasibility; report the backlog counts |
| Teammate time crunch near the midterm | Results freeze on Dec 1; the minimum viable project is done by Nov 24 |
| Solver/license issues | Everything runs on SciPy's HiGHS; Gurobi is optional |

---

## 14. Limitations

- Outcomes are observed only for the employee who actually received each ticket. Improvements are model-based
  estimates, not measured results.
- The dispatcher uses ticket content we don't see. Category-level routing is a simplification.
- About 5% of tickets are unresolved and excluded, which biases toward faster tickets.
- Resolution times trend downward over time (the test period is faster), so comparisons are only valid within the same period.
- Capacities are inferred from history (90th percentile), not actual staffing.
- Cross-training costs for new pairs rest on an assumption (§6, M4).
- Wall-clock time is a proxy for service quality, not for effort.

---

## 15. Appendix: preliminary numbers (Nov 12 feasibility check)

These come from a quick preliminary check to confirm the project is viable. **Do not cite them.** Reproduce
them with your own code; your numbers should land in the same neighborhood.

| Check | Result |
|---|---|
| Usable Done tickets (2021-01 to 2023-02, 4 categories, with first assignee) | about 11,000 |
| Train / test tickets | about 7,000 / about 4,000 |
| Employees with ≥5 training tickets: T-normal / T-high / T-urgent / Service | 82 / 38 / 33 / 40 |
| Train-vs-test correlation of cell means (100 cells with ≥5 tickets in both periods) | 0.65 (0.42 after removing category means) |
| Test weeks with ≥50 tickets | 33 (median about 117 tickets, about 21 eligible employees) |
| Per-employee weekly load (test) | median 2, 90th percentile 11 |
| M1 fractionality with integer K_i | 0 (integer, as TU predicts) |
| Geometric-mean hours, test weeks: historical | 68.5 |
| M1 naive (k=5, z=0): in-sample → out-of-sample | looks about 34% better than history → 76.1 (about 11% worse) |
| M2 robust (k=20, z=2) | 69.5 |
| Round-robin | 82.8 |
| Fairness cap α=2 with k=20, z=1 | backlog/infeasibility in about 1/3 of weeks; α must be loosened or the backlog reported |
