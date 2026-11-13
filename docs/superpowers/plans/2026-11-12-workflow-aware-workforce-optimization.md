> **SUPERSEDED (2026-11-12)** by `docs/PLAN.md`. Labor-hours scope is not supported by the data. Do not reuse the code below (course GenAI policy).

# Workflow-Aware Workforce Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python pipeline that turns the Mendeley Help Desk Tickets data into weekly ticket→employee assignment instances. It solves them with a generalized-assignment MILP (plus a workload-cap variant and a makespan/precedence extension), compares the results to historical, historical-proportion and round-robin baselines, and runs the sensitivity, shadow-price and cross-training analyses promised in the 6.C571 interim report.

**Architecture:** A linear pipeline of small modules under `src/wfo/`: load → categorize → estimate effort (t_ij) → build weekly instances → solve (MILP / LP / CP-SAT) → evaluate → experiments → figures. Each stage is a pure function over pandas/numpy objects so it can be unit-tested on tiny synthetic fixtures. `scripts/run_all.py` runs the whole thing from `config/default.yaml` and writes CSVs + figures to `results/`.

**Tech Stack:** Python 3.12 (via `uv`), pandas, numpy, OR-Tools (`pywraplp` with SCIP for MILP, GLOP for LP duals, CP-SAT for scheduling; Gurobi optional through the same `pywraplp` interface), networkx + matplotlib for the workflow map and figures, PyYAML, pytest.

**Spec:** `~/Downloads/6_C571_Final_Project.pdf` (interim report). Design decisions that fill gaps in the report are recorded in the "Design decisions" section below. Executors read both.

## Global Constraints

- Python `>=3.12,<3.13` (system Python is 3.14; OR-Tools wheels are safest on 3.12). Use `uv run` for every command.
- Raw data lives in `data/raw/` and is **never committed** (≈68 MB). It is already downloaded at `~/workforce-opt/data/raw/`.
- Read every CSV with `pd.read_csv(..., low_memory=False)` and parse every timestamp with `pd.to_datetime(x, utc=True, errors="coerce", format="mixed")`. The files mix timestamp formats, and plain parsing leaves them as strings.
- All IDs from CSVs are floats on load. Cast `id` / `issueid` to `int64` after dropping NaN.
- All solver-facing times are **hours** (floats) for the MILP and **integer minutes** for CP-SAT.
- Ineligible (employee, task) pairs are `np.inf` in planning matrices. Never use a large finite number.
- Every random draw goes through `np.random.default_rng(cfg.seed + offset)`. No global RNG.
- Effort numbers are model estimates in "assumed active hours". Figures and the report must never call them measured labor hours.
- Run `uv run pytest -q` after every task. All tests must pass before committing.

---

## What the data actually looks like (verified 2026-11-12)

These facts drive the design. They were checked against the downloaded files.

| Fact | Consequence |
|---|---|
| `issues.csv` 66,691 rows; `issues_snapshot.csv` 90,963 rows (one per assignee turn); `issues_change_history.csv` 257,508 rows (`field` ∈ {status, assignee}) | Snapshot = unit of "who did what"; history = status timeline |
| `issue_type` is 68% "Ticket"; non-work types exist (Vacation, Retrospective, Sprint Summary, Epic, Project) | Type alone is a weak task category → use **type group × priority band**; drop non-work types |
| `issue_priority` is 51% "unknown" overall but only 2 of 24,921 rows in 2021+ | Restrict to 2021-01 → 2023-03, where priority is usable |
| 2021+: 24,921 turns, 251 assignees, 94 with ≥50 turns | Big enough for per-employee estimates; ~50 active employees/week |
| `wf_*` and `wf_total_time` are **wall-clock seconds** (`wf_total_time == ended − started`) | Not effort. Need an active-time proxy |
| Per-turn `ended − started`: median 261 h, p90 60,000 h | Elapsed time is useless as t_ij directly |
| `wf_in_progress` for single-contributor 2021+ tickets: median 7.6 h, p75 136 h, 47% missing | Even "in progress" is wall-clock and heavy-tailed → use it only for **relative speed** |
| `issues_change_history.issueid` and `issues_snapshot.id` both join 100% to `issues.id` | Safe joins on int64 id |
| Snapshot `issue_assignee` is sometimes NaN | Drop those turns |

## Design decisions (fill the gaps in the interim report)

1. **Task category** `c(j)`: `ticket/{low,med,high,urgent}`, `service/{low,med,high,urgent}`, `deploy`. That gives 9 categories. Types kept: Ticket, Service, HD Service (→service), Deployment. Priority bands: Lowest/Low→low, Medium/unknown→med, High→high, Highest/Blocker→urgent.
2. **Effort model, t_ij = base_c × exp(s_ic)**:
   - `base_c` is an **assumed** active-effort hours per category, set in config (the report's "reasonable assumptions about active processing times").
   - `s_ic` is the employee's **relative speed**. It is the log-ratio of their business-hours in-progress time to the category median, with two levels of shrinkage: s_ic shrinks toward the employee's overall speed s_i, and s_i shrinks toward 0, with strength `k`. Clamp exp(s_ic) to [0.25, 4].
   - This keeps the part the data can support (who is relatively faster at what) separate from the part it can't (absolute effort).
3. **Active time per turn** = business-hours (Mon–Fri 09:00–18:00 UTC) overlap between that turn's [started, ended] window and the issue's `in_progress` intervals from the change history, clipped at 40 h.
4. **Eligibility** `e_ij = 1` iff employee i has ≥ `min_turns_eligible` (default 3) training turns in category c(j).
5. **Planning horizon** = one calendar week. Tasks = issues *created* that week with a valid category. Historical assignee = turn-1 assignee from the snapshot (the routing decision being optimized).
6. **Active employees** in week w: ≥1 turn started in [w − 14 d, w + 21 d) **and** eligible for at least one category.
7. **Capacity** `H_i` = `capacity_hours` (default 24 assumed effort-hours/week = 40 h × 0.6 focus). Task 6 includes a calibration check.
8. **Train/test split to avoid the optimizer's curse:** t_ij for *planning* is estimated on 2021-01-01 → 2022-07-01. Plans are *evaluated* with t_ij re-estimated on the test period 2022-07-01 → 2023-03-01 (falling back to the train estimate, then to `base_c × novice_multiplier` for pairs never observed). Reporting in-sample savings would overstate the gain.
9. **Feasibility:** each task may go to a dummy "unassigned/backlog" option at penalty `unassigned_penalty` (1000 h). The MILP is then always feasible, and infeasibility shows up as a reported count.
10. **Workload-cap variant** (linear): `load_i ≤ s · Σ_k load_k` with `s = m / n_active` for fair-share multipliers m ∈ {1.5, 2, 3}. Sweeping m traces the total-time vs. balance frontier.
11. **"Where an extra hour saves the most"** = absolute value of the capacity-constraint duals in the LP relaxation (GLOP), checked against a finite-difference MILP re-solve for the top employees.
12. **Cross-training value** = for the bottleneck category, make each currently ineligible employee eligible at `base_c × exp(s_i)` and record the drop in the MILP objective.
13. **Extension (makespan):** a ticket whose history includes `validation`, `under_review`, `in_review` or `pending_deployment` gets a second "verify" stage of length `verify_fraction × t` (default 0.3). The verify stage must start after the work stage, on a **different** employee (four-eyes rule). CP-SAT minimizes makespan with release times = creation time, run on daily sub-batches (≤40 tasks) so it stays tractable.

## File Structure

```
workforce-opt/
├── pyproject.toml                 # deps, pytest config
├── .gitignore                     # data/raw, results/, .venv
├── config/default.yaml            # every tunable number lives here
├── src/wfo/
│   ├── __init__.py
│   ├── config.py                  # Config dataclass + load_config()
│   ├── load.py                    # CSV readers with robust parsing
│   ├── categories.py              # type × priority → category
│   ├── effort.py                  # business hours, in-progress intervals, turn active hours, speeds, time table
│   ├── workflow_map.py            # status transition matrix, dwell times, bottleneck table, graph figure
│   ├── instance.py                # Instance dataclass, active employees, weekly instance builder
│   ├── model.py                   # GAP MILP (+share cap), LP relaxation duals
│   ├── baselines.py               # historical, historical-proportions, round-robin
│   ├── metrics.py                 # evaluate(): total hours, balance, utilization, overloads
│   ├── scheduling.py              # CP-SAT makespan with precedence + four-eyes
│   ├── experiments.py             # benchmark, sensitivity, duals, cross-training, makespan runs
│   └── plots.py                   # figures for the report
├── scripts/run_all.py             # end-to-end entry point
├── tests/
│   ├── conftest.py                # tiny synthetic DataFrames
│   ├── test_config.py
│   ├── test_categories.py
│   ├── test_effort.py
│   ├── test_workflow_map.py
│   ├── test_instance.py
│   ├── test_model.py
│   ├── test_baselines.py
│   ├── test_metrics.py
│   ├── test_scheduling.py
│   └── test_experiments.py
├── data/raw/                      # downloaded CSVs (gitignored)
├── results/                       # CSVs + figures (gitignored, regenerable)
└── report/                        # final write-up (LaTeX), pulls from results/
```

---

### Task 1: Project scaffold, config, and data loading

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config/default.yaml`, `src/wfo/__init__.py`, `src/wfo/config.py`, `src/wfo/load.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass; `load_config(path: str) -> Config`; `ts(series) -> Series[datetime64[ns, UTC]]`; `load_issues(raw_dir) -> DataFrame`, `load_snapshot(raw_dir) -> DataFrame`, `load_history(raw_dir) -> DataFrame`.

- [ ] **Step 1: Init repo and environment**

```bash
cd ~/workforce-opt
git init
cat > .gitignore <<'EOF'
data/raw/
results/
.venv/
__pycache__/
*.pyc
.pytest_cache/
EOF
cat > pyproject.toml <<'EOF'
[project]
name = "wfo"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "pandas>=2.2",
  "numpy>=1.26",
  "ortools>=9.10",
  "networkx>=3.2",
  "matplotlib>=3.8",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
gurobi = ["gurobipy>=11"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/wfo"]
EOF
uv venv --python 3.12 && uv pip install -e ".[dev]"
mkdir -p src/wfo tests config scripts results report && touch src/wfo/__init__.py
```

- [ ] **Step 2: Write `config/default.yaml`**

```yaml
raw_dir: data/raw
results_dir: results
seed: 571

train_start: "2021-01-01"
train_end: "2022-07-01"
test_start: "2022-07-01"
test_end: "2023-03-01"
n_test_weeks: 20

business_day: [9, 18]          # UTC working window used for active time
effort_cap_hours: 40.0          # clip per-turn active hours
shrinkage_k: 5.0
speed_clamp: [0.25, 4.0]
min_turns_eligible: 3
novice_multiplier: 1.5

# ASSUMED active effort (hours) per task category. Scale is an assumption; relative speeds come from data.
base_hours:
  ticket/low: 1.0
  ticket/med: 1.5
  ticket/high: 2.0
  ticket/urgent: 3.0
  service/low: 1.0
  service/med: 1.0
  service/high: 1.5
  service/urgent: 2.0
  deploy: 2.0

capacity_hours: 24.0            # 40 h/week × 0.6 focus factor
active_window_days: 14
unassigned_penalty: 1000.0
fair_share_multipliers: [1.5, 2.0, 3.0]

solver: SCIP                    # set to GUROBI if a license is available
time_limit_s: 60

# sensitivity grids
noise_sigmas: [0.1, 0.25, 0.5]
noise_draws: 5
capacity_scales: [0.5, 0.75, 1.0, 1.25, 1.5]
min_turns_grid: [1, 3, 5, 10]

# makespan extension
verify_fraction: 0.3
verify_statuses: [validation, under_review, in_review, pending_deployment]
makespan_batch_size: 40
makespan_time_limit_s: 30
```

- [ ] **Step 3: Write the failing config test**

`tests/test_config.py`:
```python
from pathlib import Path
from wfo.config import load_config

ROOT = Path(__file__).resolve().parents[1]

def test_load_default_config():
    cfg = load_config(ROOT / "config/default.yaml")
    assert cfg.seed == 571
    assert cfg.business_day == [9, 18]
    assert set(cfg.base_hours) == {
        "ticket/low", "ticket/med", "ticket/high", "ticket/urgent",
        "service/low", "service/med", "service/high", "service/urgent", "deploy",
    }
    assert cfg.capacity_hours == 24.0
```

- [ ] **Step 4: Run it, expect failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'wfo.config'`

- [ ] **Step 5: Implement `src/wfo/config.py`**

```python
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    raw_dir: str
    results_dir: str
    seed: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_test_weeks: int
    business_day: list[int]
    effort_cap_hours: float
    shrinkage_k: float
    speed_clamp: list[float]
    min_turns_eligible: int
    novice_multiplier: float
    base_hours: dict[str, float]
    capacity_hours: float
    active_window_days: int
    unassigned_penalty: float
    fair_share_multipliers: list[float]
    solver: str
    time_limit_s: int
    noise_sigmas: list[float]
    noise_draws: int
    capacity_scales: list[float]
    min_turns_grid: list[int]
    verify_fraction: float
    verify_statuses: list[str]
    makespan_batch_size: int
    makespan_time_limit_s: int


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config(**yaml.safe_load(f))
```

- [ ] **Step 6: Implement `src/wfo/load.py`**

```python
import pandas as pd

WORK_COLS = ["id", "issue_type", "issue_priority", "issue_proj",
             "issue_created", "issue_resolution_date", "issue_contr_count"]


def ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def load_issues(raw_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{raw_dir}/issues.csv", low_memory=False, usecols=WORK_COLS)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype("int64")
    df["issue_created"] = ts(df["issue_created"])
    df["issue_resolution_date"] = ts(df["issue_resolution_date"])
    return df


def load_snapshot(raw_dir: str) -> pd.DataFrame:
    cols = ["idx", "id", "issue_assignee", "turn", "started", "ended"]
    df = pd.read_csv(f"{raw_dir}/issues_snapshot.csv", low_memory=False, usecols=cols)
    df["started"] = ts(df["started"])
    df["ended"] = ts(df["ended"])
    df = df.dropna(subset=["id", "issue_assignee", "started", "ended"])
    df["id"] = df["id"].astype("int64")
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce").astype("Int64")
    return df.rename(columns={"issue_assignee": "assignee"})


def load_history(raw_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{raw_dir}/issues_change_history.csv", low_memory=False)
    df = df.dropna(subset=["issueid", "created"])
    df["issueid"] = df["issueid"].astype("int64")
    df["created"] = ts(df["created"])
    return df.sort_values(["issueid", "created", "id"]).reset_index(drop=True)
```

- [ ] **Step 7: Run tests and do a smoke check on real data**

Run: `uv run pytest tests/test_config.py -v` → PASS
Run: `uv run python -c "from wfo.load import *; print(len(load_issues('data/raw')), len(load_snapshot('data/raw')), len(load_history('data/raw')))"`
Expected: roughly `66691 9xxxx 257508` (snapshot is slightly under 90,963 after dropping NaN assignees).

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml config src tests
git commit -m "chore: scaffold project, config, and robust CSV loaders"
```

---

### Task 2: Task categories

**Files:**
- Create: `src/wfo/categories.py`, `tests/conftest.py`
- Test: `tests/test_categories.py`

**Interfaces:**
- Consumes: issues DataFrame with `issue_type`, `issue_priority`.
- Produces: `categorize(issues: DataFrame) -> Series[str | NaN]`; constants `TYPE_GROUP`, `PRIORITY_BAND`.

- [ ] **Step 1: Write the failing test**

`tests/test_categories.py`:
```python
import pandas as pd
from wfo.categories import categorize

def test_categorize_groups_and_bands():
    df = pd.DataFrame({
        "issue_type": ["Ticket", "HD Service", "Service", "Deployment", "Vacation", "Ticket"],
        "issue_priority": ["Blocker", "Low", "unknown", "High", "Medium", "Highest"],
    })
    out = categorize(df).tolist()
    assert out[:4] == ["ticket/urgent", "service/low", "service/med", "deploy"]
    assert pd.isna(out[4])
    assert out[5] == "ticket/urgent"
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_categories.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/wfo/categories.py`**

```python
import numpy as np
import pandas as pd

TYPE_GROUP = {"Ticket": "ticket", "Service": "service", "HD Service": "service", "Deployment": "deploy"}
PRIORITY_BAND = {"Lowest": "low", "Low": "low", "Medium": "med", "unknown": "med",
                 "High": "high", "Highest": "urgent", "Blocker": "urgent"}


def categorize(issues: pd.DataFrame) -> pd.Series:
    group = issues["issue_type"].map(TYPE_GROUP)
    band = issues["issue_priority"].map(PRIORITY_BAND).fillna("med")
    cat = np.where(group == "deploy", "deploy", group + "/" + band)
    return pd.Series(cat, index=issues.index).where(group.notna())
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Create shared fixtures `tests/conftest.py`** (later tasks use them)

```python
import pandas as pd
import pytest

U = lambda s: pd.Timestamp(s, tz="UTC")


@pytest.fixture
def tiny_history():
    # issue 1: open -> in_progress (Mon 10:00) -> resolved (Mon 14:00)
    # issue 2: open -> resolved
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "issueid": [1, 1, 1, 2, 2],
        "field": ["status"] * 5,
        "value": ["open", "in_progress", "resolved", "open", "resolved"],
        "created": [U("2024-01-01 09:00"), U("2024-01-01 10:00"), U("2024-01-01 14:00"),
                    U("2024-01-02 09:00"), U("2024-01-02 11:00")],
        "change_group_id": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def tiny_snapshot():
    return pd.DataFrame({
        "idx": [0, 1, 2],
        "id": [1, 1, 2],
        "assignee": ["a", "b", "a"],
        "turn": [1, 2, 1],
        "started": [U("2024-01-01 09:00"), U("2024-01-01 12:00"), U("2024-01-02 09:00")],
        "ended": [U("2024-01-01 12:00"), U("2024-01-01 14:00"), U("2024-01-02 11:00")],
    })
```

- [ ] **Step 6: Commit** — `git add src/wfo/categories.py tests && git commit -m "feat: task categories from type x priority"`

---

### Task 3: Effort model (business hours → active hours → relative speeds → time table)

**Files:**
- Create: `src/wfo/effort.py`
- Test: `tests/test_effort.py`

**Interfaces:**
- Consumes: `load_history`, `load_snapshot` frames; `categorize`.
- Produces:
  - `business_hours_between(start: Timestamp, end: Timestamp, day_start: int, day_end: int) -> float`
  - `status_intervals(history: DataFrame, status: str = "in_progress") -> DataFrame[issueid, start, end]`
  - `turn_active_hours(snapshot, intervals, day_start, day_end, cap) -> Series` (indexed like snapshot)
  - `estimate_speeds(turns: DataFrame[assignee, category, active_hours], k: float) -> DataFrame[assignee, category, n, speed_log, emp_speed_log]`
  - `build_time_table(speeds, base_hours: dict, clamp: list[float]) -> DataFrame[assignee, category, n, speed_log, emp_speed_log, hours]`

- [ ] **Step 1: Write the failing tests**

`tests/test_effort.py`:
```python
import numpy as np
import pandas as pd
import pytest
from wfo.effort import (business_hours_between, status_intervals, turn_active_hours,
                        estimate_speeds, build_time_table)

U = lambda s: pd.Timestamp(s, tz="UTC")  # 2024-01-01 is a Monday

@pytest.mark.parametrize("s,e,expected", [
    ("2024-01-01 10:00", "2024-01-01 12:00", 2.0),   # same day
    ("2024-01-05 17:00", "2024-01-08 10:00", 2.0),   # Fri 17 -> Mon 10 skips weekend
    ("2024-01-01 09:00", "2024-01-08 09:00", 45.0),  # one full work week
    ("2024-01-06 10:00", "2024-01-06 12:00", 0.0),   # Saturday
    ("2024-01-01 20:00", "2024-01-01 21:00", 0.0),   # after hours
    ("2024-01-01 12:00", "2024-01-01 10:00", 0.0),   # reversed
])
def test_business_hours_between(s, e, expected):
    assert business_hours_between(U(s), U(e), 9, 18) == pytest.approx(expected)

def test_status_intervals(tiny_history):
    iv = status_intervals(tiny_history, "in_progress")
    assert len(iv) == 1
    assert iv.iloc[0].issueid == 1
    assert iv.iloc[0].start == U("2024-01-01 10:00") and iv.iloc[0].end == U("2024-01-01 14:00")

def test_turn_active_hours_splits_by_turn_window(tiny_history, tiny_snapshot):
    iv = status_intervals(tiny_history, "in_progress")
    hrs = turn_active_hours(tiny_snapshot, iv, 9, 18, cap=40.0)
    # turn a (09-12) overlaps in_progress 10-14 for 2h; turn b (12-14) for 2h; issue 2 has none
    assert hrs.tolist() == [2.0, 2.0, 0.0]

def test_turn_active_hours_cap(tiny_history, tiny_snapshot):
    iv = status_intervals(tiny_history, "in_progress")
    assert turn_active_hours(tiny_snapshot, iv, 9, 18, cap=1.5).max() == 1.5

def test_speeds_median_worker_is_base_and_fast_worker_is_faster():
    turns = pd.DataFrame({
        "assignee": ["mid"] * 10 + ["fast"] * 10 + ["slow"] * 10,
        "category": ["ticket/med"] * 30,
        "active_hours": [4.0] * 10 + [1.0] * 10 + [12.0] * 10,
    })
    sp = estimate_speeds(turns, k=5.0)
    tt = build_time_table(sp, {"ticket/med": 1.5}, clamp=[0.25, 4.0]).set_index("assignee")
    assert tt.loc["mid", "hours"] == pytest.approx(1.5)
    assert tt.loc["fast", "hours"] < 1.5 < tt.loc["slow", "hours"]
    assert (tt["n"] == 10).all()

def test_shrinkage_pulls_sparse_cells_toward_employee_speed():
    turns = pd.DataFrame({
        "assignee": ["x"] * 21,
        "category": ["ticket/med"] * 20 + ["ticket/high"],
        "active_hours": [1.0] * 20 + [100.0],
    })
    # add peers so category medians are defined independently of x
    peers = pd.DataFrame({"assignee": ["p"] * 20, "category": ["ticket/med"] * 10 + ["ticket/high"] * 10,
                          "active_hours": [4.0] * 20})
    sp = estimate_speeds(pd.concat([turns, peers]), k=5.0).set_index(["assignee", "category"])
    one_obs = sp.loc[("x", "ticket/high"), "speed_log"]
    raw = np.log(100.25) - np.log(4.25)
    assert one_obs < raw  # shrunk away from the single extreme observation
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_effort.py -v`.

- [ ] **Step 3: Implement `src/wfo/effort.py`**

```python
import numpy as np
import pandas as pd

EPS_HOURS = 0.25  # smooths log-ratios for near-zero active times


def business_hours_between(start: pd.Timestamp, end: pd.Timestamp, day_start: int, day_end: int) -> float:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0

    def overlap(day: pd.Timestamp) -> float:
        if day.weekday() >= 5:
            return 0.0
        ws, we = day + pd.Timedelta(hours=day_start), day + pd.Timedelta(hours=day_end)
        lo, hi = max(start, ws), min(end, we)
        return max((hi - lo).total_seconds(), 0.0) / 3600.0

    first, last = start.normalize(), end.normalize()
    if first == last:
        return overlap(first)
    full_days = np.busday_count((first + pd.Timedelta(days=1)).date(), last.date())
    return overlap(first) + overlap(last) + float(full_days) * (day_end - day_start)


def status_intervals(history: pd.DataFrame, status: str = "in_progress") -> pd.DataFrame:
    h = history[history["field"] == "status"].sort_values(["issueid", "created", "id"])
    nxt = h.groupby("issueid")["created"].shift(-1)
    m = h["value"] == status
    out = pd.DataFrame({"issueid": h.loc[m, "issueid"], "start": h.loc[m, "created"], "end": nxt[m]})
    return out.dropna(subset=["end"]).reset_index(drop=True)  # open-ended intervals dropped


def turn_active_hours(snapshot: pd.DataFrame, intervals: pd.DataFrame,
                      day_start: int, day_end: int, cap: float) -> pd.Series:
    j = snapshot[["idx", "id", "started", "ended"]].merge(
        intervals, left_on="id", right_on="issueid", how="inner")
    lo = j[["started", "start"]].max(axis=1)
    hi = j[["ended", "end"]].min(axis=1)
    keep = hi > lo
    j = j[keep].assign(lo=lo[keep], hi=hi[keep])
    j["h"] = [business_hours_between(a, b, day_start, day_end) for a, b in zip(j["lo"], j["hi"])]
    per_turn = j.groupby("idx")["h"].sum()
    out = snapshot["idx"].map(per_turn).fillna(0.0).clip(upper=cap)
    out.index = snapshot.index
    return out


def estimate_speeds(turns: pd.DataFrame, k: float) -> pd.DataFrame:
    med = turns.groupby("category")["active_hours"].median()
    lr = np.log(turns["active_hours"] + EPS_HOURS) - np.log(turns["category"].map(med) + EPS_HOURS)
    t = turns.assign(lr=lr)
    emp = t.groupby("assignee")["lr"].agg(["sum", "count"])
    s_i = emp["sum"] / (emp["count"] + k)
    ec = t.groupby(["assignee", "category"])["lr"].agg(["sum", "count"]).reset_index()
    ec["emp_speed_log"] = ec["assignee"].map(s_i)
    ec["speed_log"] = (ec["sum"] + k * ec["emp_speed_log"]) / (ec["count"] + k)
    return ec.rename(columns={"count": "n"})[["assignee", "category", "n", "speed_log", "emp_speed_log"]]


def build_time_table(speeds: pd.DataFrame, base_hours: dict[str, float], clamp: list[float]) -> pd.DataFrame:
    mult = np.exp(speeds["speed_log"]).clip(clamp[0], clamp[1])
    return speeds.assign(hours=speeds["category"].map(base_hours) * mult)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Real-data sanity check (no commit of output)**

```bash
uv run python - <<'EOF'
from wfo.config import load_config; from wfo.load import *; from wfo.categories import categorize
from wfo.effort import *
cfg=load_config("config/default.yaml"); iss=load_issues(cfg.raw_dir); snap=load_snapshot(cfg.raw_dir); hist=load_history(cfg.raw_dir)
iss["category"]=categorize(iss); snap=snap.merge(iss[["id","category"]],on="id").dropna(subset=["category"])
snap["active_hours"]=turn_active_hours(snap,status_intervals(hist),*cfg.business_day,cfg.effort_cap_hours)
tr=snap[(snap.started>=cfg.train_start)&(snap.started<cfg.train_end)]
print(tr.active_hours.describe()); print((tr.active_hours==0).mean())
tt=build_time_table(estimate_speeds(tr,cfg.shrinkage_k),cfg.base_hours,cfg.speed_clamp)
print(tt.groupby("category").hours.describe()[["count","min","50%","max"]])
EOF
```
Expected: per-category hours spread around `base_hours` (from ≈0.25× to 4× base), with a non-trivial share of exactly-zero turns. **Record the zero share in the report's limitations section.** If more than ~60% of turns are zero, raise `EPS_HOURS` to 1.0 and note the change.

- [ ] **Step 6: Commit** — `git add src/wfo/effort.py tests/test_effort.py && git commit -m "feat: active-time proxy and shrunk relative-speed effort model"`

---

### Task 4: Workflow map

**Files:**
- Create: `src/wfo/workflow_map.py`
- Test: `tests/test_workflow_map.py`

**Interfaces:**
- Consumes: history DataFrame; `business_hours_between`.
- Produces:
  - `transition_counts(history) -> DataFrame[src, dst, count]` (includes `START`, `END`)
  - `transition_matrix(counts) -> DataFrame` (row-normalized, index=src, columns=dst)
  - `dwell_table(history) -> DataFrame[status, visits, median_hours, total_hours, share_total]`, sorted by `total_hours` desc (wall-clock; the top row is the bottleneck)
  - `draw_workflow_map(matrix, dwell, path, min_prob=0.05) -> None`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from wfo.workflow_map import transition_counts, transition_matrix, dwell_table

def test_transitions(tiny_history):
    c = transition_counts(tiny_history).set_index(["src", "dst"])["count"]
    assert c[("START", "open")] == 2
    assert c[("open", "in_progress")] == 1
    assert c[("open", "resolved")] == 1
    assert c[("resolved", "END")] == 2
    P = transition_matrix(transition_counts(tiny_history))
    assert P.loc["open", "in_progress"] == pytest.approx(0.5)
    assert P.sum(axis=1).round(9).eq(1).all()

def test_dwell(tiny_history):
    d = dwell_table(tiny_history).set_index("status")
    assert d.loc["in_progress", "total_hours"] == pytest.approx(4.0)
    assert d.loc["open", "visits"] == 2
    assert d.loc["open", "total_hours"] == pytest.approx(3.0)  # 1h on issue 1 + 2h on issue 2
    assert d.index[0] == "in_progress"                          # bottleneck = largest total dwell
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/workflow_map.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def _status(history: pd.DataFrame) -> pd.DataFrame:
    return history[history["field"] == "status"].sort_values(["issueid", "created", "id"])


def transition_counts(history: pd.DataFrame) -> pd.DataFrame:
    h = _status(history)
    src = h.groupby("issueid")["value"].shift(1).fillna("START")
    pairs = pd.DataFrame({"src": src, "dst": h["value"]})
    last = h.groupby("issueid")["value"].last()
    ends = pd.DataFrame({"src": last.values, "dst": "END"})
    allp = pd.concat([pairs, ends], ignore_index=True)
    return allp.value_counts(["src", "dst"]).rename("count").reset_index()


def transition_matrix(counts: pd.DataFrame) -> pd.DataFrame:
    m = counts.pivot_table(index="src", columns="dst", values="count", aggfunc="sum", fill_value=0)
    return m.div(m.sum(axis=1), axis=0)


def dwell_table(history: pd.DataFrame) -> pd.DataFrame:
    h = _status(history)
    nxt = h.groupby("issueid")["created"].shift(-1)
    hrs = (nxt - h["created"]).dt.total_seconds() / 3600.0
    t = pd.DataFrame({"status": h["value"], "hours": hrs})
    out = t.groupby("status").agg(visits=("hours", "size"),
                                  median_hours=("hours", "median"),
                                  total_hours=("hours", "sum")).reset_index()
    out["share_total"] = out["total_hours"] / out["total_hours"].sum()
    return out.sort_values("total_hours", ascending=False).reset_index(drop=True)


def draw_workflow_map(matrix: pd.DataFrame, dwell: pd.DataFrame, path: str, min_prob: float = 0.05) -> None:
    g = nx.DiGraph()
    for s in matrix.index:
        for d, p in matrix.loc[s].items():
            if p >= min_prob:
                g.add_edge(s, d, weight=p)
    size = dwell.set_index("status")["share_total"].to_dict()
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(g, seed=571, k=1.2)
    nx.draw_networkx_nodes(g, pos, node_size=[300 + 6000 * size.get(n, 0) for n in g.nodes], ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(g, pos, width=[3 * g[u][v]["weight"] for u, v in g.edges],
                           arrowsize=10, connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edge_labels(g, pos, {(u, v): f"{g[u][v]['weight']:.2f}" for u, v in g.edges},
                                 font_size=6, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
```

- [ ] **Step 4: Run, expect PASS.** Then a real-data smoke run: build the matrix and dwell table for 2021+ issues and confirm `closed`, `in_progress`, `resolved`, `waiting` dominate (they lead the status counts).

- [ ] **Step 5: Commit** — `git commit -am "feat: workflow map (transitions, dwell, bottleneck table, figure)"` (after `git add` of the new files).

---

### Task 5: Weekly instances

**Files:**
- Create: `src/wfo/instance.py`
- Test: `tests/test_instance.py`

**Interfaces:**
- Consumes: issues with `category`; snapshot; time tables from `build_time_table`.
- Produces:
  - `@dataclass Instance: week_start, employees: list[str], task_ids: list[int], categories: list[str], created: list[Timestamp], t_plan: ndarray (n_emp × n_task, inf = ineligible), t_eval: ndarray (finite everywhere), capacity: ndarray, hist_assignee: list[str | None], speed_log_emp: dict[str, float]`
  - `active_employees(snapshot, week_start, window_days) -> set[str]`
  - `build_week_instance(week_start, issues, first_turn_assignee: dict[int,str], tt_plan, tt_eval, active: set[str], cfg, min_turns: int | None = None) -> Instance`
  - `sample_test_weeks(cfg) -> list[Timestamp]` (Mondays in the test window, `n_test_weeks` drawn without replacement, sorted)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from types import SimpleNamespace
from wfo.instance import build_week_instance, active_employees

U = lambda s: pd.Timestamp(s, tz="UTC")

def _cfg():
    return SimpleNamespace(base_hours={"ticket/med": 2.0, "deploy": 3.0}, novice_multiplier=1.5,
                           capacity_hours=24.0, min_turns_eligible=3, speed_clamp=[0.25, 4.0])

def test_build_week_instance_matrices():
    issues = pd.DataFrame({"id": [10, 11, 12], "category": ["ticket/med", "deploy", "ticket/med"],
                           "issue_created": [U("2024-01-01 10:00"), U("2024-01-02 10:00"), U("2024-01-20")]})
    tt_plan = pd.DataFrame({"assignee": ["a", "a", "b"], "category": ["ticket/med", "deploy", "ticket/med"],
                            "n": [5, 1, 4], "speed_log": [0, 0, 0], "emp_speed_log": [0, 0, 0],
                            "hours": [1.0, 3.0, 2.5]})
    tt_eval = tt_plan.assign(hours=[1.2, 3.0, 2.0])
    inst = build_week_instance(U("2024-01-01"), issues, {10: "b", 11: "a"}, tt_plan, tt_eval, {"a", "b"}, _cfg())
    assert inst.task_ids == [10, 11]                      # issue 12 is outside the week
    ia, ib = inst.employees.index("a"), inst.employees.index("b")
    assert inst.t_plan[ia, 0] == 1.0 and inst.t_plan[ib, 0] == 2.5
    assert np.isinf(inst.t_plan[ia, 1])                   # n=1 < min_turns -> ineligible
    assert np.isinf(inst.t_plan[ib, 1])
    assert inst.t_eval[ib, 1] == 3.0 * 1.5                # never observed -> novice
    assert inst.t_eval[ia, 0] == 1.2                      # test-period estimate used for evaluation
    assert inst.hist_assignee == ["b", "a"]
    assert (inst.capacity == 24.0).all()

def test_active_employees():
    snap = pd.DataFrame({"assignee": ["a", "b", "c"],
                         "started": [U("2023-12-25"), U("2024-01-03"), U("2024-03-01")]})
    assert active_employees(snap, U("2024-01-01"), 14) == {"a", "b"}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/instance.py`**

```python
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Instance:
    week_start: pd.Timestamp
    employees: list[str]
    task_ids: list[int]
    categories: list[str]
    created: list[pd.Timestamp]
    t_plan: np.ndarray
    t_eval: np.ndarray
    capacity: np.ndarray
    hist_assignee: list[str | None]
    speed_log_emp: dict[str, float] = field(default_factory=dict)


def active_employees(snapshot: pd.DataFrame, week_start: pd.Timestamp, window_days: int) -> set[str]:
    lo = week_start - pd.Timedelta(days=window_days)
    hi = week_start + pd.Timedelta(days=7 + window_days)
    m = (snapshot["started"] >= lo) & (snapshot["started"] < hi)
    return set(snapshot.loc[m, "assignee"])


def build_week_instance(week_start, issues, first_turn_assignee, tt_plan, tt_eval, active, cfg,
                        min_turns: int | None = None) -> Instance:
    k = cfg.min_turns_eligible if min_turns is None else min_turns
    wk = issues[(issues["issue_created"] >= week_start)
                & (issues["issue_created"] < week_start + pd.Timedelta(days=7))
                & issues["category"].notna()].sort_values("issue_created")
    elig = tt_plan[tt_plan["n"] >= k]
    emps = sorted(set(elig["assignee"]) & set(active))
    cats = wk["category"].tolist()
    plan = elig.set_index(["assignee", "category"])["hours"].to_dict()
    ev = tt_eval.set_index(["assignee", "category"])["hours"].to_dict()
    tp = tt_plan.set_index(["assignee", "category"])["hours"].to_dict()
    n, m = len(emps), len(cats)
    t_plan = np.full((n, m), np.inf)
    t_eval = np.empty((n, m))
    for i, e in enumerate(emps):
        for j, c in enumerate(cats):
            if (e, c) in plan:
                t_plan[i, j] = plan[(e, c)]
            t_eval[i, j] = ev.get((e, c), tp.get((e, c), cfg.base_hours[c] * cfg.novice_multiplier))
    emp_speed = tt_plan.groupby("assignee")["emp_speed_log"].first().to_dict()
    return Instance(
        week_start=week_start, employees=emps, task_ids=wk["id"].tolist(), categories=cats,
        created=wk["issue_created"].tolist(), t_plan=t_plan, t_eval=t_eval,
        capacity=np.full(n, float(cfg.capacity_hours)),
        hist_assignee=[first_turn_assignee.get(i) for i in wk["id"]],
        speed_log_emp={e: emp_speed.get(e, 0.0) for e in emps},
    )


def sample_test_weeks(cfg) -> list[pd.Timestamp]:
    mondays = pd.date_range(cfg.test_start, cfg.test_end, freq="W-MON", tz="UTC", inclusive="left")
    mondays = mondays[mondays + pd.Timedelta(days=7) <= pd.Timestamp(cfg.test_end, tz="UTC")]
    rng = np.random.default_rng(cfg.seed)
    pick = rng.choice(len(mondays), size=min(cfg.n_test_weeks, len(mondays)), replace=False)
    return sorted(mondays[pick])
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `git add src/wfo/instance.py tests/test_instance.py && git commit -m "feat: weekly assignment instances with plan/eval time matrices"`

---

### Task 6: GAP MILP, workload-cap variant, LP duals

**Files:**
- Create: `src/wfo/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `t: ndarray (n×m, inf=ineligible)`, `capacity: ndarray (n,)`.
- Produces:
  - `@dataclass Solution: assign: ndarray[int] (len m, -1 = unassigned), objective: float, solve_seconds: float, status: str`
  - `solve_gap(t, capacity, share_cap: float | None = None, unassigned_penalty=1000.0, solver="SCIP", time_limit_s=60) -> Solution`. `objective` is labor hours only (penalty excluded).
  - `lp_capacity_duals(t, capacity, unassigned_penalty=1000.0) -> ndarray (n,)` = hours of labor saved per extra capacity hour (≥0)

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from wfo.model import solve_gap, lp_capacity_duals

INF = np.inf

def test_gap_respects_capacity():
    t = np.array([[1.0, 1, 1], [2.0, 2, 2]])
    sol = solve_gap(t, np.array([2.0, 10.0]))
    assert sol.status == "OPTIMAL"
    assert sol.objective == pytest.approx(4.0)
    assert (sol.assign == 0).sum() == 2

def test_gap_eligibility():
    t = np.array([[INF, 1.0], [5.0, 5.0]])
    sol = solve_gap(t, np.array([10.0, 10.0]))
    assert sol.assign.tolist() == [1, 0]

def test_gap_unassigned_when_capacity_short():
    t = np.array([[3.0, 3.0]])
    sol = solve_gap(t, np.array([4.0]))
    assert (sol.assign == -1).sum() == 1
    assert sol.objective == pytest.approx(3.0)

def test_share_cap_balances():
    t = np.array([[1.0] * 4, [1.5] * 4])
    assert solve_gap(t, np.array([10.0, 10.0])).objective == pytest.approx(4.0)
    capped = solve_gap(t, np.array([10.0, 10.0]), share_cap=0.5)
    assert capped.objective == pytest.approx(5.0)
    assert (capped.assign == 0).sum() == 2

def test_lp_duals_value_binding_capacity():
    t = np.array([[1.0, 1.0], [3.0, 3.0]])
    d = lp_capacity_duals(t, np.array([1.0, 10.0]))
    assert d[0] == pytest.approx(2.0)
    assert d[1] == pytest.approx(0.0)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/model.py`**

```python
import time
from dataclasses import dataclass
import numpy as np
from ortools.linear_solver import pywraplp

_STATUS = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
           pywraplp.Solver.INFEASIBLE: "INFEASIBLE", pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"}


@dataclass
class Solution:
    assign: np.ndarray
    objective: float
    solve_seconds: float
    status: str


def _build(s, t, capacity, share_cap, unassigned_penalty, integer):
    n, m = t.shape
    elig = np.isfinite(t)
    var = s.BoolVar if integer else (lambda name: s.NumVar(0.0, 1.0, name))
    x = {(i, j): var(f"x_{i}_{j}") for i in range(n) for j in range(m) if elig[i, j]}
    u = [var(f"u_{j}") for j in range(m)]
    for j in range(m):
        s.Add(sum(x[i, j] for i in range(n) if elig[i, j]) + u[j] == 1)
    loads = [sum(t[i, j] * x[i, j] for j in range(m) if elig[i, j]) for i in range(n)]
    caps = []
    for i in range(n):
        if elig[i].any():
            caps.append(s.Add(loads[i] <= float(capacity[i])))
        else:
            caps.append(None)
    labor = sum(l for i, l in enumerate(loads) if elig[i].any())
    if share_cap is not None:
        for i in range(n):
            if elig[i].any():
                s.Add(loads[i] <= share_cap * labor)
    s.Minimize(labor + unassigned_penalty * sum(u))
    return x, u, caps, elig


def solve_gap(t, capacity, share_cap=None, unassigned_penalty=1000.0, solver="SCIP", time_limit_s=60) -> Solution:
    s = pywraplp.Solver.CreateSolver(solver)
    if s is None:
        raise RuntimeError(f"solver {solver} unavailable")
    s.SetTimeLimit(int(time_limit_s * 1000))
    x, u, _, elig = _build(s, t, capacity, share_cap, unassigned_penalty, integer=True)
    t0 = time.perf_counter()
    st = s.Solve()
    dt = time.perf_counter() - t0
    n, m = t.shape
    assign = np.full(m, -1, dtype=int)
    for (i, j), v in x.items():
        if v.solution_value() > 0.5:
            assign[j] = i
    obj = float(sum(t[assign[j], j] for j in range(m) if assign[j] >= 0))
    return Solution(assign=assign, objective=obj, solve_seconds=dt, status=_STATUS.get(st, str(st)))


def lp_capacity_duals(t, capacity, unassigned_penalty=1000.0) -> np.ndarray:
    s = pywraplp.Solver.CreateSolver("GLOP")
    _, _, caps, _ = _build(s, t, capacity, None, unassigned_penalty, integer=False)
    s.Solve()
    return np.array([abs(c.dual_value()) if c is not None else 0.0 for c in caps])
```

> Note: with `share_cap`, the right-hand side includes total labor, so in principle the model could pad total labor to loosen the cap. Minimizing labor works against that, and the frontier plot will show whether it happens. Mention it in the report as a modeling caveat.

- [ ] **Step 4: Run, expect PASS.** Also time one real week: build an instance from a test week (Task 5) and call `solve_gap`. Expect ~50 × ~150 binaries and solve time under 10 s on SCIP. **Calibration check:** print the historical-assignment utilization (Task 8's `evaluate`, once built) and the MILP's unassigned count. If the MILP leaves more than 5% unassigned at `capacity_hours: 24`, the base_hours/capacity scale is inconsistent. Then lower `base_hours` proportionally and document it.

- [ ] **Step 5: Commit** — `git add src/wfo/model.py tests/test_model.py && git commit -m "feat: GAP MILP with backlog option, share-cap variant, LP capacity duals"`

---

### Task 7: Baselines

**Files:**
- Create: `src/wfo/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `Instance`.
- Produces (all return `ndarray[int]` of length m, -1 = not assignable to an in-instance employee):
  - `historical(inst) -> ndarray`: the actual turn-1 assignee if they are in `inst.employees`, else -1
  - `historical_proportions(inst, train_counts: DataFrame[assignee, category, n], rng) -> ndarray`: sample i ∝ the employee's training count in c(j), restricted to `inst.employees`; -1 if nobody has history
  - `round_robin(inst) -> ndarray`: tasks in creation order, global pointer over `inst.employees`, skipping employees with `t_plan = inf`; -1 if nobody is eligible

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
from types import SimpleNamespace
from wfo.baselines import historical, historical_proportions, round_robin

INF = np.inf

def _inst(t, hist=None, cats=None):
    n, m = t.shape
    return SimpleNamespace(employees=[f"e{i}" for i in range(n)], t_plan=t,
                           hist_assignee=hist or [None] * m, categories=cats or ["c"] * m)

def test_round_robin_cycles():
    assert round_robin(_inst(np.ones((3, 5)))).tolist() == [0, 1, 2, 0, 1]

def test_round_robin_skips_ineligible():
    t = np.array([[1, INF, 1], [1, 1, 1.0]])
    assert round_robin(_inst(t)).tolist() == [0, 1, 0]

def test_historical_maps_names_and_unknowns():
    inst = _inst(np.ones((2, 3)), hist=["e1", "zzz", None])
    assert historical(inst).tolist() == [1, -1, -1]

def test_historical_proportions_degenerate():
    inst = _inst(np.ones((2, 3)), cats=["c", "c", "d"])
    counts = pd.DataFrame({"assignee": ["e1", "e0"], "category": ["c", "d"], "n": [10, 4]})
    out = historical_proportions(inst, counts, np.random.default_rng(0))
    assert out.tolist() == [1, 1, 0]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/baselines.py`**

```python
import numpy as np
import pandas as pd


def historical(inst) -> np.ndarray:
    idx = {e: i for i, e in enumerate(inst.employees)}
    return np.array([idx.get(h, -1) if h is not None else -1 for h in inst.hist_assignee], dtype=int)


def historical_proportions(inst, train_counts: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    idx = {e: i for i, e in enumerate(inst.employees)}
    tc = train_counts[train_counts["assignee"].isin(idx)]
    by_cat = {c: g for c, g in tc.groupby("category")}
    out = np.full(len(inst.categories), -1, dtype=int)
    for j, c in enumerate(inst.categories):
        g = by_cat.get(c)
        if g is None or g["n"].sum() == 0:
            continue
        p = g["n"].to_numpy(dtype=float)
        pick = rng.choice(len(g), p=p / p.sum())
        out[j] = idx[g["assignee"].iloc[pick]]
    return out


def round_robin(inst) -> np.ndarray:
    n, m = inst.t_plan.shape
    out = np.full(m, -1, dtype=int)
    ptr = 0
    for j in range(m):
        for step in range(n):
            i = (ptr + step) % n
            if np.isfinite(inst.t_plan[i, j]):
                out[j] = i
                ptr = i + 1
                break
    return out
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git add src/wfo/baselines.py tests/test_baselines.py && git commit -m "feat: historical, proportional, round-robin baselines"`

---

### Task 8: Metrics

**Files:**
- Create: `src/wfo/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `assign: ndarray[int]`, `t_eval: ndarray (n×m)`, `capacity: ndarray`.
- Produces: `gini(x: ndarray) -> float`; `evaluate(assign, t_eval, capacity) -> dict` with keys `total_hours, n_tasks, n_unassigned, max_load, mean_load, cv_load, gini_load, mean_util, max_util, n_over_capacity, overload_hours`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from wfo.metrics import evaluate, gini

def test_gini():
    assert gini(np.array([2.0, 2, 2])) == pytest.approx(0.0)
    assert gini(np.array([0.0, 0, 4])) == pytest.approx(2 / 3)
    assert gini(np.zeros(3)) == 0.0

def test_evaluate():
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r = evaluate(np.array([0, 0, -1]), t, np.array([2.0, 10.0]))
    assert r["total_hours"] == pytest.approx(3.0)
    assert r["n_unassigned"] == 1
    assert r["max_load"] == pytest.approx(3.0)
    assert r["n_over_capacity"] == 1
    assert r["overload_hours"] == pytest.approx(1.0)
    assert r["mean_util"] == pytest.approx((1.5 + 0.0) / 2)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/metrics.py`**

```python
import numpy as np


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2 * x.size ** 2 * x.mean()))


def evaluate(assign: np.ndarray, t_eval: np.ndarray, capacity: np.ndarray) -> dict:
    n, m = t_eval.shape
    loads = np.zeros(n)
    for j, i in enumerate(assign):
        if i >= 0:
            loads[i] += t_eval[i, j]
    util = np.divide(loads, capacity, out=np.zeros(n), where=capacity > 0)
    over = np.clip(loads - capacity, 0, None)
    mean = loads.mean() if n else 0.0
    return {
        "total_hours": float(loads.sum()),
        "n_tasks": int(m),
        "n_unassigned": int((assign < 0).sum()),
        "max_load": float(loads.max()) if n else 0.0,
        "mean_load": float(mean),
        "cv_load": float(loads.std() / mean) if mean > 0 else 0.0,
        "gini_load": gini(loads),
        "mean_util": float(util.mean()) if n else 0.0,
        "max_util": float(util.max()) if n else 0.0,
        "n_over_capacity": int((over > 1e-9).sum()),
        "overload_hours": float(over.sum()),
    }
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git add src/wfo/metrics.py tests/test_metrics.py && git commit -m "feat: evaluation metrics (labor, balance, utilization, overload)"`

> **Comparison caveat for the report:** historical and proportional baselines can leave tasks at -1 (the assignee left or has no history), while the MILP assigns nearly everything. So compare **hours per assigned task** as well as total hours, and always report `n_unassigned` beside them.

---

### Task 9: Makespan extension (CP-SAT, precedence, four-eyes)

**Files:**
- Create: `src/wfo/scheduling.py`
- Test: `tests/test_scheduling.py`

**Interfaces:**
- Consumes: integer-minute durations.
- Produces:
  - `solve_makespan(ops: list[tuple[int,int]], dur: ndarray[int] (n_emp × n_ops, -1 = ineligible), release: dict[int,int], four_eyes: bool = True, time_limit_s: float = 30.0) -> dict` with keys `status, makespan, emp (list[int] per op), start (list[int] per op)`. `ops[o] = (task_j, stage_s)`, stages listed in order within a task.
  - `build_ops(inst, task_cols: list[int], has_verify: set[int], verify_fraction: float, fixed_assign: ndarray | None = None) -> tuple[ops, dur, release]`. Converts an `Instance` sub-batch into CP-SAT inputs. When `fixed_assign` is given, the work stage may only run on that employee, which lets us measure the makespan of the GAP plan.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from wfo.scheduling import solve_makespan

def test_single_employee_sequential():
    r = solve_makespan([(0, 0), (1, 0)], np.array([[2, 3]]), {0: 0, 1: 0})
    assert r["status"] in ("OPTIMAL", "FEASIBLE") and r["makespan"] == 5

def test_two_employees_parallel():
    r = solve_makespan([(0, 0), (1, 0)], np.array([[2, 3], [2, 3]]), {0: 0, 1: 0})
    assert r["makespan"] == 3

def test_release_times():
    r = solve_makespan([(0, 0)], np.array([[2]]), {0: 10})
    assert r["makespan"] == 12 and r["start"][0] == 10

def test_precedence_and_four_eyes():
    ops = [(0, 0), (0, 1)]
    dur = np.array([[2, 1], [2, 1]])
    r = solve_makespan(ops, dur, {0: 0}, four_eyes=True)
    assert r["makespan"] == 3
    assert r["emp"][0] != r["emp"][1]
    assert r["start"][1] >= r["start"][0] + 2

def test_ineligible_respected():
    r = solve_makespan([(0, 0)], np.array([[-1], [4]]), {0: 0})
    assert r["emp"] == [1]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/scheduling.py`**

```python
from collections import defaultdict
import numpy as np
from ortools.sat.python import cp_model

_ST = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
       cp_model.INFEASIBLE: "INFEASIBLE", cp_model.UNKNOWN: "UNKNOWN"}


def solve_makespan(ops, dur, release, four_eyes=True, time_limit_s=30.0) -> dict:
    m = cp_model.CpModel()
    n_emp, n_ops = dur.shape
    horizon = int(max(release.values(), default=0) + dur.clip(min=0).max(axis=0).sum())
    starts, ends, pres = [], [], {}
    per_emp = defaultdict(list)
    for o, (j, _s) in enumerate(ops):
        st = m.NewIntVar(int(release[j]), horizon, f"s{o}")
        en = m.NewIntVar(0, horizon, f"e{o}")
        lits = []
        for i in range(n_emp):
            if dur[i, o] < 0:
                continue
            p = m.NewBoolVar(f"p{i}_{o}")
            per_emp[i].append(m.NewOptionalIntervalVar(st, int(dur[i, o]), en, p, f"iv{i}_{o}"))
            pres[i, o] = p
            lits.append(p)
        m.AddExactlyOne(lits)
        starts.append(st)
        ends.append(en)
    by_task = defaultdict(list)
    for o, (j, s) in enumerate(ops):
        by_task[j].append((s, o))
    for j, lst in by_task.items():
        lst.sort()
        for (_, a), (_, b) in zip(lst, lst[1:]):
            m.Add(starts[b] >= ends[a])
            if four_eyes:
                for i in range(n_emp):
                    if (i, a) in pres and (i, b) in pres:
                        m.Add(pres[i, a] + pres[i, b] <= 1)
    for ivs in per_emp.values():
        m.AddNoOverlap(ivs)
    mk = m.NewIntVar(0, horizon, "makespan")
    m.AddMaxEquality(mk, ends)
    m.Minimize(mk)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = 8
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": _ST.get(st, str(st)), "makespan": None, "emp": [], "start": []}
    emp = [next(i for i in range(n_emp) if (i, o) in pres and solver.Value(pres[i, o])) for o in range(n_ops)]
    return {"status": _ST[st], "makespan": solver.Value(mk), "emp": emp,
            "start": [solver.Value(s) for s in starts]}


def build_ops(inst, task_cols, has_verify, verify_fraction, fixed_assign=None):
    t0 = min(inst.created[j] for j in task_cols)
    ops, cols, release = [], [], {}
    for j in task_cols:
        release[j] = int((inst.created[j] - t0).total_seconds() // 60)
        ops.append((j, 0))
        cols.append(("work", j))
        if inst.task_ids[j] in has_verify:
            ops.append((j, 1))
            cols.append(("verify", j))
    n = len(inst.employees)
    dur = np.full((n, len(ops)), -1, dtype=int)
    for o, (kind, j) in enumerate(cols):
        for i in range(n):
            if not np.isfinite(inst.t_plan[i, j]):
                continue
            if kind == "work" and fixed_assign is not None and fixed_assign[j] != i:
                continue
            frac = 1.0 if kind == "work" else verify_fraction
            dur[i, o] = max(1, int(round(inst.t_plan[i, j] * frac * 60)))
    return ops, dur, release
```

> Assumption to state in the report: time is continuous (no working-hours calendar inside the scheduling model), and release times are minutes since the batch's first ticket. Batches are one day's tickets, capped at `makespan_batch_size`.

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `git add src/wfo/scheduling.py tests/test_scheduling.py && git commit -m "feat: CP-SAT makespan extension with precedence and four-eyes rule"`

---

### Task 10: Experiments and end-to-end runner

**Files:**
- Create: `src/wfo/experiments.py`, `scripts/run_all.py`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `prepare(cfg) -> dict` with keys `issues, snapshot, history, tt_train, tt_test, first_turn, train_counts, has_verify`
  - `run_benchmark_week(inst, cfg, train_counts, rng) -> list[dict]`: one row per method: `historical`, `hist_proportions`, `round_robin`, `gap`, `gap_cap_{m}` for each fair-share multiplier m. Each row carries `evaluate()` output, `method`, `week`, `solve_seconds`, `plan_objective`, `n_employees`, and `hours_per_assigned_task`.
  - `run_noise_sensitivity(inst, cfg, rng) -> list[dict]`: rows `sigma, draw, eval_total_hours, regret_vs_clean`
  - `run_capacity_sensitivity(inst, cfg) -> list[dict]`: rows `scale, total_hours, n_unassigned, gini_load`
  - `run_duals(inst, cfg) -> list[dict]`: rows `employee, dual_saving_per_hour, fd_saving_1h` (finite difference for the top 5 by dual)
  - `run_cross_training(inst, cfg) -> list[dict]`: rows `category, employee, objective_drop`
  - `run_makespan(inst, cfg, has_verify) -> list[dict]`: rows `day, n_tasks, makespan_gap_plan, makespan_joint`
  - `scripts/run_all.py`: writes `results/benchmark.csv`, `results/sens_noise.csv`, `results/sens_capacity.csv`, `results/sens_min_turns.csv`, `results/duals.csv`, `results/cross_training.csv`, `results/makespan.csv`, `results/workflow_transitions.csv`, `results/workflow_dwell.csv`, `results/workflow_map.png`, and `results/summary.md`

- [ ] **Step 1: Write the failing integration test on a synthetic instance**

```python
import numpy as np
import pandas as pd
from types import SimpleNamespace
from wfo.instance import Instance
from wfo.experiments import run_benchmark_week, run_noise_sensitivity, run_duals

def _cfg():
    return SimpleNamespace(seed=1, unassigned_penalty=1000.0, solver="SCIP", time_limit_s=10,
                           fair_share_multipliers=[1.5], noise_sigmas=[0.25], noise_draws=2,
                           base_hours={"c": 1.0}, novice_multiplier=1.5)

def _inst():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.5, 3.0, size=(4, 12))
    t[0, :3] = np.inf
    return Instance(week_start=pd.Timestamp("2024-01-01", tz="UTC"), employees=list("abcd"),
                    task_ids=list(range(12)), categories=["c"] * 12,
                    created=[pd.Timestamp("2024-01-01", tz="UTC")] * 12,
                    t_plan=t, t_eval=np.where(np.isfinite(t), t, 2.0), capacity=np.full(4, 24.0),
                    hist_assignee=list("abcdabcdabcd"))

def test_benchmark_gap_beats_round_robin():
    counts = pd.DataFrame({"assignee": list("abcd"), "category": ["c"] * 4, "n": [5] * 4})
    rows = pd.DataFrame(run_benchmark_week(_inst(), _cfg(), counts, np.random.default_rng(0)))
    r = rows.set_index("method")["total_hours"]
    assert set(rows.method) >= {"historical", "hist_proportions", "round_robin", "gap", "gap_cap_1.5"}
    assert r["gap"] <= r["round_robin"] + 1e-9
    assert r["gap"] <= r["gap_cap_1.5"] + 1e-9

def test_noise_regret_nonnegative():
    rows = run_noise_sensitivity(_inst(), _cfg(), np.random.default_rng(0))
    assert len(rows) == 2 and all(r["regret_vs_clean"] >= -1e-6 for r in rows)

def test_duals_shape():
    rows = run_duals(_inst(), _cfg())
    assert len(rows) == 4 and all(r["dual_saving_per_hour"] >= 0 for r in rows)
```

> Why `regret_vs_clean ≥ 0`: in this synthetic test, `t_eval == t_plan` on eligible cells, so the clean-plan optimum is the true optimum, and any noisy plan can only do worse or equal when evaluated with the same eligibility. Noisy plans must therefore be solved with the same eligibility mask as `t_plan`.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `src/wfo/experiments.py`**

```python
import numpy as np
import pandas as pd

from wfo.baselines import historical, historical_proportions, round_robin
from wfo.categories import categorize
from wfo.effort import build_time_table, estimate_speeds, status_intervals, turn_active_hours
from wfo.load import load_history, load_issues, load_snapshot
from wfo.metrics import evaluate
from wfo.model import lp_capacity_duals, solve_gap


def _window(df, col, lo, hi):
    return df[(df[col] >= pd.Timestamp(lo, tz="UTC")) & (df[col] < pd.Timestamp(hi, tz="UTC"))]


def prepare(cfg) -> dict:
    issues = load_issues(cfg.raw_dir)
    issues["category"] = categorize(issues)
    snap = load_snapshot(cfg.raw_dir).merge(issues[["id", "category"]], on="id")
    snap = snap.dropna(subset=["category"])
    hist = load_history(cfg.raw_dir)
    snap["active_hours"] = turn_active_hours(snap, status_intervals(hist, "in_progress"),
                                             cfg.business_day[0], cfg.business_day[1], cfg.effort_cap_hours)
    train = _window(snap, "started", cfg.train_start, cfg.train_end)
    test = _window(snap, "started", cfg.test_start, cfg.test_end)
    tt_train = build_time_table(estimate_speeds(train, cfg.shrinkage_k), cfg.base_hours, cfg.speed_clamp)
    tt_test = build_time_table(estimate_speeds(test, cfg.shrinkage_k), cfg.base_hours, cfg.speed_clamp)
    first = snap[snap["turn"] == 1].drop_duplicates("id").set_index("id")["assignee"].to_dict()
    st = hist[(hist["field"] == "status") & hist["value"].isin(cfg.verify_statuses)]
    return {"issues": issues, "snapshot": snap, "history": hist, "tt_train": tt_train, "tt_test": tt_test,
            "first_turn": first, "train_counts": tt_train[["assignee", "category", "n"]],
            "has_verify": set(st["issueid"])}


def _row(method, inst, assign, extra=None):
    r = evaluate(assign, inst.t_eval, inst.capacity)
    n_assigned = r["n_tasks"] - r["n_unassigned"]
    r.update(method=method, week=str(inst.week_start.date()), n_employees=len(inst.employees),
             hours_per_assigned_task=r["total_hours"] / n_assigned if n_assigned else np.nan)
    r.update(extra or {})
    return r


def run_benchmark_week(inst, cfg, train_counts, rng) -> list[dict]:
    rows = [_row("historical", inst, historical(inst)),
            _row("hist_proportions", inst, historical_proportions(inst, train_counts, rng)),
            _row("round_robin", inst, round_robin(inst))]
    sol = solve_gap(inst.t_plan, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
    rows.append(_row("gap", inst, sol.assign, {"solve_seconds": sol.solve_seconds,
                                               "plan_objective": sol.objective, "status": sol.status}))
    n = max(len(inst.employees), 1)
    for mult in cfg.fair_share_multipliers:
        s = solve_gap(inst.t_plan, inst.capacity, min(1.0, mult / n), cfg.unassigned_penalty,
                      cfg.solver, cfg.time_limit_s)
        rows.append(_row(f"gap_cap_{mult}", inst, s.assign, {"solve_seconds": s.solve_seconds,
                                                             "plan_objective": s.objective, "status": s.status}))
    return rows


def run_noise_sensitivity(inst, cfg, rng) -> list[dict]:
    clean = evaluate(solve_gap(inst.t_plan, inst.capacity, None, cfg.unassigned_penalty,
                               cfg.solver, cfg.time_limit_s).assign, inst.t_eval, inst.capacity)["total_hours"]
    rows = []
    for sigma in cfg.noise_sigmas:
        for d in range(cfg.noise_draws):
            noisy = inst.t_plan * rng.lognormal(0.0, sigma, size=inst.t_plan.shape)  # inf stays inf
            a = solve_gap(noisy, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s).assign
            tot = evaluate(a, inst.t_eval, inst.capacity)["total_hours"]
            rows.append({"week": str(inst.week_start.date()), "sigma": sigma, "draw": d,
                         "eval_total_hours": tot, "regret_vs_clean": tot - clean})
    return rows


def run_capacity_sensitivity(inst, cfg) -> list[dict]:
    rows = []
    for scale in cfg.capacity_scales:
        cap = inst.capacity * scale
        a = solve_gap(inst.t_plan, cap, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s).assign
        r = evaluate(a, inst.t_eval, cap)
        rows.append({"week": str(inst.week_start.date()), "scale": scale, "total_hours": r["total_hours"],
                     "n_unassigned": r["n_unassigned"], "gini_load": r["gini_load"]})
    return rows


def run_duals(inst, cfg, top_k: int = 5) -> list[dict]:
    duals = lp_capacity_duals(inst.t_plan, inst.capacity, cfg.unassigned_penalty)
    base = solve_gap(inst.t_plan, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
    base_pen = base.objective + cfg.unassigned_penalty * (base.assign < 0).sum()
    fd = {}
    for i in np.argsort(-duals)[:top_k]:
        cap = inst.capacity.copy(); cap[i] += 1.0
        s = solve_gap(inst.t_plan, cap, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
        fd[i] = base_pen - (s.objective + cfg.unassigned_penalty * (s.assign < 0).sum())
    return [{"week": str(inst.week_start.date()), "employee": e, "dual_saving_per_hour": float(duals[i]),
             "fd_saving_1h": fd.get(i, np.nan)} for i, e in enumerate(inst.employees)]


def run_cross_training(inst, cfg, top_k: int = 10) -> list[dict]:
    base = solve_gap(inst.t_plan, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
    loads = pd.Series({c: sum(inst.t_plan[base.assign[j], j] for j in range(len(inst.categories))
                              if inst.categories[j] == c and base.assign[j] >= 0)
                       for c in set(inst.categories)})
    bottleneck = loads.idxmax()
    cols = [j for j, c in enumerate(inst.categories) if c == bottleneck]
    cand = [i for i in range(len(inst.employees)) if not np.isfinite(inst.t_plan[i, cols]).any()]
    cand = sorted(cand, key=lambda i: inst.speed_log_emp.get(inst.employees[i], 0.0))[:top_k]
    rows = []
    for i in cand:
        t = inst.t_plan.copy()
        t[i, cols] = cfg.base_hours[bottleneck] * np.exp(inst.speed_log_emp.get(inst.employees[i], 0.0))
        s = solve_gap(t, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
        rows.append({"week": str(inst.week_start.date()), "category": bottleneck,
                     "employee": inst.employees[i], "objective_drop": base.objective - s.objective,
                     "unassigned_drop": int((base.assign < 0).sum() - (s.assign < 0).sum())})
    return rows


def run_makespan(inst, cfg, has_verify) -> list[dict]:
    from wfo.scheduling import build_ops, solve_makespan
    base = solve_gap(inst.t_plan, inst.capacity, None, cfg.unassigned_penalty, cfg.solver, cfg.time_limit_s)
    by_day = {}
    for j, c in enumerate(inst.created):
        if base.assign[j] >= 0:
            by_day.setdefault(c.date(), []).append(j)
    rows = []
    for day, cols in sorted(by_day.items()):
        cols = cols[: cfg.makespan_batch_size]
        ops, dur, rel = build_ops(inst, cols, has_verify, cfg.verify_fraction, fixed_assign=base.assign)
        fixed = solve_makespan(ops, dur, rel, True, cfg.makespan_time_limit_s)
        ops, dur, rel = build_ops(inst, cols, has_verify, cfg.verify_fraction)
        joint = solve_makespan(ops, dur, rel, True, cfg.makespan_time_limit_s)
        rows.append({"week": str(inst.week_start.date()), "day": str(day), "n_tasks": len(cols),
                     "n_ops": len(ops), "makespan_gap_plan_min": fixed["makespan"],
                     "makespan_joint_min": joint["makespan"], "joint_status": joint["status"]})
    return rows
```

- [ ] **Step 4: Run tests, expect PASS.**

- [ ] **Step 5: Implement `scripts/run_all.py`**

```python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wfo.config import load_config
from wfo.experiments import (prepare, run_benchmark_week, run_capacity_sensitivity, run_cross_training,
                             run_duals, run_makespan, run_noise_sensitivity)
from wfo.instance import active_employees, build_week_instance, sample_test_weeks
from wfo.workflow_map import draw_workflow_map, dwell_table, transition_counts, transition_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--weeks", type=int, default=None, help="override n_test_weeks (use 2 for a smoke run)")
    ap.add_argument("--skip-makespan", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.weeks:
        cfg.n_test_weeks = a.weeks
    out = Path(cfg.results_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    d = prepare(cfg)

    h = d["history"][d["history"]["issueid"].isin(
        d["issues"].loc[d["issues"]["issue_created"] >= pd.Timestamp(cfg.train_start, tz="UTC"), "id"])]
    tc = transition_counts(h); P = transition_matrix(tc); dw = dwell_table(h)
    tc.to_csv(out / "workflow_transitions.csv", index=False); dw.to_csv(out / "workflow_dwell.csv", index=False)
    draw_workflow_map(P, dw, str(out / "workflow_map.png"))

    bench, noise, capr, mt, duals, xt, mk = [], [], [], [], [], [], []
    for w in sample_test_weeks(cfg):
        act = active_employees(d["snapshot"], w, cfg.active_window_days)
        inst = build_week_instance(w, d["issues"], d["first_turn"], d["tt_train"], d["tt_test"], act, cfg)
        print(f"{w.date()}: {len(inst.employees)} employees x {len(inst.task_ids)} tasks", flush=True)
        bench += run_benchmark_week(inst, cfg, d["train_counts"], rng)
        noise += run_noise_sensitivity(inst, cfg, rng)
        capr += run_capacity_sensitivity(inst, cfg)
        duals += run_duals(inst, cfg)
        xt += run_cross_training(inst, cfg)
        for k in cfg.min_turns_grid:
            ik = build_week_instance(w, d["issues"], d["first_turn"], d["tt_train"], d["tt_test"], act, cfg, min_turns=k)
            r = [x for x in run_benchmark_week(ik, cfg, d["train_counts"], rng) if x["method"] == "gap"][0]
            mt.append({"week": str(w.date()), "min_turns": k, **{c: r[c] for c in
                       ("total_hours", "n_unassigned", "gini_load", "n_employees")}})
        if not a.skip_makespan:
            mk += run_makespan(inst, cfg, d["has_verify"])

    for name, rows in [("benchmark", bench), ("sens_noise", noise), ("sens_capacity", capr),
                       ("sens_min_turns", mt), ("duals", duals), ("cross_training", xt), ("makespan", mk)]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)

    b = pd.DataFrame(bench)
    summ = b.groupby("method")[["total_hours", "hours_per_assigned_task", "n_unassigned", "gini_load",
                                "max_util", "n_over_capacity", "solve_seconds"]].mean().round(3)
    (out / "summary.md").write_text("# Benchmark (mean over test weeks)\n\n" + summ.to_markdown() + "\n")
    print(summ)


if __name__ == "__main__":
    main()
```
Add `tabulate` to dependencies (needed by `to_markdown`): `uv pip install tabulate` and add `"tabulate>=0.9"` to `pyproject.toml`.

- [ ] **Step 6: Smoke run on real data**

Run: `uv run python scripts/run_all.py --weeks 2 --skip-makespan`
Expected: two lines like `2022-xx-xx: ~40-60 employees x ~100-250 tasks`, then a summary table where `gap` has the lowest `hours_per_assigned_task`, `round_robin` the highest, and `gap_cap_*` sits between. If `gap` shows a high `n_unassigned`, go back to the Task 6 calibration note.

- [ ] **Step 7: Full run** — `uv run python scripts/run_all.py` (expect minutes, dominated by sensitivity re-solves and CP-SAT). Check that `results/*.csv` exist and are non-empty.

- [ ] **Step 8: Commit** — `git add src/wfo/experiments.py scripts/run_all.py tests/test_experiments.py pyproject.toml && git commit -m "feat: experiments and end-to-end runner"`

---

### Task 11: Figures

**Files:**
- Create: `src/wfo/plots.py`; add a `--plots-only` flag to `scripts/run_all.py` that reads the existing CSVs.

**Interfaces:**
- Consumes: `results/*.csv`.
- Produces: `make_figures(results_dir: str) -> list[str]` (paths written).

**Before writing any plotting code, invoke the `dataviz` skill** and follow its palette and form guidance.

Figures (one function each, all reading from CSV only):
1. **Benchmark bars:** mean hours per assigned task by method, with min–max whiskers across weeks.
2. **Balance vs. efficiency frontier:** x = Gini of load, y = total hours; one point per method, connecting `gap` → `gap_cap_3.0` → `gap_cap_2.0` → `gap_cap_1.5`. This tests the "cap restores balance at small cost" claim.
3. **Load distribution:** per-employee load histogram/strip for historical vs. gap vs. gap_cap_2.0 on one representative week.
4. **Noise sensitivity:** regret (% of clean objective) vs. σ, mean ± sd over draws and weeks.
5. **Capacity sensitivity:** total hours and unassigned count vs. capacity scale.
6. **Shadow prices:** top-15 employees by mean `dual_saving_per_hour`, with the `fd_saving_1h` points overlaid.
7. **Cross-training:** objective drop per candidate employee for the bottleneck category.
8. **Makespan:** GAP-plan vs. joint-optimal makespan per day (paired dots).
9. `workflow_map.png` (already produced in Task 10).

- [ ] **Step 1:** Write `tests/test_experiments.py::test_make_figures_from_csv` that writes tiny CSVs to `tmp_path` and asserts `make_figures` returns existing PNG paths.
- [ ] **Step 2:** Implement `plots.py` (matplotlib, `Agg` backend, 200 dpi, one function per figure, `make_figures` calls them all and skips any figure whose CSV is missing or empty).
- [ ] **Step 3:** `uv run pytest -q` → PASS. Then run `uv run python scripts/run_all.py --plots-only` and look at every PNG.
- [ ] **Step 4: Commit** — `git commit -am "feat: report figures"` (after `git add src/wfo/plots.py`).

---

### Task 12: Final report

**Files:**
- Create: `report/final_report.tex` (or reuse the interim report's LaTeX template), `report/figures/` (copied from `results/`)

Structure, mapped to the interim report's promises:

| Section | Source |
|---|---|
| Problem & data | Interim report + the "What the data actually looks like" table above |
| Workflow map & bottlenecks | `workflow_map.png`, `workflow_dwell.csv` (top statuses by total dwell share; rework rate = reopened visits) |
| Effort model & assumptions | Design decisions 2–3, 7; zero-active-time share from Task 3 Step 5 |
| MILP formulation | Interim formulation + backlog variable + share-cap constraint |
| Results vs. baselines | `summary.md`, figures 1–3; report hours per assigned task **and** unassigned count |
| Workload cap trade-off | Figure 2 |
| Sensitivity | Figures 4–5, `sens_min_turns.csv` |
| Managerial levers | Figures 6–7 (extra capacity hour, cross-training) |
| Extension: makespan | Figure 8, design decision 13 |
| Limitations | Wall-clock-derived effort proxy; assumed `base_c` scale; UTC business hours; turn-1 assignee as the routing decision; no quality dimension (the `issues_snapshot_sample.xlsx` manager scores could become a future quality constraint) |

- [ ] Draft each section, insert numbers from `results/`, compile, and read it end to end against the interim report's "Expected Results" paragraph. **Explicitly confirm or refute each of its three predictions:** (a) big gain over round-robin, (b) smaller gain over historical, (c) the cap restores balance at small cost.
- [ ] Commit the report source.

---

## Suggested team split (3 people)

| Person | Tasks | Notes |
|---|---|---|
| A | 1, 2, 3 | Data + effort model. Others depend on it, so start immediately |
| B | 6, 7, 8, 9 | Solvers + baselines + metrics. Can start on day 1 because all tests use synthetic matrices |
| C | 4, 5, then 11, 12 | Workflow map + instance builder, then figures and report |
| All | 10 | Integration once 3, 5, 6–9 are merged |

## Self-review notes

- **Spec coverage:** GAP MILP (T6); eligibility and capacity (T3, T5, T6); historical-trace workloads (T5); Gurobi or OR-Tools (T6 `solver` config); precedence, start times, non-overlap, makespan (T9); workload-share cap (T6, T10); baselines historical-proportion + round-robin (+ actual historical) (T7); objective, balance, utilization, solve time (T8, T10); sensitivity to processing times and capacity (T10); workflow map, dependencies, bottlenecks (T4); routing rules, cross-training, extra-hour value (T10, T11). Covered.
- **Known risks:** (1) The effort proxy may be weakly related to true effort, so lean on relative comparisons and the noise-sensitivity results. (2) The base_hours/capacity scale needs the Task 6 calibration check. (3) If CP-SAT does not reach optimality on 40-task batches within 30 s, cut `makespan_batch_size` to 20 and report the `FEASIBLE` status.
