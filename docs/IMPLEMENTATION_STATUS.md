# Implementation and verification record

Task 1: complete — audited ingestion, first-turn join, categories, shrinkage, strict cutoffs, capacities.
Task 2: complete — M1/M2/M3 transportation LP, fairness and baselines, M4 MILP.
Task 3: complete — validation, scoring and E1–E9 in reproduction and strict scenarios.
Task 4: local artifacts complete — executed notebooks, visual overview, results narratives, tests and CI configuration. GitHub publication is in progress.

Ruling: used the existing dedicated workforce-opt directory as a new isolated repository because it already contained the actual project data and plans; did not edit any other repository.
Ruling: maintained both plan reproduction and stricter evaluation so changes in assumptions remain visible; results across modes are different scenarios, not a controlled one-factor experiment.
Ruling: selected k,z using validation rather than preliminary expected outcomes; reproduction selected z=0 and its improvement is shrinkage-only.
Ruling: used an already installed Python 3.11 numerical environment and a plain-Python notebook executor because this session cannot fetch dependencies; notebooks are valid nbformat 4 JSON and can also run in Jupyter.

Independent reviewer checked mathematical formulation and temporal validity. Fixed the material finding: strict validation oracle now excludes the 340 outcomes unresolved by the July 1, 2022 test-policy fitting date. Added and passed regression test, then reran both scenarios through notebook 02. All 30 cross-training runs returned solver status 0 and gap 0 on the local data. Missing assignment times (1,028 arrivals) are explicitly excluded from prior-roster inference.

Course-ready communication still requires the team to review and author the final technical report, presentation and video; this repository supplies the research implementation and evidence.
