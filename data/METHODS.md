**Methods (brief).**
- Data: MLB Statcast (pybaseball extract) 2021–2023; RosterResource injury lists (mapped by MLBAM ID).
- Labeling: pitcher-season injury = 1 if pitcher appears on that season’s injury list; pitch-level labels aggregated to season via max().
- Features: per-game workload & rest days; average & 95th-percentile velocity; average spin; pitch-mix proportions.
- Engineered: velocity delta (season vs prior-year or early-season), 14-day workload spike (max/median), breaking-usage delta (season–early).
- Split: train on 2021–2022, test on 2023 (no leakage).
- Models: Logistic Regression (balanced), XGBoost (scale_pos_weight); median imputation; standardization for LR; threshold tuning by ROC/PR analysis.