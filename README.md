# Cluster-validated, calibrated and explainable AI for child undernutrition severity

Analysis code for the manuscript "Beyond Prediction: Cluster-Validated, Calibrated and Explainable AI for Child Undernutrition Severity" (Informatics in Medicine Unlocked, IMU-D-26-00829).

## Data
The Ethiopia Demographic and Health Survey 2024/25 Children's Recode (`ETKR8AFL.dta`) is free but requires registration and an approved request from the DHS Program (https://dhsprogram.com). The data cannot be redistributed and are not included here; `.dta`, `.csv` and `.pkl` files are git-ignored.

## Setup
Python 3.11 or later; `pip install -r requirements.txt`. All random seeds are fixed (42).

## Run order (inside `analysis/`, with the data in the same folder)
1. `python extract_ml_data_rev.py ETKR8AFL.dta .`   builds `ml_malnutrition_analytic_rev.csv` (5,817 children, 33 model inputs) and `selection_measured_vs_not.csv`
2. `python rev_nested_cv.py stunting_cat`, then `underweight_cat`, then `wasting_cat`   nested cross-validation grouped by survey cluster, with SMOTENC, class-weight and unweighted variants and Frank-Hall models (writes `nested_<outcome>.pkl`)
3. `python rev_postprocess.py`   prior-shift correction, all metrics, cluster-bootstrap 95% CIs, calibration, survey-weighted prevalences (writes `rev_preds.pkl`, `rev_metrics_ci.csv` and other tables)
4. `python rev_paired.py`   final-model choice and paired contrasts (writes `rev_final.csv`)
5. `python rev_weighted_final.py`, `python rev_riley.py`   survey-weighted sensitivity and Riley sample-size criteria
6. `python rev_shap.py`   out-of-fold SHAP for the final models
7. `python rev_figures.py`, `python rev_tables.py`   figures, decision-curve summary and LaTeX table fragments

## Other folders
- `eda/`  exploratory figures (outcome distributions, predictor distributions, z-score pairs). Each script has a `DATA_PATH` line near the top that points to a local file; edit it to your copy of the analytic data before running.
- `original_submission_superseded/`  code of the first submission (single hold-out split, SMOTENC before evaluation). Kept for transparency only; its results are superseded by `analysis/`. These scripts contain local file paths that need editing before use.
