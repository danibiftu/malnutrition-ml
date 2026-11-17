# Cluster-validated, calibrated and explainable AI for child undernutrition severity

Analysis code for the manuscript "Beyond Prediction: Cluster-Validated, Calibrated and Explainable AI for Child Undernutrition Severity" .

The study classifies the severity of stunting, underweight and wasting (severe, moderate, normal) among children under five using the 2024/25 Ethiopia Demographic and Health Survey. It compares ordinal logistic regression, Random Forest, XGBoost, LightGBM, support vector machines and Frank-Hall ordinal variants with nested cross-validation grouped by survey cluster, corrects the probabilities distorted by SMOTENC oversampling (prior-shift adjustment), and assesses calibration, decision curves, sample size, survey-weighted sensitivity and out-of-fold SHAP explanations.

All code is in one file: `malnutrition_analysis.py`.

## Data
The Ethiopia DHS 2024/25 Children's Recode (`ETKR8AFL.dta`) is free of charge but requires registration and an approved request from the DHS Program (https://dhsprogram.com). The data cannot be redistributed and are not included in this repository.

## Setup
Python 3.11 or later.

```
pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost lightgbm statsmodels shap pyreadstat matplotlib
```

All random seeds are fixed (seed 42).

## Usage
Place `ETKR8AFL.dta` and `malnutrition_analysis.py` in one working directory and run the stages in this order:

```
python malnutrition_analysis.py extract ETKR8AFL.dta .     # builds the analytic data set (5,817 children, 33 model inputs)
python malnutrition_analysis.py cv stunting_cat            # nested cluster-grouped cross-validation (long run)
python malnutrition_analysis.py cv underweight_cat
python malnutrition_analysis.py cv wasting_cat
python malnutrition_analysis.py post                       # prior-shift correction, metrics, 95% cluster-bootstrap CIs
python malnutrition_analysis.py paired                     # final-model choice and paired contrasts
python malnutrition_analysis.py weighted                   # survey-weighted sensitivity of the final models
python malnutrition_analysis.py riley                      # Riley sample-size criteria
python malnutrition_analysis.py shap                       # out-of-fold SHAP for the final models
python malnutrition_analysis.py figures                    # figures and decision-curve summary
python malnutrition_analysis.py tables                     # LaTeX table fragments
```

Running `python malnutrition_analysis.py` with no argument prints this list. The `cv` stage takes several hours per outcome. `post` accepts the number of bootstrap replicates (default 1000), for example `post 200`.

Optional environment variables: `REV_DATA` (path to the analytic CSV, default `ml_malnutrition_analytic_rev.csv`) and `REV_OUTS` (comma-separated outcomes for `post` and `shap`).

## Main outputs
| Stage | Files written |
|---|---|
| extract | `ml_malnutrition_analytic_rev.csv`, `.dta`, `selection_measured_vs_not.csv` |
| cv | `nested_<outcome>.pkl` (out-of-fold probabilities) |
| post | `rev_preds.pkl`, `rev_metrics_ci.csv`, `rev_calibration.csv`, `rev_tuning_optimism.csv`, `rev_regional_weighted.csv` |
| paired | `rev_final.csv`, `rev_paired_contrasts.csv` |
| weighted, riley | `rev_weighted_final.csv`, `riley_sample_size.csv` |
| shap | `rev_shap_*.csv`, `rev_fig_shap_*.png` |
| figures, tables | `rev_fig_*.png`, `rev_dca_summary.csv`, `tab_*.tex` |

## Citation
Bekalo DB, Esenogho E. Beyond Prediction: Cluster-Validated, Calibrated and Explainable AI for Child Undernutrition Severity. 

## Contact
Daniel Biftu Bekalo, University of South Africa (UNISA), bekaldb@unisa.ac.za
