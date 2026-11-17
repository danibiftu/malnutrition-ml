"""
Complete analysis code (single file) for:
  "Beyond Prediction: Cluster-Validated, Calibrated and Explainable AI for Child Undernutrition Severity"

Data: Ethiopia DHS 2024/25 Children's Recode (ETKR8AFL.dta), obtained from the DHS Program (https://dhsprogram.com).
Python >= 3.11; pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost lightgbm statsmodels shap pyreadstat matplotlib
All random seeds are fixed (SEED = 42).

Usage (run the stages in this order, in one working directory):
  python malnutrition_analysis.py extract  ETKR8AFL.dta .     # analytic data set (5,817 children, 33 inputs)
  python malnutrition_analysis.py cv stunting_cat             # nested cluster-grouped CV (long run), then
  python malnutrition_analysis.py cv underweight_cat          #   underweight_cat and wasting_cat
  python malnutrition_analysis.py cv wasting_cat
  python malnutrition_analysis.py post [B]                    # prior-shift correction, metrics, bootstrap CIs (B = 1000)
  python malnutrition_analysis.py paired                      # final-model choice and paired contrasts
  python malnutrition_analysis.py weighted                    # survey-weighted sensitivity of the final models
  python malnutrition_analysis.py riley                       # Riley sample-size criteria
  python malnutrition_analysis.py shap                        # out-of-fold SHAP of the final models
  python malnutrition_analysis.py figures                     # figures and decision-curve summary
  python malnutrition_analysis.py tables                      # LaTeX table fragments
Environment variables: REV_DATA (analytic CSV path), REV_OUTS (comma-separated outcomes for post/shap).
"""
import sys

# ============================== COMMON FUNCTIONS ==============================
import os, warnings
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings("ignore")

SEED = 42
OUTCOMES = ["stunting_cat", "underweight_cat", "wasting_cat"]
ONAME = {"stunting_cat": "Stunting", "underweight_cat": "Underweight", "wasting_cat": "Wasting"}
ORDER = {"Severe": 0, "Moderate": 1, "Normal": 2}
CLASSES = ["Severe", "Moderate", "Normal"]

DATA = os.environ.get("REV_DATA", "ml_malnutrition_analytic_rev.csv")

def load():
    df = pd.read_csv(DATA)
    return df

def build_features(df):
    """Return X (numeric, NaN allowed for continuous/binary), feature list, and
    SMOTENC group definitions. Categorical NaNs (<=1.2%) are set to the modal category."""
    X = pd.DataFrame(index=df.index)
    X["child_age_months"] = df["child_age_months"]
    X["mother_bmi"] = df["mother_bmi"]
    X["mother_height_cm"] = df["mother_height_cm"]
    X["birth_order"] = df["birth_order"]
    X["water_time_min"] = df["water_time_min"]
    X["mother_education_ord"] = df["mother_education"].map({"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3})
    X["wealth_index_ord"] = df["wealth_index"].map({"Poorest": 0, "Poorer": 1, "Middle": 2, "Richer": 3, "Richest": 4})
    X["child_sex_male"] = (df["child_sex"] == "Male").astype(int)
    X["residence_urban"] = (df["residence"] == "Urban").astype(int)
    X["diarrhoea_2wk"] = df["diarrhoea_2wk"]
    X["fever_2wk"] = df["fever_2wk"]
    cont = ["child_age_months", "mother_bmi", "mother_height_cm", "birth_order", "water_time_min",
            "mother_education_ord", "wealth_index_ord"]
    binary = ["child_sex_male", "residence_urban", "diarrhoea_2wk", "fever_2wk"]
    groups = {}
    for col, prefix, ref in [("birth_interval_cat", "birth_interval", "First birth"),
                             ("region", "region", None), ("interview_season", "season", None),
                             ("water_source", "water", None), ("sanitation", "sanitation", None)]:
        s = df[col].fillna(df[col].mode().iloc[0])
        cats = sorted(s.unique()) if ref is None else [ref] + sorted(c for c in s.unique() if c != ref)
        if col == "region": cats = ["Addis Ababa"] + [c for c in cats if c != "Addis Ababa"]
        if col == "interview_season": cats = ["Bega (Oct-Jan)", "Kiremt (Jun-Sep)", "Belg (Feb-May)"]
        if col == "water_source": cats = ["Improved", "Unimproved", "Surface water"]
        if col == "sanitation": cats = ["Improved", "Unimproved", "Open defecation"]
        cols = []
        for c in cats[1:]:
            name = f"{prefix}_" + c.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("<", "lt").replace("+", "plus")
            X[name] = (s == c).astype(int); cols.append(name)
        groups[prefix] = cols
    return X, cont, binary, groups

def impute_fold(Xtr, Xte, cont, binary):
    fill = Xtr[cont + binary].median()
    return Xtr.fillna(fill), Xte.fillna(fill)

def smotenc_resample(Xp, yp, cont, binary, groups, seed=SEED):
    """SMOTENC with each one-hot variable collapsed to one categorical code (so every
    synthetic child has exactly one category per variable), expanded back afterwards."""
    counts = Counter(yp); k = min(5, min(counts.values()) - 1)
    if k < 1: return Xp, yp
    Z = Xp[cont + binary].copy()
    for g, cols in groups.items():
        sub = Xp[cols]
        Z[g + "_code"] = np.where(sub.sum(axis=1) == 0, 0, sub.values.argmax(axis=1) + 1)
    cat_idx = [Z.columns.get_loc(c) for c in binary + [g + "_code" for g in groups]]
    sm = SMOTENC(categorical_features=cat_idx, random_state=seed, k_neighbors=k)
    Zr, yr = sm.fit_resample(Z, yp)
    out = Zr[cont + binary].copy()
    for g, cols in groups.items():
        for j, c in enumerate(cols):
            out[c] = (Zr[g + "_code"].values == j + 1).astype(int)
    return out[Xp.columns], np.asarray(yr)

GRIDS = {
    "Random Forest": {"n_estimators": [200, 400, 600], "max_depth": [4, 6, 8, 12, None],
                      "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.5]},
    "XGBoost": {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.02, 0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10]},
    "LightGBM": {"n_estimators": [100, 200, 300, 500], "max_depth": [3, 4, 5, -1],
                 "learning_rate": [0.02, 0.03, 0.05, 0.1], "num_leaves": [7, 15, 31, 63],
                 "min_child_samples": [10, 20, 50], "colsample_bytree": [0.6, 0.8, 1.0]},
    "SVM": {"C": [0.1, 0.5, 1, 5, 10], "gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
}

def make_est(name, params=None, binary=False, class_weight=None, n_jobs=2):
    params = dict(params or {})
    if name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, class_weight=class_weight, **params)
    if name == "XGBoost":
        kw = dict(objective="binary:logistic") if binary else dict(objective="multi:softprob")
        return XGBClassifier(random_state=SEED, verbosity=0, n_jobs=n_jobs, tree_method="hist", **kw, **params)
    if name == "LightGBM":
        kw = dict(objective="binary") if binary else dict(objective="multiclass")
        return LGBMClassifier(random_state=SEED, verbosity=-1, n_jobs=n_jobs, class_weight=class_weight, **kw, **params)
    if name == "SVM":
        return SVC(random_state=SEED, kernel="rbf", probability=True, class_weight=class_weight, **params)
    raise ValueError(name)

def balanced_weights(y):
    cnt = Counter(y); n = len(y); k = len(cnt)
    return np.array([n / (k * cnt[v]) for v in y])

def fit_predict(name, params, Xtr, ytr, Xte, variant="smote", scale=None):
    """Fit one learner and return class probabilities (columns 0,1,2) on Xte.
    variant: 'smote' (training data already resampled by caller), 'cw' (class weights), 'none'."""
    Xa, Xb = Xtr, Xte
    if name == "SVM":
        sc = StandardScaler().fit(Xtr); Xa, Xb = sc.transform(Xtr), sc.transform(Xte)
    est = make_est(name, params, class_weight="balanced" if (variant == "cw" and name != "XGBoost") else None)
    if name == "XGBoost" and variant == "cw":
        est.fit(Xa, ytr, sample_weight=balanced_weights(ytr))
    else:
        est.fit(Xa, ytr)
    return est.predict_proba(Xb), est

def fit_predict_olr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = OrderedModel(np.asarray(ytr), sc.transform(Xtr), distr="logit")
    res = m.fit(method="bfgs", maxiter=500, disp=False)
    return res.model.predict(res.params, exog=sc.transform(Xte)), res, sc

def prior_shift(P, prior_train, prior_target):
    """Prior-shift (Saerens/Elkan) correction: reweight class probabilities by the ratio of
    target to training class priors and renormalise."""
    w = np.asarray(prior_target) / np.asarray(prior_train)
    Q = P * w
    return Q / Q.sum(axis=1, keepdims=True)


# ============================== METRIC FUNCTIONS ==============================
import numpy as np
from scipy.stats import rankdata
def auc_rank(s, pos):
    n1 = pos.sum(); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def ovr_auc(P, y):
    return np.nanmean([auc_rank(P[:, k], y == k) for k in range(3)])

def c_index(P, y):
    s = 2 * P[:, 2] + P[:, 1]; num = den = 0
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        m = (y == a) | (y == b); na, nb = (y == a).sum(), (y == b).sum()
        if na == 0 or nb == 0: continue
        num += auc_rank(s[m], y[m] == b) * na * nb; den += na * nb
    return num / den

def cls_metrics(P, y):
    pred = P.argmax(1); cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
    acc = np.trace(cm) / cm.sum()
    tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0).mean()
    W = (np.subtract.outer(range(3), range(3)) ** 2) / 4.0
    E = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    qwk = 1 - (W * cm).sum() / (W * E).sum()
    return acc, f1, qwk, rec

def brier(P, y):
    Y = np.eye(3)[y]; return ((P - Y) ** 2).mean()          # mean over classes and children

def brier_ref(y):
    pi = np.bincount(y, minlength=3) / len(y); return (pi * (1 - pi)).mean()


R = sys.modules[__name__]   # scripts below refer to the common functions as R.<name>


# ============================== STAGES ==============================

def stage_extract(RAW_PATH="ETKR8AFL.dta", OUTDIR="."):
    """
    REVISION (IMU-D-26-00829): extract the expanded analytic dataset from the raw
    EDHS 2024/25 Children's Recode (KR file, ETKR8AFL.dta).

    Changes relative to extract_ml_data.py
      * keeps survey-design variables: cluster (v001), household (v002 -> hhid),
        PSU (v021), strata (v022), normalised child weight (v005/1e6)
      * adds maternal height (v438), interview season (v006), diarrhoea (h11),
        fever (h22) and WASH (drinking-water source v113, sanitation v116,
        time to water v115)
      * writes a table comparing children with / without an anthropometric
        measurement (selection into the 5,926-child analytic subsample)

    Usage:  python extract_ml_data_rev.py  [RAW_DTA]  [OUTDIR]
    """
    import sys, os
    import numpy as np
    import pandas as pd
    import pyreadstat

    os.makedirs(OUTDIR, exist_ok=True)
    OUT_DTA = f"{OUTDIR}/ml_malnutrition_analytic_rev.dta"
    OUT_CSV = f"{OUTDIR}/ml_malnutrition_analytic_rev.csv"

    df, meta = pyreadstat.read_dta(RAW_PATH)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")

    # ---------------------------------------------------------------
    # 1. Analytic sample: children with a valid measurement (HW13 == 0)
    # ---------------------------------------------------------------
    def clean_z(col):
        z = num(col)
        return z.where(z < 9994) / 100.0

    measured = df["hw13"] == 0
    haz, waz, whz = clean_z("hw70"), clean_z("hw71"), clean_z("hw72")
    keep = measured & haz.notna() & waz.notna() & whz.notna()
    print(f"Children in KR file: {len(df)}")
    print(f"Measured (HW13==0): {measured.sum()};  with 3 valid z-scores: {keep.sum()}")

    # ---------------------------------------------------------------
    # 2. Selection table: measured vs not measured (all living children)
    # ---------------------------------------------------------------
    def status(r):
        if r["b5"] == 0: return "Child dead"
        return {0: "Measured", 3: "Not present", 4: "Refused", 6: "Other",
                7: "No measurement found in household"}.get(r["hw13"], "Missing")
    sel = df[["b5", "hw13"]].copy()
    sel["status"] = sel.apply(status, axis=1)
    sel["analytic"] = keep.values
    sel["child_age_yrs"] = num("b8")
    sel["male"] = (num("b4") == 1).astype(int)
    sel["urban"] = (num("v025") == 1).astype(int)
    sel["educ_none"] = (num("v106") == 0).astype(int)
    sel["poorest_poorer"] = num("v190").isin([1, 2]).astype(int)
    sel["bord"] = num("bord")
    sel["region"] = num("v024")
    sel_alive = sel[sel["b5"] == 1]
    rows = []
    for name, g in [("Analytic (measured)", sel_alive[sel_alive.analytic]),
                    ("Alive, not in analytic sample", sel_alive[~sel_alive.analytic])]:
        rows.append({"Group": name, "n": len(g),
                     "Mean age (yrs)": g.child_age_yrs.mean(), "% male": 100 * g.male.mean(),
                     "% urban": 100 * g.urban.mean(), "% mother no education": 100 * g.educ_none.mean(),
                     "% poorest/poorer": 100 * g.poorest_poorer.mean(), "Mean birth order": g.bord.mean()})
    sel_tab = pd.DataFrame(rows).round(2)
    sel_tab.to_csv(f"{OUTDIR}/selection_measured_vs_not.csv", index=False)
    sel.groupby("status").size().rename("n").to_csv(f"{OUTDIR}/hw13_status_counts.csv")
    print(sel_tab.to_string(index=False))
    reg = pd.crosstab(sel_alive.region, sel_alive.analytic, normalize="index").mul(100).round(1)
    reg.columns = ["% not in sample", "% in sample"]
    reg.to_csv(f"{OUTDIR}/selection_by_region.csv")

    # ---------------------------------------------------------------
    # 3. Build the analytic frame
    # ---------------------------------------------------------------
    d = pd.DataFrame(index=df.index[keep])
    k = lambda c: df.loc[keep, c]
    kn = lambda c: num(c)[keep]

    def to_category(z):
        cat = pd.Series(np.select([z < -3, (z >= -3) & (z < -2), z >= -2], [1, 2, 3], default=np.nan), index=z.index)
        return pd.Categorical(cat, categories=[1, 2, 3], ordered=True).rename_categories(
            {1: "Severe", 2: "Moderate", 3: "Normal"})

    d["haz"], d["waz"], d["whz"] = haz[keep], waz[keep], whz[keep]
    d["stunting_cat"] = to_category(d["haz"])
    d["underweight_cat"] = to_category(d["waz"])
    d["wasting_cat"] = to_category(d["whz"])

    # --- survey design ---
    d["cluster_id"] = kn("v001").astype(int)
    d["psu"] = kn("v021").astype(int)
    d["strata"] = kn("v022").astype(int)
    d["hh_id"] = (kn("v001") * 10000 + kn("v002")).astype(int)
    d["svy_weight"] = kn("v005") / 1e6

    # --- original covariates ---
    d["child_age_months"] = kn("hw1")
    d["child_sex"] = kn("b4").map({1: "Male", 2: "Female"}).astype("category")
    d["mother_education"] = pd.Categorical(kn("v106").map({0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"}),
        categories=["No education", "Primary", "Secondary", "Higher"], ordered=True)
    bmi = kn("v445"); d["mother_bmi"] = bmi.where(bmi < 9990) / 100.0
    d["wealth_index"] = pd.Categorical(kn("v190").map({1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}),
        categories=["Poorest", "Poorer", "Middle", "Richer", "Richest"], ordered=True)
    d["birth_order"] = kn("bord")
    bord, b11 = kn("bord"), kn("b11")
    def bic(bo, iv):
        if bo == 1: return "First birth"
        if pd.isna(iv): return np.nan
        return "<24 months" if iv < 24 else ("24-47 months" if iv < 48 else "48+ months")
    d["birth_interval_cat"] = pd.Categorical([bic(a, b) for a, b in zip(bord, b11)],
        categories=["First birth", "<24 months", "24-47 months", "48+ months"], ordered=False)
    d["breastfeeding_status"] = pd.Categorical(kn("m4").map({93: "Ever breastfed, stopped", 94: "Never breastfed", 95: "Still breastfeeding"}),
        categories=["Never breastfed", "Ever breastfed, stopped", "Still breastfeeding"], ordered=False)
    anc = kn("m14"); d["anc_visits"] = anc.where(anc < 90)
    def dp(code):
        if pd.isna(code): return np.nan
        c = int(code)
        if c in (10, 11, 12): return "Home"
        if c in (20, 21, 22, 23, 26): return "Public facility"
        if c in (30, 31, 32, 36): return "Private facility"
        if c in (40, 41, 46): return "NGO facility"
        return "Other"
    d["delivery_place"] = pd.Categorical(kn("m15").map(dp),
        categories=["Home", "Public facility", "Private facility", "NGO facility", "Other"], ordered=False)
    d["residence"] = kn("v025").map({1: "Urban", 2: "Rural"}).astype("category")
    region_map = {1: "Tigray", 2: "Afar", 3: "Amhara", 4: "Oromia", 5: "Somali", 6: "Benishangul-Gumuz",
                  7: "Central Ethiopia", 8: "Sidama", 9: "South West Ethiopia", 10: "South Ethiopia",
                  12: "Gambella", 13: "Harari", 14: "Addis Ababa", 15: "Dire Dawa"}
    d["region"] = pd.Categorical(kn("v024").map(region_map), categories=list(region_map.values()), ordered=False)

    # --- NEW: maternal height (cm; codes >= 9990 = missing; implausible <120 cm set missing) ---
    h = kn("v438"); h = h.where(h < 9990) / 10.0
    n_implaus = int((h < 120).sum()); h = h.where(~(h < 120))
    d["mother_height_cm"] = h
    print(f"Maternal height: missing/implausible set to NaN (<120 cm: {n_implaus})")

    # --- NEW: interview season. v006/v007 are Gregorian (fieldwork Apr 2024 - Mar 2025; v008 CMC check) ---
    m = kn("v006")
    season = np.select([m.isin([6, 7, 8, 9]), m.isin([10, 11, 12, 1]), m.isin([2, 3, 4, 5])],
                       ["Kiremt (Jun-Sep)", "Bega (Oct-Jan)", "Belg (Feb-May)"], default="")
    d["interview_season"] = pd.Categorical(np.where(season == "", None, season),
        categories=["Bega (Oct-Jan)", "Kiremt (Jun-Sep)", "Belg (Feb-May)"], ordered=False)

    # --- NEW: morbidity (h11 diarrhoea in last 2 weeks; h22 fever in last 2 weeks; 8 = don't know -> NaN) ---
    d["diarrhoea_2wk"] = kn("h11").map({0: 0, 1: 1, 2: 1}).astype(float)
    d["fever_2wk"] = kn("h22").map({0: 0, 1: 1}).astype(float)

    # --- NEW: WASH (JMP-style grouping; 97 'not de jure resident' -> NaN) ---
    water_map = {**{c: "Improved" for c in (11, 12, 13, 14, 21, 31, 41, 51, 61, 62, 71, 72)},
                 **{c: "Unimproved" for c in (32, 42, 96)}, 43: "Surface water"}
    d["water_source"] = pd.Categorical(kn("v113").map(water_map),
        categories=["Improved", "Unimproved", "Surface water"], ordered=False)
    toilet_map = {**{c: "Improved" for c in (11, 12, 13, 21, 22, 41)},
                  **{c: "Unimproved" for c in (14, 15, 23, 42, 43, 96)}, 31: "Open defecation"}
    d["sanitation"] = pd.Categorical(kn("v116").map(toilet_map),
        categories=["Improved", "Unimproved", "Open defecation"], ordered=False)
    tw = kn("v115")
    d["water_time_min"] = tw.where(~tw.isin([997, 998])).replace(996, 0)  # 996 = water on premises

    d = d.reset_index(drop=True)

    # ---------------------------------------------------------------
    # 4. Save (Stata 14 format) + missingness report
    # ---------------------------------------------------------------
    labels = {"haz": "Height-for-age z-score", "waz": "Weight-for-age z-score", "whz": "Weight-for-height z-score",
              "stunting_cat": "Stunting (HAZ) category", "underweight_cat": "Underweight (WAZ) category",
              "wasting_cat": "Wasting (WHZ) category", "cluster_id": "Cluster (v001)", "psu": "PSU (v021)",
              "strata": "Sample strata (v022)", "hh_id": "Household id (v001*10000+v002)",
              "svy_weight": "Sample weight (v005/1e6)", "child_age_months": "Child age, months",
              "child_sex": "Child sex", "mother_education": "Maternal education", "mother_bmi": "Maternal BMI",
              "wealth_index": "Wealth quintile", "birth_order": "Birth order", "birth_interval_cat": "Birth interval",
              "breastfeeding_status": "Breastfeeding status", "anc_visits": "ANC visits",
              "delivery_place": "Place of delivery", "residence": "Residence", "region": "Region",
              "mother_height_cm": "Maternal height, cm", "interview_season": "Season of interview",
              "diarrhoea_2wk": "Diarrhoea in last 2 weeks", "fever_2wk": "Fever in last 2 weeks",
              "water_source": "Drinking-water source", "sanitation": "Sanitation facility",
              "water_time_min": "Minutes to water source (0 = on premises)"}
    pyreadstat.write_dta(d, OUT_DTA, column_labels=[labels.get(c, c) for c in d.columns], version=14)
    d.to_csv(OUT_CSV, index=False)
    miss = (d.isna().mean() * 100).round(2)
    miss[miss > 0].rename("% missing").to_csv(f"{OUTDIR}/missingness_rev.csv")
    print(f"\nFinal analytic sample: n = {len(d)}; clusters = {d.cluster_id.nunique()}; households = {d.hh_id.nunique()}")
    print("\nColumns with missing values (%):"); print(miss[miss > 0].to_string())
    print("\nNew-variable distributions:")
    for c in ["interview_season", "water_source", "sanitation"]: print(d[c].value_counts(dropna=False).to_string(), "\n")
    print(d[["mother_height_cm", "water_time_min", "diarrhoea_2wk", "fever_2wk"]].describe().round(2).to_string())


def stage_cv(outcome):
    """
    Cluster-grouped NESTED cross-validation for one outcome.

    Outer loop : 5-fold StratifiedGroupKFold (groups = EDHS cluster, so siblings / neighbours
                 never straddle train and test).  Every child is predicted exactly once by a model
                 whose training data and hyper-parameter search never saw the child's cluster.
    Inner loop : 3-fold StratifiedGroupKFold inside each outer-training set; SMOTENC is applied to
                 the inner-training part only (validation folds keep the natural class mix);
                 random search (n_iter configs) maximising macro one-vs-rest AUC.

    Per outer fold and outcome we store out-of-fold class probabilities for
      OLR (non-resampled), RF/XGB/LGBM/SVM (SMOTENC; main), Frank-Hall ordinal RF/XGB/LGBM (SMOTENC),
      and, for the three tree learners, class-weight (no SMOTENC) and unweighted variants.
    Also stored: inner-CV best AUC (for optimism), chosen hyper-parameters, and training-fold priors.
    Usage: python rev_nested_cv.py stunting_cat
    """
    import sys, time, pickle, json
    import numpy as np, pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
    from sklearn.metrics import roc_auc_score
    N_ITER = {"Random Forest": 12, "XGBoost": 12, "LightGBM": 12, "SVM": 6}
    OUT = f"nested_{outcome}.pkl"

    df = R.load(); X, cont, binary, groups = R.build_features(df)
    y = df[outcome].map(R.ORDER).values; g = df["cluster_id"].values
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=R.SEED).split(X, y, g))

    def auc_ovr(yt, P):
        """Macro one-vs-rest AUC; works for any score matrix (probabilities or SVM decision values)."""
        return float(np.mean([roc_auc_score((yt == k).astype(int), P[:, k]) for k in range(3)]))

    def tune(name, Xtr, ytr, gtr):
        """Random search with SMOTENC inside each inner training split; returns best params + inner AUC."""
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=R.SEED + 1).split(Xtr, ytr, gtr))
        prepared = []
        for itr, iva in inner:
            Xa, Xb = R.impute_fold(Xtr.iloc[itr], Xtr.iloc[iva], cont, binary)
            Xr, yr = R.smotenc_resample(Xa, ytr[itr], cont, binary, groups)
            prepared.append((Xr, yr, Xb, ytr[iva]))
        best, best_s = None, -1
        for p in ParameterSampler(R.GRIDS[name], n_iter=N_ITER[name], random_state=R.SEED):
            sc = []
            for Xr, yr, Xb, yb in prepared:
                if name == "SVM":   # no Platt scaling during search: rank with decision_function (cheaper)
                    from sklearn.svm import SVC; from sklearn.preprocessing import StandardScaler
                    s = StandardScaler().fit(Xr)
                    m = SVC(kernel="rbf", decision_function_shape="ovr", random_state=R.SEED, **p).fit(s.transform(Xr), yr)
                    sc.append(auc_ovr(yb, m.decision_function(s.transform(Xb))))
                else:
                    P, _ = R.fit_predict(name, p, Xr, yr, Xb); sc.append(auc_ovr(yb, P))
            if np.mean(sc) > best_s: best, best_s = p, float(np.mean(sc))
        return best, best_s

    def fh_predict(name, params, Xr, yr, Xte):
        """Frank & Hall (2001) ordinal decomposition: P(Y>0) and P(Y>1) from two binary learners."""
        cum = []
        for k in (0, 1):
            est = R.make_est(name, params, binary=True)
            if name == "SVM": raise ValueError
            est.fit(Xr, (yr > k).astype(int)); cum.append(est.predict_proba(Xte)[:, 1])
        p1, p2 = cum                       # P(Y>=moderate), P(Y=normal)
        p2 = np.minimum(p2, p1)            # enforce monotone cumulative probabilities
        P = np.column_stack([1 - p1, p1 - p2, p2])
        return P / P.sum(axis=1, keepdims=True)

    res = {"outcome": outcome, "folds": [], "y": y, "cluster": g, "outer_test_idx": [te for _, te in outer]}
    oof = {}      # key -> n x 3 array
    def put(key, te, P):
        if key not in oof: oof[key] = np.full((len(y), 3), np.nan)
        oof[key][te] = P

    t0 = time.time()
    for f, (tr, te) in enumerate(outer):
        Xtr_raw, Xte_raw = X.iloc[tr], X.iloc[te]
        Xtr, Xte = R.impute_fold(Xtr_raw, Xte_raw, cont, binary)
        ytr, yte = y[tr], y[te]
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        Xr, yr = R.smotenc_resample(Xtr, ytr, cont, binary, groups)
        fold = {"fold": f, "n_train": len(tr), "n_test": len(te), "prior_train": prior.tolist(),
                "prior_resampled": (np.bincount(yr, minlength=3) / len(yr)).tolist(), "params": {}, "inner_auc": {}}
        # OLR on the ORIGINAL (non-resampled) data
        P, _, _ = R.fit_predict_olr(Xtr, ytr, Xte); put("OLR", te, P)
        for name in ["Random Forest", "XGBoost", "LightGBM", "SVM"]:
            p, s = tune(name, Xtr_raw, ytr, g[tr])
            fold["params"][name] = p; fold["inner_auc"][name] = s
            P, _ = R.fit_predict(name, p, Xr, yr, Xte); put(name, te, P)       # SMOTENC (main)
            if name != "SVM":
                put("FH-" + name, te, fh_predict(name, p, Xr, yr, Xte))       # Frank-Hall ordinal
                P, _ = R.fit_predict(name, p, Xtr, ytr, Xte, variant="cw"); put(name + " [class-weight]", te, P)
                P, _ = R.fit_predict(name, p, Xtr, ytr, Xte, variant="none"); put(name + " [unweighted]", te, P)
            print(f"[{outcome}] fold {f} {name:14s} inner AUC={s:.3f} outer AUC={auc_ovr(yte, oof[name][te]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        res["folds"].append(fold)
        res["oof"] = oof
        pickle.dump(res, open(OUT, "wb"))
    print("done", time.time() - t0)


def stage_post(B=1000):
    """
    Post-processing of the nested-CV out-of-fold predictions:
      * prior-shift correction (fold-wise, resampled prior -> training-fold prior)
      * discrimination / classification metrics with 95% cluster-bootstrap CIs
      * Brier score and Brier skill vs the uninformative (prevalence-only) reference
      * calibration slope / intercept (severe class), tuning optimism, paired AUC differences
      * survey-weighted sensitivity metrics; design-based regional prevalence
    Run after all three nested_*.pkl exist.   python rev_postprocess.py [B]
    """
    import sys, pickle, json
    import numpy as np, pandas as pd
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(R.SEED)
    df = R.load()
    MAIN = ["OLR", "Random Forest", "XGBoost", "LightGBM", "SVM"]
    FH = ["FH-Random Forest", "FH-XGBoost", "FH-LightGBM"]
    VARS = [m + s for m in ["Random Forest", "XGBoost", "LightGBM"] for s in [" [class-weight]", " [unweighted]"]]
    ALL = MAIN + FH + VARS
    NEEDS_CORR = lambda k: k not in ("OLR",) and not k.endswith("[unweighted]")

    def auc_rank(s, pos):
        n1 = pos.sum(); n0 = len(pos) - n1
        if n1 == 0 or n0 == 0: return np.nan
        r = rankdata(s); return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    def ovr_auc(P, y):
        return np.nanmean([auc_rank(P[:, k], y == k) for k in range(3)])

    def c_index(P, y):
        s = 2 * P[:, 2] + P[:, 1]; num = den = 0
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            m = (y == a) | (y == b); na, nb = (y == a).sum(), (y == b).sum()
            if na == 0 or nb == 0: continue
            num += auc_rank(s[m], y[m] == b) * na * nb; den += na * nb
        return num / den

    def cls_metrics(P, y):
        pred = P.argmax(1); cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
        acc = np.trace(cm) / cm.sum()
        tp = np.diag(cm); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0).mean()
        W = (np.subtract.outer(range(3), range(3)) ** 2) / 4.0
        E = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
        qwk = 1 - (W * cm).sum() / (W * E).sum()
        return acc, f1, qwk, rec

    def brier(P, y):
        Y = np.eye(3)[y]; return ((P - Y) ** 2).mean()          # mean over classes and children

    def brier_ref(y):
        pi = np.bincount(y, minlength=3) / len(y); return (pi * (1 - pi)).mean()

    def calib_sev(p, ysev):
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        lr = LogisticRegression(penalty=None, max_iter=500).fit(z[:, None], ysev)
        lr0 = LogisticRegression(penalty=None, max_iter=500, fit_intercept=True)
        # calibration-in-the-large with slope fixed at 1: offset model via Newton on intercept only
        a = 0.0
        for _ in range(50):
            q = 1 / (1 + np.exp(-(z + a))); a += (ysev - q).sum() / max((q * (1 - q)).sum(), 1e-9)
        return lr.coef_[0, 0], lr.intercept_[0], a

    def metric_row(P, Pc, y, Pb=None):
        Pb = P if Pb is None else Pb
        acc, f1, qwk, rec = cls_metrics(Pb, y)
        return {"Accuracy": acc, "Macro-F1": f1, "QWK": qwk, "AUC": ovr_auc(P, y), "AUC_corr": ovr_auc(Pc, y),
                "C-index": c_index(P, y), "Brier_raw": brier(P, y), "Brier_corr": brier(Pc, y),
                "BSS_raw": 1 - brier(P, y) / brier_ref(y), "BSS_corr": 1 - brier(Pc, y) / brier_ref(y),
                "Recall_severe": rec[0], "Recall_moderate": rec[1], "Recall_normal": rec[2]}

    out_main, out_boot, out_opt, out_diff, out_cal, out_w, out_prior = [], [], [], [], [], [], []
    preds = {}
    import os
    OUTS = os.environ.get('REV_OUTS', ','.join(R.OUTCOMES)).split(',')
    for o in OUTS:
        res = pickle.load(open(f"nested_{o}.pkl", "rb")); y = res["y"]; g = res["cluster"]
        assert len(res["folds"]) == 5, f"{o}: nested CV not finished"
        oof = res["oof"]; corr = {}; bal = {}
        for k in ALL:
            P = oof[k]; Pc = P.copy(); Pb = P.copy()
            for f, te in enumerate(res["outer_test_idx"]):
                fd = res["folds"][f]
                if NEEDS_CORR(k): Pc[te] = R.prior_shift(P[te], fd["prior_resampled"], fd["prior_train"])
                else: Pb[te] = R.prior_shift(P[te], fd["prior_train"], [1/3] * 3)   # natural-prior models -> balanced decision rule
            corr[k] = Pc; bal[k] = Pb
        preds[o] = {"y": y, "raw": oof, "corr": corr, "bal": bal, "cluster": g, "folds": res["folds"]}
        # --- point estimates ---
        pts = {k: metric_row(oof[k], corr[k], y, bal[k]) for k in ALL}
        # --- cluster bootstrap ---
        uc = np.unique(g); idx_by_c = {c: np.where(g == c)[0] for c in uc}
        boots = {k: [] for k in ALL}; bsel = []
        for b in range(B):
            samp = rng.choice(uc, len(uc), replace=True)
            ii = np.concatenate([idx_by_c[c] for c in samp]); yb = y[ii]
            if len(np.unique(yb)) < 3: continue
            for k in ALL: boots[k].append(metric_row(oof[k][ii], corr[k][ii], yb, bal[k][ii]))
        for k in ALL:
            bd = pd.DataFrame(boots[k]); row = {"Outcome": R.ONAME[o], "Model": k}
            for m in bd.columns:
                row[m] = pts[k][m]; row[m + "_lo"] = bd[m].quantile(0.025); row[m + "_hi"] = bd[m].quantile(0.975)
            out_boot.append(row)
        # --- fold-level, tuning optimism, selection ---
        fold_auc = {k: [ovr_auc(oof[k][te], y[te]) for te in res["outer_test_idx"]] for k in MAIN}
        for k in MAIN[1:]:
            inner = np.mean([f["inner_auc"][k] for f in res["folds"]])
            out_opt.append({"Outcome": R.ONAME[o], "Model": k, "Inner-CV AUC (tuning, SMOTENC-in-fold)": inner,
                            "Outer-fold mean AUC": np.mean(fold_auc[k]), "Outer-fold SD": np.std(fold_auc[k], ddof=1),
                            "Optimism (inner - outer)": inner - np.mean(fold_auc[k])})
        sel = max(MAIN, key=lambda k: pts[k]["AUC"])
        runner = sorted(MAIN, key=lambda k: -pts[k]["AUC"])
        for k in MAIN:
            if k == sel: continue
            d = np.array([a["AUC"] for a in boots[sel]]) - np.array([a["AUC"] for a in boots[k]])
            out_diff.append({"Outcome": R.ONAME[o], "Selected": sel, "Comparator": k, "AUC diff": pts[sel]["AUC"] - pts[k]["AUC"],
                             "lo": np.quantile(d, .025), "hi": np.quantile(d, .975), "CI excludes 0": bool(np.quantile(d, .025) > 0)})
        out_prior.append({"Outcome": R.ONAME[o], "Selected (nested-CV AUC)": sel,
                          "Ranking": " > ".join(f"{k} ({pts[k]['AUC']:.3f})" for k in runner),
                          "Resampled prior": np.round(res["folds"][0]["prior_resampled"], 3).tolist(),
                          "Natural prior (fold 1)": np.round(res["folds"][0]["prior_train"], 3).tolist()})
        # --- calibration of severe class (selected model) ---
        for k in [sel] + [v for v in VARS if v.startswith(sel)] + ["FH-" + sel] if sel != "OLR" else [sel]:
            for version, PP in [("raw (SMOTENC-trained)" if NEEDS_CORR(k) else "as fitted", oof[k]), ("prior-shift corrected", corr[k])]:
                if version == "prior-shift corrected" and not NEEDS_CORR(k): continue
                sl, ic, cil = calib_sev(PP[:, 0], (y == 0).astype(int))
                out_cal.append({"Outcome": R.ONAME[o], "Model": k, "Version": version,
                                "Mean predicted P(severe)": PP[:, 0].mean(), "Observed P(severe)": (y == 0).mean(),
                                "Calibration slope": sl, "Calibration intercept": ic, "Calibration-in-the-large": cil,
                                "Brier (all classes)": brier(PP, y), "Brier ref": brier_ref(y),
                                "BSS": 1 - brier(PP, y) / brier_ref(y)})
        # --- survey-weighted sensitivity ---
        w = df["svy_weight"].values
        for k in MAIN:
            row = {"Outcome": R.ONAME[o], "Model": k,
                   "AUC unweighted": pts[k]["AUC"],
                   "AUC weighted": np.mean([roc_auc_score((y == c).astype(int), oof[k][:, c], sample_weight=w) for c in range(3)])}
            Pc = corr[k]; Y = np.eye(3)[y]
            row["Brier_corr unweighted"] = pts[k]["Brier_corr"]
            row["Brier_corr weighted"] = np.average(((Pc - Y) ** 2).mean(1), weights=w)
            pi_w = np.array([w[y == c].sum() for c in range(3)]) / w.sum()
            row["Brier ref weighted"] = (pi_w * (1 - pi_w)).mean()
            row["BSS weighted"] = 1 - row["Brier_corr weighted"] / row["Brier ref weighted"]
            out_w.append(row)

    pd.DataFrame(out_boot).to_csv("rev_metrics_ci.csv", index=False)
    pd.DataFrame(out_opt).to_csv("rev_tuning_optimism.csv", index=False)
    pd.DataFrame(out_diff).to_csv("rev_auc_diff_vs_selected.csv", index=False)
    pd.DataFrame(out_cal).to_csv("rev_calibration.csv", index=False)
    pd.DataFrame(out_w).to_csv("rev_weighted_sensitivity.csv", index=False)
    pd.DataFrame(out_prior).to_csv("rev_selection.csv", index=False)
    pickle.dump(preds, open("rev_preds.pkl", "wb"))

    # --- design-based prevalence by region (Taylor linearisation, PSU within strata) ---
    def svy_prop(ind, w, psu, strata):
        p = (w * ind).sum() / w.sum(); z = w * (ind - p) / w.sum()
        d = pd.DataFrame({"z": z, "psu": psu, "h": strata}).groupby(["h", "psu"]).z.sum().reset_index()
        var = 0
        for h, s in d.groupby("h"):
            n = len(s); var += n / (n - 1) * ((s.z - s.z.mean()) ** 2).sum() if n > 1 else 0
        return p, np.sqrt(var)
    rows = []
    for o in R.OUTCOMES:
        ind_all = (df[o] == "Severe").astype(float).values
        for lab, ind in [("Severe", ind_all), ("Severe or moderate", (df[o] != "Normal").astype(float).values)]:
            for reg in ["All"] + sorted(df.region.unique()):
                m = np.ones(len(df), bool) if reg == "All" else (df.region == reg).values
                p, se = svy_prop(ind[m], df.svy_weight.values[m], df.psu.values[m], df.strata.values[m])
                rows.append({"Outcome": R.ONAME[o], "Level": lab, "Region": reg, "n": int(m.sum()),
                             "Unweighted %": 100 * ind[m].mean(), "Weighted %": 100 * p,
                             "Weighted 95% CI lo": 100 * (p - 1.96 * se), "Weighted 95% CI hi": 100 * (p + 1.96 * se),
                             "Share of sample %": 100 * m.mean(), "Weighted share %": 100 * df.svy_weight.values[m].sum() / df.svy_weight.sum()})
    pd.DataFrame(rows).to_csv("rev_regional_weighted.csv", index=False)
    print("postprocess complete")


def stage_paired():
    """
    Final model choice and paired cluster-bootstrap contrasts (uses rev_preds.pkl).
    Final model per outcome = tree learner trained WITHOUT resampling (unweighted) with the highest out-of-fold macro AUC.
    Contrasts: final vs OLR; final vs same learner with SMOTENC (corrected) and with class weights.
    """
    import pickle, numpy as np, pandas as pd
    P = pickle.load(open("rev_preds.pkl", "rb")); rng = np.random.default_rng(R.SEED + 7)
    rows, final = [], []
    for o in R.OUTCOMES:
        d = P[o]; y = d["y"]; g = d["cluster"]; nm = R.ONAME[o]
        cands = [f"{m} [unweighted]" for m in ["Random Forest", "XGBoost", "LightGBM"]]
        best = max(cands, key=lambda k: ovr_auc(d["corr"][k], y)); learner = best.replace(" [unweighted]", "")
        final.append({"Outcome": nm, "Final model": best, "Learner": learner, "AUC": ovr_auc(d["corr"][best], y)})
        comps = {"OLR": "OLR", f"{learner} (SMOTENC, corrected)": learner, f"{learner} (class weights, corrected)": learner + " [class-weight]"}
        uc = np.unique(g); idx = {c: np.where(g == c)[0] for c in uc}
        dif = {k: {"AUC": [], "C": [], "BSS": []} for k in comps}
        for _ in range(1000):
            ii = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]); yb = y[ii]
            if len(np.unique(yb)) < 3: continue
            fb = d["corr"][best][ii]; a, c_, b_ = ovr_auc(fb, yb), c_index(fb, yb), 1 - brier(fb, yb) / brier_ref(yb)
            for lab, k in comps.items():
                pk = d["corr"][k][ii]
                dif[lab]["AUC"].append(a - ovr_auc(pk, yb)); dif[lab]["C"].append(c_ - c_index(pk, yb)); dif[lab]["BSS"].append(b_ - (1 - brier(pk, yb) / brier_ref(yb)))
        for lab, k in comps.items():
            pt = {"AUC": ovr_auc(d["corr"][best], y) - ovr_auc(d["corr"][k], y), "C": c_index(d["corr"][best], y) - c_index(d["corr"][k], y),
                  "BSS": (1 - brier(d["corr"][best], y) / brier_ref(y)) - (1 - brier(d["corr"][k], y) / brier_ref(y))}
            r = {"Outcome": nm, "Final model": best, "Comparator": lab}
            for m in ("AUC", "C", "BSS"): r[m + " diff"] = pt[m]; r[m + " lo"] = np.quantile(dif[lab][m], .025); r[m + " hi"] = np.quantile(dif[lab][m], .975)
            rows.append(r)
    pd.DataFrame(final).to_csv("rev_final.csv", index=False); pd.DataFrame(rows).to_csv("rev_paired_contrasts.csv", index=False)
    print(pd.DataFrame(final)); print(pd.DataFrame(rows).round(3).to_string())


def stage_weighted():
    """
    Survey-weighted AUC / Brier skill for OLR and the final model of each outcome (sensitivity analysis).
    """
    import pickle, numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score
    P = pickle.load(open("rev_preds.pkl", "rb")); df = R.load(); w = df.svy_weight.values; fin = pd.read_csv("rev_final.csv").set_index("Outcome")
    rows = []
    for o in R.OUTCOMES:
        nm = R.ONAME[o]; y = P[o]["y"]; Y = np.eye(3)[y]
        pi = np.bincount(y, minlength=3) / len(y); piw = np.array([w[y == c].sum() for c in range(3)]) / w.sum()
        for lab, k in [("OLR", "OLR"), (fin.loc[nm, "Final model"].replace("[unweighted]", "(no resampling)"), fin.loc[nm, "Final model"])]:
            Pc = P[o]["corr"][k]; b = ((Pc - Y) ** 2).mean(1)
            rows.append({"Outcome": nm, "Model": lab,
                         "AUC unweighted": np.mean([roc_auc_score(y == c, Pc[:, c]) for c in range(3)]),
                         "AUC weighted": np.mean([roc_auc_score(y == c, Pc[:, c], sample_weight=w) for c in range(3)]),
                         "BSS unweighted": 1 - b.mean() / (pi * (1 - pi)).mean(),
                         "BSS weighted": 1 - np.average(b, weights=w) / (piw * (1 - piw)).mean()})
    pd.DataFrame(rows).to_csv("rev_weighted_final.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())


def stage_riley():
    """
    Riley et al. (2020, Stat Med 'Calculating the sample size required for developing a clinical
    prediction model' / BMJ 2020;368:m441) criteria 1-3 for binary outcomes, applied to the two
    cumulative binary contrasts underlying each 3-category outcome (Frank-Hall decomposition).
    P = 33 candidate predictor parameters. Anticipated Cox-Snell R2 = 15% of the maximum (Riley default),
    with 5% and 30% as sensitivity values.
    """
    import numpy as np, pandas as pd
    df = R.load(); P = 33; N = len(df)
    rows = []
    for o in R.OUTCOMES:
        c = df[o].value_counts(normalize=True)
        for label, phi in [("Severe vs. moderate/normal", c["Severe"]),
                           ("Severe/moderate vs. normal", 1 - c["Normal"])]:
            maxr2 = 1 - np.exp(2 * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)))
            for frac in (0.05, 0.15, 0.30):
                r2 = frac * maxr2; S = 0.9
                n1 = P / ((S - 1) * np.log(1 - r2 / S))
                svh = r2 / (r2 + 0.05 * maxr2)
                n2 = P / ((svh - 1) * np.log(1 - r2 / svh))
                n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
                n = max(n1, n2, n3)
                rows.append({"Outcome": R.ONAME[o], "Contrast": label, "Prevalence": round(phi, 4),
                             "Events in sample": int(round(phi * N)), "Max R2_CS": round(maxr2, 3),
                             "Anticipated R2_CS": round(r2, 3), "R2 as % of max": int(frac * 100),
                             "n crit1 (shrinkage)": int(np.ceil(n1)), "n crit2 (optimism)": int(np.ceil(n2)),
                             "n crit3 (risk precision)": int(np.ceil(n3)), "n required": int(np.ceil(n)),
                             "Events required": int(np.ceil(n * phi)), "EPP required": round(n * phi / P, 1),
                             "Actual EPP": round(phi * N / P, 1), "Sample size met": N >= n})
    t = pd.DataFrame(rows); t.to_csv("riley_sample_size.csv", index=False)
    print(t[t["R2 as % of max"] == 15][["Outcome", "Contrast", "Prevalence", "Events in sample", "n required", "Events required", "Actual EPP", "Sample size met"]].to_string(index=False))
    print(); print(t[["Outcome", "Contrast", "R2 as % of max", "n required", "Sample size met"]].to_string(index=False))


def stage_shap():
    """
    Out-of-fold SHAP for the selected model of each outcome.
    For every outer fold of the nested CV, the selected learner is refitted (that fold's tuned
    hyper-parameters, SMOTENC on the fold's training children) and SHAP values are computed on the fold's
    held-out children, so every child's explanation comes from a model that never saw their cluster.
    """
    import pickle, sys, numpy as np, pandas as pd, shap
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fin = pd.read_csv("rev_final.csv"); SEL = dict(zip(fin.Outcome, fin.Learner))
    df = R.load(); X, cont, binary, groups = R.build_features(df); feats = X.columns.tolist()
    NICE = {"child_age_months": "Child age (months)", "mother_bmi": "Maternal BMI", "mother_height_cm": "Maternal height (cm)",
            "birth_order": "Birth order", "water_time_min": "Minutes to water", "mother_education_ord": "Maternal education",
            "wealth_index_ord": "Household wealth", "child_sex_male": "Male sex", "residence_urban": "Urban residence",
            "diarrhoea_2wk": "Diarrhoea (2 wk)", "fever_2wk": "Fever (2 wk)"}
    nice = lambda f: NICE.get(f, f.replace("lt24", "<24").replace("48plus", "48+").replace("24_47", "24-47").replace("Jun_Sep", "Jun-Sep").replace("Oct_Jan", "Oct-Jan").replace("Feb_May", "Feb-May").replace("_", " ").replace("region ", "Region: ").replace("birth interval ", "Birth interval: ")
                               .replace("season ", "Season: ").replace("water ", "Water: ").replace("sanitation ", "Sanitation: "))
    def split_shap(sv):
        sv = np.asarray(sv) if not isinstance(sv, list) else np.stack(sv, axis=-1)
        return sv if sv.shape[1] == len(feats) else np.transpose(sv, (1, 2, 0))   # -> (n, features, classes)

    import os, glob
    OUTS = os.environ.get('REV_OUTS', ','.join(R.OUTCOMES)).split(',')
    allrows = []
    for o in OUTS:
        name = R.ONAME[o]; learner = SEL[name]
        if learner in ("OLR", "SVM"):
            cim = pd.read_csv("rev_metrics_ci.csv"); cim = cim[(cim.Outcome == name) & cim.Model.isin(["Random Forest", "XGBoost", "LightGBM"])]
            learner = cim.sort_values("AUC", ascending=False).Model.iloc[0]; print(f"{name}: selected model is not tree-based; SHAP for best tree model {learner}")
        res = pickle.load(open(f"nested_{o}.pkl", "rb")); y = res["y"]
        SV = np.zeros((len(y), len(feats), 3)); Xi = pd.DataFrame(index=X.index, columns=feats, dtype=float)
        cached = os.environ.get("REV_REPLOT") and os.path.exists(f"rev_shap_{o}.pkl")
        if cached:
            c = pickle.load(open(f"rev_shap_{o}.pkl", "rb")); SV, Xi = c["shap"], c["X"]
        for f, te in ([] if cached else enumerate(res["outer_test_idx"])):
            tr = np.setdiff1d(np.arange(len(y)), te)
            Xtr, Xte = R.impute_fold(X.iloc[tr], X.iloc[te], cont, binary)
            est = R.make_est(learner, res["folds"][f]["params"][learner]); est.fit(Xtr, y[tr])
            SV[te] = split_shap(shap.TreeExplainer(est).shap_values(Xte)); Xi.iloc[te] = Xte.values
            print(name, "fold", f, "shap done", flush=True)
        if not cached: pickle.dump({"shap": SV, "X": Xi, "learner": learner, "y": y}, open(f"rev_shap_{o}.pkl", "wb"))
        imp = pd.DataFrame(np.abs(SV).mean(0), index=feats, columns=R.CLASSES); imp["Mean over classes"] = imp.mean(1)
        imp = imp.sort_values("Mean over classes", ascending=False); imp.insert(0, "Outcome", name); imp.insert(1, "Model", learner)
        imp.to_csv(f"rev_shap_importance_{o}.csv"); allrows.append(imp)
        # bar plot
        top = imp.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 5.5)); left = np.zeros(len(top))
        for c, col in zip(R.CLASSES, [plt.cm.viridis(.05), plt.cm.viridis(.45), plt.cm.viridis(.85)]):
            ax.barh([nice(f) for f in top.index], top[c], left=left, color=col, label=c); left += top[c].values
        ax.set_xlabel("Mean |SHAP| (out-of-fold)"); ax.set_title(f"{name} ({learner})"); ax.legend(); plt.tight_layout()
        plt.savefig(f"rev_fig_shap_bar_{o}.png", dpi=300); plt.close()
        # beeswarm, severe class
        plt.figure(); shap.summary_plot(SV[:, :, 0], Xi.astype(float).rename(columns=nice), max_display=15, show=False)
        plt.title(f"{name} ({learner}): severe class"); plt.tight_layout(); plt.savefig(f"rev_fig_shap_swarm_{o}.png", dpi=300, bbox_inches="tight"); plt.close()
        # dependence: top-3 continuous/ordinal predictors
        cands = [f for f in imp.index if f in ("child_age_months", "mother_bmi", "mother_height_cm", "birth_order", "water_time_min", "mother_education_ord", "wealth_index_ord")][:3]
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for ax, f in zip(axs, cands):
            j = feats.index(f); ax.scatter(Xi[f].astype(float), SV[:, j, 0], s=6, alpha=.4, c=plt.cm.viridis(.3))
            ax.axhline(0, c="grey", lw=.8); ax.set_xlabel(nice(f)); ax.set_ylabel("SHAP (severe class)")
        fig.suptitle(f"{name} ({learner})"); plt.tight_layout(); plt.savefig(f"rev_fig_shap_dep_{o}.png", dpi=300); plt.close()
        # binned mean SHAP (severe class)
        b = []
        d = pd.DataFrame({"age": Xi["child_age_months"].astype(float), "wealth": Xi["wealth_index_ord"].astype(float), "bmi": Xi["mother_bmi"].astype(float),
                          "bord": Xi["birth_order"].astype(float), "ht": Xi["mother_height_cm"].astype(float)})
        for v, bins, labs in [("age", [-1, 5, 11, 23, 35, 47, 60], ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59"]),
                              ("wealth", [-1, 0, 1, 2, 3, 4], ["Poorest", "Poorer", "Middle", "Richer", "Richest"]),
                              ("bmi", [0, 18.5, 20, 22, 25, 99], ["<18.5", "18.5-19.99", "20-21.99", "22-24.99", ">=25"]),
                              ("bord", [0, 1, 2, 4, 6, 20], ["1", "2", "3-4", "5-6", "7+"]),
                              ("ht", [0, 150, 155, 160, 165, 250], ["<150", "150-154.9", "155-159.9", "160-164.9", ">=165"])]:
            fj = {"age": "child_age_months", "wealth": "wealth_index_ord", "bmi": "mother_bmi", "bord": "birth_order", "ht": "mother_height_cm"}[v]
            cat = pd.cut(d[v], bins, labels=labs); s = pd.Series(SV[:, feats.index(fj), 0])
            for l in labs: b.append({"Outcome": name, "Variable": nice(fj), "Bin": l, "n": int((cat == l).sum()), "Mean SHAP (severe)": s[(cat == l).values].mean()})
        pd.DataFrame(b).to_csv(f"rev_shap_binned_{o}.csv", index=False)
    pd.concat([pd.read_csv(f"rev_shap_importance_{o}.csv", index_col=0) for o in R.OUTCOMES]).to_csv("rev_shap_importance_all.csv")
    print("shap done")


def stage_figures():
    """
    Figures + DCA tables from nested-CV out-of-fold predictions (rev_preds.pkl). Run after rev_postprocess.py.
    """
    import pickle, numpy as np, pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib as mpl
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.calibration import calibration_curve
    vir = mpl.colormaps["viridis"]; COL = [vir(0.05), vir(0.45), vir(0.85)]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    P = pickle.load(open("rev_preds.pkl", "rb")); ci = pd.read_csv("rev_metrics_ci.csv"); fin = pd.read_csv("rev_final.csv"); SEL = dict(zip(fin.Outcome, fin["Final model"])); LRN = dict(zip(fin.Outcome, fin.Learner))
    rng = np.random.default_rng(R.SEED)

    # ---- 1. model comparison (AUC and C-index with 95% CI) ----
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex="col")
    models = ["OLR", "Random Forest", "XGBoost", "LightGBM", "SVM", "FH-Random Forest", "FH-XGBoost", "FH-LightGBM", "Random Forest [unweighted]", "XGBoost [unweighted]", "LightGBM [unweighted]"]
    for j, o in enumerate(R.OUTCOMES):
        for i, (m, lab) in enumerate([("AUC", "Macro one-vs-rest AUC"), ("C-index", "Ordinal C-index")]):
            ax = axs[i, j]; d = ci[ci.Outcome == R.ONAME[o]].set_index("Model").loc[models]
            y = np.arange(len(models))[::-1]
            ax.errorbar(d[m], y, xerr=[d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]], fmt="o", color=vir(0.3), capsize=3)
            ax.axvline(0.5, ls=":", c="grey"); ax.set_yticks(y); ax.set_yticklabels([m.replace("[unweighted]","(no resampling)").replace("FH-","Frank-Hall ") + ("" if "(" in m or m in ("OLR",) or "[" in m else " (SMOTENC)") if m!="OLR" else "OLR" for m in models] if j == 0 else [])
            ax.set_title(R.ONAME[o] if i == 0 else ""); ax.set_xlabel(lab if i == 1 or True else "")
    plt.tight_layout(); plt.savefig("rev_fig_comparison.png", dpi=300); plt.close()

    # ---- 2. confusion matrices & ROC (selected model, pooled out-of-fold, SMOTENC decision rule) ----
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, o in zip(axs, R.OUTCOMES):
        y = P[o]["y"]; k = SEL[R.ONAME[o]]; pred = P[o]["bal"][k].argmax(1)
        cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3); pc = 100 * cm / cm.sum(1, keepdims=True)
        ax.imshow(pc, cmap="viridis", vmin=0, vmax=100)
        for a in range(3):
            for b in range(3): ax.text(b, a, f"{cm[a,b]}\n({pc[a,b]:.1f}%)", ha="center", va="center", color="white" if pc[a, b] < 60 else "black", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(R.CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(R.CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(f"{R.ONAME[o]} ({k.replace(' [unweighted]', ', no resampling')})")
    plt.tight_layout(); plt.savefig("rev_fig_confusion.png", dpi=300); plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, o in zip(axs, R.OUTCOMES):
        y = P[o]["y"]; k = SEL[R.ONAME[o]]; Pm = P[o]["raw"][k]
        for c in range(3):
            fpr, tpr, _ = roc_curve(y == c, Pm[:, c]); ax.plot(fpr, tpr, c=COL[c], lw=2, label=f"{R.CLASSES[c]} (AUC {roc_auc_score(y==c, Pm[:,c]):.2f})")
        ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"{R.ONAME[o]} ({k.replace(' [unweighted]', ', no resampling')})"); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig("rev_fig_roc.png", dpi=300); plt.close()

    # ---- 3. calibration: SMOTENC raw / corrected / final (selected learner) ----
    fig, axs = plt.subplots(3, 3, figsize=(13, 12))
    for j, o in enumerate(R.OUTCOMES):
        y = P[o]["y"]; nm = R.ONAME[o]; k = SEL[nm]; lr = LRN[nm]
        for i, (ver, Pm) in enumerate([(f"{lr}, SMOTENC (uncorrected)", P[o]["raw"][lr]), (f"{lr}, SMOTENC (prior-shift corrected)", P[o]["corr"][lr]), (f"{lr}, no resampling", P[o]["corr"][k])]):
            ax = axs[i, j]
            for c in range(3):
                ft, mp = calibration_curve(y == c, Pm[:, c], n_bins=10, strategy="quantile")
                ax.plot(mp, ft, "o-", c=COL[c], label=R.CLASSES[c])
            ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
            ax.set_title(f"{nm}: {ver}", fontsize=10); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("rev_fig_calibration.png", dpi=300); plt.close()

    # ---- 4. decision curves on recalibrated probabilities ----
    TH = np.round(np.arange(0.01, 0.801, 0.01), 2)
    def nb(y, s, t):
        n = len(y); flag = s[:, None] >= t[None, :]
        tp = (flag & (y[:, None] == 1)).sum(0); fp = (flag & (y[:, None] == 0)).sum(0)
        return tp / n - fp / n * t / (1 - t)
    def nb_all(y, t): pr = y.mean(); return pr - (1 - pr) * t / (1 - t)
    rows = []; fig, axs = plt.subplots(3, 2, figsize=(11, 12))
    for i, o in enumerate(R.OUTCOMES):
        y = P[o]["y"]; k = SEL[R.ONAME[o]]; lr = LRN[R.ONAME[o]]; g = P[o]["cluster"]; uc = np.unique(g); idx = {c: np.where(g == c)[0] for c in uc}
        for j, (lab, yy, s_raw, s_cor) in enumerate([
            ("Severe vs. rest", (y == 0).astype(int), P[o]["raw"][lr][:, 0], P[o]["corr"][k][:, 0]),
            ("Any malnutrition vs. normal", (y != 2).astype(int), 1 - P[o]["raw"][lr][:, 2], 1 - P[o]["corr"][k][:, 2])]):
            nm, nr, na = nb(yy, s_cor, TH), nb(yy, s_raw, TH), nb_all(yy, TH); best = np.maximum(na, 0); gain = nm - best
            ax = axs[i, j]; ax.plot(TH, nr, c="grey", lw=1.5, alpha=.6, label="SMOTENC, uncorrected")
            ax.plot(TH, nb(yy, P[o]["corr"][lr][:, 0] if j == 0 else 1 - P[o]["corr"][lr][:, 2], TH), c=vir(0.6), lw=1.5, ls="-.", label="SMOTENC, corrected"); ax.plot(TH, nm, c=vir(0.3), lw=2.5, label="No resampling"); ax.plot(TH, na, "--", c=vir(0.85), label="Treat all"); ax.axhline(0, ls=":", c="k", label="Treat none")
            ax.set_ylim(-0.05, max(yy.mean() * 1.05, .05)); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
            ax.set_title(f"{R.ONAME[o]} ({k.replace(' [unweighted]', ', no resampling')}): {lab}\nprevalence {100*yy.mean():.2f}%", fontsize=10)
            if i == 0 and j == 0: ax.legend(fontsize=8)
            t_star = int(np.argmax(gain)); useful = TH[gain > 0.0005]
            # bootstrap CI for gain at the best threshold
            bg = []
            for _ in range(500):
                ii = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]); yb = yy[ii]
                t1 = TH[t_star:t_star + 1]; bg.append((nb(yb, s_cor[ii], t1) - np.maximum(nb_all(yb, t1), 0))[0])
            rows.append({"Outcome": R.ONAME[o], "Model": k, "Framing": lab, "Prevalence": yy.mean(),
                         "Max gain (corrected)": gain[t_star], "at threshold": TH[t_star], "gain 95% CI lo": np.quantile(bg, .025), "gain 95% CI hi": np.quantile(bg, .975),
                         "Max gain (SMOTENC uncorrected)": (nr - best).max(), "at threshold (unc.)": TH[int(np.argmax(nr - best))],
                         "Thresholds with gain>0.0005": f"{useful.min():.2f}-{useful.max():.2f}" if len(useful) else "none"})
    plt.tight_layout(); plt.savefig("rev_fig_dca.png", dpi=300); plt.close()
    pd.DataFrame(rows).to_csv("rev_dca_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def stage_tables():
    """
    Generate LaTeX table fragments from the revision result CSVs.
    """
    import pandas as pd, numpy as np
    ci = pd.read_csv("rev_metrics_ci.csv"); f2 = lambda x: f"{x:.2f}"
    def est(r, m): return f"{r[m]:.2f} ({r[m+'_lo']:.2f}--{r[m+'_hi']:.2f})"
    MAIN = ["OLR", "Random Forest", "XGBoost", "LightGBM", "SVM", "FH-Random Forest", "FH-XGBoost", "FH-LightGBM", "Random Forest [unweighted]", "XGBoost [unweighted]", "LightGBM [unweighted]"]
    LAB = {"Random Forest [unweighted]": "Random Forest, no resampling", "XGBoost [unweighted]": "XGBoost, no resampling", "LightGBM [unweighted]": "LightGBM, no resampling", "Random Forest": "Random Forest, SMOTENC", "XGBoost": "XGBoost, SMOTENC", "LightGBM": "LightGBM, SMOTENC", "SVM": "SVM, SMOTENC", "FH-Random Forest": "Frank--Hall RF, SMOTENC", "FH-XGBoost": "Frank--Hall XGBoost, SMOTENC", "FH-LightGBM": "Frank--Hall LightGBM, SMOTENC"}
    # ---- main table ----
    L = ["\\begin{tabularx}{\\textwidth}{ll*{5}{>{\\centering\\arraybackslash}X}}", "\\toprule",
         "Outcome & Model & Accuracy & Macro-F1 & QWK & Macro AUC & C-index \\\\", "\\midrule"]
    for i, o in enumerate(R.OUTCOMES):
        d = ci[ci.Outcome == R.ONAME[o]].set_index("Model")
        for j, m in enumerate(MAIN):
            r = d.loc[m]; pre = f"\\multirow{{{len(MAIN)}}}{{*}}{{{R.ONAME[o]}}}" if j == 0 else ""
            L.append(f"{pre} & {LAB.get(m, m)} & " + " & ".join(est(r, x) for x in ["Accuracy", "Macro-F1", "QWK", "AUC", "C-index"]) + " \\\\")
        L.append("\\midrule" if i < 2 else "\\bottomrule")
    L.append("\\end{tabularx}"); open("tab_main.tex", "w").write("\n".join(L))
    # ---- probability quality: selected + variants ----
    sel = pd.read_csv("rev_selection.csv"); cal = pd.read_csv("rev_calibration.csv")
    L = ["\\begin{tabularx}{\\textwidth}{ll*{5}{>{\\centering\\arraybackslash}X}}", "\\toprule",
         "Outcome & Model and training & Macro AUC & Brier (as fitted) & Brier (corrected) & Brier skill (corrected) & Severe-class calibration slope \\\\", "\\midrule"]
    for i, o in enumerate(R.OUTCOMES):
        nm = R.ONAME[o]; d = ci[ci.Outcome == nm].set_index("Model")
        s = sel[sel.Outcome == nm]["Selected (nested-CV AUC)"].iloc[0]
        tree = pd.read_csv("rev_final.csv").set_index("Outcome").loc[nm, "Learner"]
        rows = [("OLR", "OLR, no resampling")] + [(tree, f"{tree}, SMOTENC"), (tree + " [class-weight]", f"{tree}, class weights"), (tree + " [unweighted]", f"{tree}, unweighted")]
        import pickle; P = pickle.load(open("rev_preds.pkl", "rb"))[o]; y = P["y"]
        from sklearn.linear_model import LogisticRegression
        for j, (k, lab) in enumerate(rows):
            r = d.loc[k]
            def slope(Pm):
                p = np.clip(Pm[:, 0], 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
                return LogisticRegression(penalty=None, max_iter=500).fit(z[:, None], (y == 0).astype(int)).coef_[0, 0]
            pre = f"\\multirow{{{len(rows)}}}{{*}}{{{nm}}}" if j == 0 else ""
            L.append(f"{pre} & {lab} & {est(r,'AUC_corr')} & {f2(r['Brier_raw'])} & {est(r,'Brier_corr')} & {est(r,'BSS_corr')} & {f2(slope(P['corr'][k]))} \\\\")
        L.append("\\midrule" if i < 2 else "\\bottomrule")
    L.append("\\end{tabularx}"); open("tab_prob.tex", "w").write("\n".join(L))
    # ---- tuning optimism + AUC differences ----
    opt = pd.read_csv("rev_tuning_optimism.csv"); df = pd.read_csv("rev_auc_diff_vs_selected.csv")
    L = ["\\begin{tabular}{llccc}", "\\toprule", "Outcome & Model & Inner-loop AUC & Outer-loop AUC & Optimism \\\\", "\\midrule"]
    for o in R.OUTCOMES:
        for j, (_, r) in enumerate(opt[opt.Outcome == R.ONAME[o]].iterrows()):
            L.append(f"{R.ONAME[o] if j==0 else ''} & {r['Model']} & {f2(r.iloc[2])} & {f2(r.iloc[3])} & {r.iloc[5]:+.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open("tab_opt.tex", "w").write("\n".join(L))
    pc = pd.read_csv("rev_paired_contrasts.csv")
    L = ["\\begin{tabular}{lllccc}", "\\toprule", "Outcome & Final model & Comparator & AUC difference & C-index difference & Brier skill difference \\\\", "\\midrule"]
    fm = lambda r, k: f"{r[k+' diff']:.2f} ({r[k+' lo']:.2f} to {r[k+' hi']:.2f})"
    for o in R.OUTCOMES:
        for j, (_, r) in enumerate(pc[pc.Outcome == R.ONAME[o]].iterrows()):
            comp = r['Comparator'].replace('(SMOTENC, corrected)', 'with SMOTENC').replace('(class weights, corrected)', 'with class weights')
            L.append(f"{R.ONAME[o] if j==0 else ''} & {r['Final model'].replace(' [unweighted]', ', no resampling') if j==0 else ''} & {comp} & {fm(r,'AUC')} & {fm(r,'C')} & {fm(r,'BSS')} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open("tab_diff.tex", "w").write("\n".join(L))
    # ---- Riley ----
    rl = pd.read_csv("riley_sample_size.csv"); rl = rl[rl["R2 as % of max"] == 15]
    L = ["\\begin{tabular}{llrrrrl}", "\\toprule", "Outcome & Contrast & Events & Required $n$ & Required events & Events per parameter & Criteria met \\\\", "\\midrule"]
    for _, r in rl.iterrows():
        L.append(f"{r['Outcome']} & {r['Contrast'].replace('Severe/moderate','Moderate or severe').replace('moderate/normal','other')} & {r['Events in sample']:,} & {r['n required']:,} & {r['Events required']:,} & {r['Actual EPP']:.2f} & {'Yes' if r['Sample size met'] else 'No'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open("tab_riley.tex", "w").write("\n".join(L))
    # ---- selection (S1) ----
    s = pd.read_csv("selection_measured_vs_not.csv")
    L = ["\\begin{tabular}{lrrrrrrr}", "\\toprule", "Group & $n$ & Age (years) & Boys (\\%) & Urban (\\%) & No education (\\%) & Two lowest quintiles (\\%) & Birth order \\\\", "\\midrule"]
    for _, r in s.iterrows():
        L.append(f"{r.iloc[0].replace('Analytic (measured)','Valid measurement (analytic sample)').replace('Alive, not in analytic sample','Living, not in analytic sample')} & {r.iloc[1]:,} & " + " & ".join(f"{v:.2f}" for v in r.iloc[2:]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open("tab_sel.tex", "w").write("\n".join(L))
    # ---- weighted sensitivity ----
    w = pd.read_csv("rev_weighted_final.csv")
    L = ["\\begin{tabular}{llcccc}", "\\toprule", "Outcome & Model & AUC unweighted & AUC weighted & Brier skill unweighted & Brier skill weighted \\\\", "\\midrule"]
    for o in R.OUTCOMES:
        for j, (_, r) in enumerate(w[w.Outcome == R.ONAME[o]].iterrows()):
            L.append(f"{R.ONAME[o] if j==0 else ''} & {r['Model']} & {f2(r['AUC unweighted'])} & {f2(r['AUC weighted'])} & {f2(r['BSS unweighted'])} & {f2(r['BSS weighted'])} \\\\")
        L.append("\\midrule" if o != R.OUTCOMES[-1] else "\\bottomrule")
    L += ["\\end{tabular}"]; open("tab_weighted.tex", "w").write("\n".join(L))
    # ---- weighted regional prevalence (severe) ----
    rg = pd.read_csv("rev_regional_weighted.csv")
    L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Region & Share of sample (\\%) & Share weighted (\\%) & Stunted, weighted \\% (95\\% CI) & Underweight, weighted \\% (95\\% CI) \\\\", "\\midrule"]
    for reg in ["All"] + sorted(set(rg.Region) - {"All"}):
        a = rg[(rg.Region == reg) & (rg.Level == "Severe or moderate")].set_index("Outcome")
        L.append(f"{reg} & {a.loc['Stunting','Share of sample %']:.2f} & {a.loc['Stunting','Weighted share %']:.2f} & " + " & ".join(f"{a.loc[x,'Weighted %']:.2f} ({a.loc[x,'Weighted 95% CI lo']:.2f}--{a.loc[x,'Weighted 95% CI hi']:.2f})" for x in ["Stunting", "Underweight"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]; open("tab_region.tex", "w").write("\n".join(L))
    print("tables written")



if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "extract": stage_extract(*(a[1:3]))
    elif cmd == "cv": stage_cv(a[1])
    elif cmd == "post": stage_post(int(a[1]) if len(a) > 1 else 1000)
    elif cmd in ("paired", "weighted", "riley", "shap", "figures", "tables"): globals()["stage_" + cmd]()
    else: sys.exit("unknown stage: " + cmd + "\n" + __doc__)
