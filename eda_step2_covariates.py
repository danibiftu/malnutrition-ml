"""
EDA Step 2: covariate distributions.
  - Categorical covariates (sex, residence, education, wealth, birth
    interval, region): bar charts, viridis-colored.
  - Continuous covariates (age, mother's BMI, birth order): summary
    statistics table instead of a plot.
"""
import os
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Change these two paths to match your own setup.
OUTDIR = r"D:\CAIMIS Research\Articles\Malnutrition"
DATA_PATH = r"D:\CAIMIS Research\Articles\Malnutrition\ml_malnutrition_clean.dta"
os.makedirs(OUTDIR, exist_ok=True)

viridis = mpl.colormaps["viridis"]
INK_PRIMARY, INK_MUTED, GRIDLINE = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK_PRIMARY,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})

df, meta = pyreadstat.read_dta(DATA_PATH)
value_labels = meta.variable_value_labels
for col, mapping in value_labels.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
n = len(df)

def vshade(t):
    return viridis(t)

# ---------------------------------------------------------------
# Continuous covariates: summary table (no plot)
# ---------------------------------------------------------------
continuous_vars = {
    "child_age_months": "Child's age (months)",
    "mother_bmi": "Mother's BMI (kg/m^2)",
    "birth_order": "Birth order",
}
summary = df[list(continuous_vars.keys())].describe().T
summary = summary.rename(index=continuous_vars)[["count", "mean", "std", "min", "50%", "max"]]
summary.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
summary = summary.round(2)
summary.to_csv(f"{OUTDIR}/eda2_continuous_summary.csv")
print("Continuous covariates \u2014 summary statistics:")
print(summary.to_string())
print(f"\nsaved eda2_continuous_summary.csv")

# ---------------------------------------------------------------
# Categorical covariates: bar charts, 2x3 grid
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

sex_order = ["Male", "Female"]
sex_counts = df["child_sex"].value_counts().reindex(sex_order)
axes[0,0].bar(sex_order, sex_counts.values / n * 100, color=[vshade(0.25), vshade(0.75)], width=0.55)
axes[0,0].set_title("Child's sex")
axes[0,0].set_ylabel("Share of children (%)")

res_order = ["Urban", "Rural"]
res_counts = df["residence"].value_counts().reindex(res_order)
axes[0,1].bar(res_order, res_counts.values / n * 100, color=[vshade(0.3), vshade(0.7)], width=0.55)
axes[0,1].set_title("Residence")

educ_order = ["No education", "Primary", "Secondary", "Higher"]
educ_counts = df["mother_education"].value_counts().reindex(educ_order)
axes[0,2].bar(educ_order, educ_counts.values / n * 100,
              color=[vshade(t) for t in np.linspace(0.15, 0.85, 4)], width=0.6)
axes[0,2].set_title("Mother's education")
axes[0,2].tick_params(axis="x", rotation=20)

wealth_order = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
wealth_counts = df["wealth_index"].value_counts().reindex(wealth_order)
axes[1,0].bar(wealth_order, wealth_counts.values / n * 100,
              color=[vshade(t) for t in np.linspace(0.1, 0.9, 5)], width=0.6)
axes[1,0].set_title("Household wealth index")
axes[1,0].set_ylabel("Share of children (%)")
axes[1,0].tick_params(axis="x", rotation=20)

interval_order = ["First birth", "<24 months", "24-47 months", "48+ months"]
interval_counts = df["birth_interval_cat"].value_counts().reindex(interval_order)
axes[1,1].bar(interval_order, interval_counts.values / n * 100,
              color=[vshade(t) for t in np.linspace(0.15, 0.85, 4)], width=0.6)
axes[1,1].set_title("Preceding birth interval")
axes[1,1].tick_params(axis="x", rotation=20)

axes[1,2].axis("off")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x")

fig.tight_layout()
fig.savefig(f"{OUTDIR}/eda2_covariate_distributions.png", dpi=300, facecolor="white", bbox_inches="tight")
print("saved eda2_covariate_distributions.png")

# ---------------------------------------------------------------
# Region (14 categories, horizontal bar)
# ---------------------------------------------------------------
region_counts = df["region"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
region_colors = [vshade(t) for t in np.linspace(0.1, 0.9, len(region_counts))]
ax.barh(region_counts.index, region_counts.values / n * 100, color=region_colors)
ax.set_xlabel("Share of children (%)")
ax.set_title("Region")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/eda2_region_distribution.png", dpi=300, facecolor="white", bbox_inches="tight")
print("saved eda2_region_distribution.png")
