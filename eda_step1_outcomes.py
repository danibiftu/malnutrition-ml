"""
EDA Step 1: distribution of the three malnutrition outcomes (stunting,
underweight, wasting), viridis-colored.
"""
import os
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

outcomes = ["stunting_cat", "underweight_cat", "wasting_cat"]
names = ["Stunting", "Underweight", "Wasting"]
cat_order = ["Severe", "Moderate", "Normal"]
cat_colors = [viridis(0.12), viridis(0.55), viridis(0.90)]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
for ax, col, name in zip(axes, outcomes, names):
    counts = df[col].value_counts().reindex(cat_order)
    props = counts / counts.sum()
    ax.bar(cat_order, props.values, color=cat_colors, width=0.62)
    ax.set_title(name, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")
axes[0].set_ylabel("Share of children")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/eda1_outcome_distributions.png", dpi=300, facecolor="white", bbox_inches="tight")
print("saved eda1_outcome_distributions.png")

print("\nCategory counts:")
for col, name in zip(outcomes, names):
    print(f"\n{name}:")
    print(df[col].value_counts().reindex(cat_order))
