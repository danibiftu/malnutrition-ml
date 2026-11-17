"""
EDA Step 3: pairwise relationships between the three anthropometric
z-scores (HAZ, WAZ, WHZ), density-colored with viridis.
"""
import os
import pyreadstat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

# Change these two paths to match your own setup.
OUTDIR = r"D:\CAIMIS Research\Articles\Malnutrition"
DATA_PATH = r"D:\CAIMIS Research\Articles\Malnutrition\ml_malnutrition_clean.dta"
os.makedirs(OUTDIR, exist_ok=True)

INK_PRIMARY, INK_MUTED, GRIDLINE = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 10.5, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_PRIMARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK_PRIMARY,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})

df, meta = pyreadstat.read_dta(DATA_PATH)

z_cols = ["haz", "waz", "whz"]
z_names = ["HAZ (stunting)", "WAZ (underweight)", "WHZ (wasting)"]
pairs = [(0, 1), (0, 2), (1, 2)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (a, b) in zip(axes, pairs):
    both = df[[z_cols[a], z_cols[b]]].dropna()
    r = np.corrcoef(both[z_cols[a]], both[z_cols[b]])[0, 1]
    xy = np.vstack([both[z_cols[a]], both[z_cols[b]]])
    dens = gaussian_kde(xy)(xy)
    order = np.argsort(dens)
    xs = both[z_cols[a]].values[order]
    ys = both[z_cols[b]].values[order]
    cs = dens[order]
    ax.scatter(xs, ys, c=cs, cmap="viridis", s=7, alpha=0.6, linewidths=0)
    ax.axvline(-2, color="white", lw=1.0, ls=":")
    ax.axhline(-2, color="white", lw=1.0, ls=":")
    ax.set_xlabel(z_names[a])
    ax.set_ylabel(z_names[b])
    ax.set_title(f"$r$ = {r:.2f}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/eda3_zscore_pairs.png", dpi=300, facecolor="white", bbox_inches="tight")
print("saved eda3_zscore_pairs.png")

print("\nPairwise Pearson correlations:")
for a, b in pairs:
    both = df[[z_cols[a], z_cols[b]]].dropna()
    r = np.corrcoef(both[z_cols[a]], both[z_cols[b]])[0, 1]
    print(f"  {z_names[a]} vs {z_names[b]}: r = {r:.3f} (n = {len(both):,})")
