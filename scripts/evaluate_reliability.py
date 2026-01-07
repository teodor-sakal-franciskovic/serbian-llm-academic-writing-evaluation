import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr


# ---------------------------------------------------------
# Helper: Krippendorff’s alpha (ordinal level)
# ---------------------------------------------------------
def krippendorff_alpha_ordinal(scores_df_long):
    """Compute Krippendorff’s alpha for ordinal data."""
    mat = scores_df_long.pivot_table(
        index="rater_id", columns="essay_id", values="score"
    )
    cats = sorted(pd.unique(scores_df_long["score"].dropna()))
    K = len(cats)
    if K <= 1:
        return np.nan

    # Coincidence matrix
    C = pd.DataFrame(0.0, index=cats, columns=cats)
    for essay, col in mat.items():
        vals = col.dropna().values
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                C.loc[vals[i], vals[j]] += 1
                C.loc[vals[j], vals[i]] += 1

    if C.values.sum() == 0:
        return np.nan

    def delta2(i, j):
        return ((abs(i - j)) / (K - 1)) ** 2

    Do = (sum(delta2(i, j) * C.loc[i, j] for i in cats for j in cats)) / C.values.sum()
    m = C.sum(axis=1).values
    De = (
        sum(delta2(cats[a], cats[b]) * m[a] * m[b] for a in range(K) for b in range(K))
    ) / (m.sum() ** 2)
    if De == 0 or np.isnan(De):
        return np.nan
    return 1 - (Do / De)


# ---------------------------------------------------------
# Load and clean CSVs
# ---------------------------------------------------------
r1 = pd.read_csv("main_grader.csv")
r2 = pd.read_csv("side_grader.csv")

# Drop unnamed or empty columns
r1 = r1.loc[:, ~r1.columns.str.contains("^Unnamed")]
r2 = r2.loc[:, ~r2.columns.str.contains("^Unnamed")]
r1 = r1.dropna(axis=1, how="all")
r2 = r2.dropna(axis=1, how="all")

# Merge on essay ID column ("Rad")
df = r1.merge(r2, on="Rad", suffixes=("_r1", "_r2"), how="inner")

# Determine which rubric dimensions exist in both graders
dims = sorted(list(set(r1.columns) & set(r2.columns) - {"Rad"}))
print(f"✅ Essays merged: {len(df)}")
print(
    f"✅ Common dimensions: {len(dims)} ({', '.join(dims[:5])}{'...' if len(dims) > 5 else ''})"
)

# Warn about columns that exist only in one file
only_r1 = sorted(list(set(r1.columns) - set(r2.columns) - {"Rad"}))
only_r2 = sorted(list(set(r2.columns) - set(r1.columns) - {"Rad"}))
if only_r1:
    print(f"⚠️  Columns only in main_grader.csv (ignored): {only_r1}")
if only_r2:
    print(f"⚠️  Columns only in side_grader.csv (ignored): {only_r2}")

# ---------------------------------------------------------
# Compute metrics per dimension
# ---------------------------------------------------------
rows = []
for d in dims:
    col_r1 = f"{d}_r1"
    col_r2 = f"{d}_r2"
    if col_r1 not in df.columns or col_r2 not in df.columns:
        continue

    s1, s2 = df[col_r1], df[col_r2]
    mask = ~(s1.isna() | s2.isna())
    s1c, s2c = s1[mask], s2[mask]

    if len(s1c) == 0:
        continue

    # Weighted Cohen’s κ
    kappa = cohen_kappa_score(s1c, s2c, weights="quadratic")

    # Exact & Adjacent agreement
    diffs = (s1c - s2c).abs()
    exact = (diffs == 0).mean() * 100
    adjacent = (diffs <= 1).mean() * 100

    # Krippendorff’s α (ordinal)
    long_df = pd.DataFrame(
        {
            "essay_id": df.loc[mask, "Rad"].tolist() * 2,
            "rater_id": ["R1"] * len(s1c) + ["R2"] * len(s2c),
            "score": pd.concat([s1c, s2c], ignore_index=True),
        }
    )
    alpha = krippendorff_alpha_ordinal(long_df)

    rows.append(
        {
            "dimension": d,
            "n_pairs": len(s1c),
            "kappa_w_quad": kappa,
            "alpha_ordinal": alpha,
            "exact_%": exact,
            "adjacent_%": adjacent,
        }
    )

metrics_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# Overall totals (sum across dimensions)
# ---------------------------------------------------------
df["total_r1"] = df[[f"{d}_r1" for d in dims if f"{d}_r1" in df]].sum(axis=1)
df["total_r2"] = df[[f"{d}_r2" for d in dims if f"{d}_r2" in df]].sum(axis=1)
rho, p = spearmanr(df["total_r1"], df["total_r2"], nan_policy="omit")

summary = {
    "N_essays": len(df),
    "N_dimensions": len(dims),
    "Spearman_rho_total": round(rho, 3),
    "Spearman_p_value": round(p, 6),
}

# ---------------------------------------------------------
# Save and display results
# ---------------------------------------------------------
metrics_df.to_csv("reliability_metrics_per_dimension_v2.csv", index=False)
pd.DataFrame([summary]).to_csv("reliability_summary_v2.csv", index=False)

print("\n=== Summary ===")
for k, v in summary.items():
    print(f"{k:25} {v}")

print("\n=== First few dimensions ===")
print(metrics_df.head(10).to_string(index=False))
print(
    "\n✅ Results saved: reliability_metrics_per_dimension.csv and reliability_summary.csv"
)
