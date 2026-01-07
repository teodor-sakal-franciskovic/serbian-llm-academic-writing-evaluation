import pandas as pd
import numpy as np
import glob
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import spearmanr

# ============================================================
# Load human (ground truth) labels
# ============================================================
human = pd.read_csv("main_grader_final.csv")
human = human.loc[:, ~human.columns.str.contains("^Unnamed")]

# Identify rubric dimensions
dims = [c for c in human.columns if c != "paper_name"]

print("\n================= HUMAN COLUMN INFO =================")
print(f"Loaded human labels: {human.shape[0]} essays")
print(f"Rubric dimensions ({len(dims)}):")
for d in dims:
    print("  -", d)
print("====================================================\n")


# ============================================================
# Helper: evaluate ONE GPT output file
# ============================================================
def evaluate_one_gpt_file(gpt_path: str):
    print("\n-------------------------------------------------------")
    print(f"🔍 ANALYZING GPT FILE: {gpt_path}")
    print("-------------------------------------------------------")

    gpt = pd.read_csv(gpt_path)
    gpt = gpt.loc[:, ~gpt.columns.str.contains("^Unnamed")]

    # ---------------- Proceed only if ID column exists ----------------
    df = human.merge(gpt, on="paper_name", suffixes=("_human", "_gpt"))

    kappa_list = []
    exact_list = []
    adjacent_list = []
    mae_dim_list = []

    # ========================================================
    # Per-dimension metrics
    # ========================================================
    for d in dims:
        col_h = f"{d}_human"
        col_g = f"{d}_gpt"

        if col_h not in df.columns or col_g not in df.columns:
            continue

        h = df[col_h]
        g = df[col_g]

        mask = ~(h.isna() | g.isna())
        h, g = h[mask], g[mask]

        if len(h) == 0:
            continue

        # ===================== DEBUG KAPPA ===========================
        try:
            kappa = cohen_kappa_score(h, g, weights="quadratic")

            if pd.isna(kappa):
                print(f"   ⚠️  KAPPA IS NaN for dimension: {d}")
                print(f"      h values: {list(h)}")
                print(f"      g values: {list(g)}")
            else:
                print(f"   ✅ KAPPA({d}) = {kappa:.3f}")

                kappa_list.append(kappa)

        except Exception as e:
            print(f"   ❌ ERROR computing kappa for {d}: {e}")
            print(f"      h values: {list(h)}")
            print(f"      g values: {list(g)}")
        # =============================================================

        exact_list.append((h == g).mean() * 100)
        adjacent_list.append((np.abs(h - g) <= 1).mean() * 100)
        mae_dim_list.append(mean_absolute_error(h, g))

    # ========================================================
    # Overall totals
    # ========================================================
    df["total_human"] = df[[f"{d}_human" for d in dims]].sum(axis=1)
    df["total_gpt"] = df[[f"{d}_gpt" for d in dims]].sum(axis=1)

    rho, _ = spearmanr(df["total_human"], df["total_gpt"])
    mae_total = mean_absolute_error(df["total_human"], df["total_gpt"])

    print("\n👉 Dimensions evaluated:", len(kappa_list))
    print(f"Kappa list: {kappa_list}")
    print(f"Kappa mean to be inserted: {np.mean(kappa_list)}")
    print("-------------------------------------------------------\n")

    return {
        "file": gpt_path,
        "kappa_mean": np.mean(kappa_list),
        "kappa_median": np.median(kappa_list),
        "exact_mean_%": np.mean(exact_list),
        "adjacent_mean_%": np.mean(adjacent_list),
        "mae_dim_mean": np.mean(mae_dim_list),
        "rho_total": rho,
        "mae_total": mae_total,
        "n_dimensions_evaluated": len(kappa_list),
    }


# ============================================================
# Process ALL GPT CSVs in results/ folder
# ============================================================
results = []
print("\n============== PROCESSING GPT FILES ==============\n")

for gpt_path in glob.glob("results/*.csv"):
    metrics = evaluate_one_gpt_file(gpt_path)
    results.append(metrics)

summary_df = pd.DataFrame(results)

summary_df.to_csv("gpt_vs_human_summary_v2.csv", index=False)

print("\n===========================================================")
print("🎉 Done! Summary saved to: gpt_vs_human_summary.csv")
print("===========================================================\n")

print(summary_df)
