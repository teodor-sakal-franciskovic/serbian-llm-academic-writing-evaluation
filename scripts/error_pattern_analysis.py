import pandas as pd
import numpy as np
import glob
import os

# ============================================================
# CONFIG
# ============================================================
HUMAN_FILE = "main_grader_final.csv"
GPT_FOLDER = "llm_results"
ID_COL = "paper_name"

OUTPUT_SUMMARY = "llm_error_summary.csv"
OUTPUT_DETAILED = "llm_error_detailed.csv"

# ============================================================
# Load human data
# ============================================================
human = pd.read_csv(HUMAN_FILE)
human = human.loc[:, ~human.columns.str.contains("^Unnamed")]

dims = [c for c in human.columns if c != ID_COL]

print(f"Loaded human labels: {len(human)} papers, {len(dims)} rubric dimensions")


# ============================================================
# Helper: analyze one LLM-evaluated file
# ============================================================
def analyze_llm_file(gpt_path):
    gpt = pd.read_csv(gpt_path)
    gpt = gpt.loc[:, ~gpt.columns.str.contains("^Unnamed")]

    df = human.merge(gpt, on=ID_COL, suffixes=("_human", "_llm"))

    deltas = []

    for d in dims:
        h = df[f"{d}_human"]
        g = df[f"{d}_llm"]

        mask = ~(h.isna() | g.isna())
        deltas.extend((g[mask] - h[mask]).tolist())

    deltas = np.array(deltas)

    total = len(deltas)

    summary = {
        "file": os.path.basename(gpt_path),
        "n_comparisons": total,
        "exact_%": np.mean(deltas == 0) * 100,
        "minor_%": np.mean(np.abs(deltas) == 1) * 100,
        "severe_%": np.mean(np.abs(deltas) == 2) * 100,
        "over_scoring_%": np.mean(deltas > 0) * 100,
        "under_scoring_%": np.mean(deltas < 0) * 100,
        "mean_delta": np.mean(deltas),
    }

    detailed = pd.DataFrame(
        {
            "file": os.path.basename(gpt_path),
            "delta": deltas,
            "error_type": np.where(
                deltas == 0, "exact", np.where(np.abs(deltas) == 1, "minor", "severe")
            ),
            "direction": np.where(
                deltas > 0, "over", np.where(deltas < 0, "under", "none")
            ),
        }
    )

    return summary, detailed


# ============================================================
# Run analysis for all LLM-evaluated files
# ============================================================
all_summaries = []
all_detailed = []

for gpt_path in glob.glob(f"{GPT_FOLDER}/*.csv"):
    print(f"Analyzing {gpt_path} ...")
    summary, detailed = analyze_llm_file(gpt_path)
    all_summaries.append(summary)
    all_detailed.append(detailed)

summary_df = pd.DataFrame(all_summaries)
detailed_df = pd.concat(all_detailed, ignore_index=True)

summary_df.to_csv(OUTPUT_SUMMARY, index=False)
detailed_df.to_csv(OUTPUT_DETAILED, index=False)

print("\n============================================")
print("RQ4 error analysis complete")
print(f"Summary saved to:   {OUTPUT_SUMMARY}")
print(f"Detailed saved to:  {OUTPUT_DETAILED}")
print("============================================")
