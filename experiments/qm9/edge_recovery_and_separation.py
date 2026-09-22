import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

QM9_RESULTS_DIR = "/home/dshahi/projects/JetGraphs-RicciCurvature/results_qm9"
EDGE_RECOVERY_PATH = os.path.join(QM9_RESULTS_DIR, "edge_recovery_summary.csv")
OUT_DIR = "/home/dshahi/projects/JetGraphs-RicciCurvature/resultsqm9-distributional_300000"
SCORES_PATH = os.path.join(OUT_DIR, "w1-subsets-all-pairs-defaults", "scores_all_slices.csv")
OUT_PATH = os.path.join(QM9_RESULTS_DIR, "spearman_f1_vs_separation.csv")
RANKS_PATH = os.path.join(OUT_DIR, "spearman_f1_vs_separation_ranks.csv")

F1_COL = "f1"  # mean over molecules
COMPARISON = "H_vs_heavy"
CURVATURES = ["orc", "frc"]
AGGREGATIONS = ["union", "per_mol_mean"]
SEP_COL = "Separability (higher better)"
FC_GRAPH = "fully_connected"
GRAPH_TYPES = None  # None = every construction in both tables


def load_f1(path):
    summary = pd.read_csv(path)
    summary["Graph"] = summary["graph_type"].str.upper()
    return summary.set_index("Graph")[F1_COL].astype(float)


def load_separation(path):
    scores = pd.read_csv(path)
    scores = scores[scores["Comparison"] == COMPARISON]
    scores = scores.dropna(subset=[SEP_COL])
    return scores[["Graph", "Curvature", "Aggregation", SEP_COL]]


def correlate(table, curvature, aggregation, label):
    row = {
        "Curvature": curvature,
        "Aggregation": aggregation,
        "constructions": label,
        "n": len(table),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "kendall_tau": np.nan,
        "kendall_p": np.nan,
    }
    if len(table) < 3:
        return row

    rho, p_rho = spearmanr(table["F1"], table[SEP_COL])
    tau, p_tau = kendalltau(table["F1"], table[SEP_COL])
    row["spearman_rho"] = float(rho)
    row["spearman_p"] = float(p_rho)
    row["kendall_tau"] = float(tau)
    row["kendall_p"] = float(p_tau)
    return row


def ranked(table, curvature, aggregation, label):
    out = table.copy()
    out["F1_rank"] = out["F1"].rank(ascending=False).astype(int)
    out["Sep_rank"] = out[SEP_COL].rank(ascending=False).astype(int)
    out["Curvature"] = curvature
    out["Aggregation"] = aggregation
    out["constructions"] = label
    return out.sort_values("F1_rank")


def main():
    f1 = load_f1(EDGE_RECOVERY_PATH)
    separation = load_separation(SCORES_PATH)

    if GRAPH_TYPES is not None:
        keep = []
        for name in GRAPH_TYPES:
            keep.append(name.upper())
        f1 = f1[f1.index.isin(keep)]
        separation = separation[separation["Graph"].isin(keep)]

    results = []
    ranks = []
    for curvature in CURVATURES:
        for aggregation in AGGREGATIONS:
            mask = (separation["Curvature"] == curvature) & (separation["Aggregation"] == aggregation)
            table = separation[mask].merge(f1.rename("F1"), left_on="Graph", right_index=True)
            table = table.dropna(subset=["F1", SEP_COL])
            if table.empty:
                print(f"[skip] {curvature} {aggregation}: no construction in both tables")
                continue

            subsets = [("all", table)]
            if curvature == "frc":
                subsets.append(("no_FC", table[table["Graph"] != FC_GRAPH.upper()]))

            for label, subset in subsets:
                results.append(correlate(subset, curvature, aggregation, label))
                ranks.append(ranked(subset, curvature, aggregation, label))

    if not results:
        raise SystemExit("nothing to correlate; check the two input paths")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_PATH, index=False)
    pd.concat(ranks, ignore_index=True).to_csv(RANKS_PATH, index=False)

    print(f"F1 ({F1_COL}) vs S_sep({COMPARISON})\n")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nfor the text:")
    for row in results:
        if not np.isfinite(row["spearman_rho"]):
            continue
        name = f"{row['Curvature'].upper():<4}{row['Aggregation']:<14}{row['constructions']:<7}"
        stats = f"rho = {row['spearman_rho']:.2f}, p = {row['spearman_p']:.1e}, n = {row['n']}"
        print(f"  {name}{stats}")
    print(f"\nwritten {OUT_PATH}\n        {RANKS_PATH}")


if __name__ == "__main__":
    main()
