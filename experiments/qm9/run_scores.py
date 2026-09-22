import os

os.environ["MPLBACKEND"] = "Agg"

import sys
import time
from itertools import product

import pandas as pd

from experiments.qm9.config import AGGREGATIONS,COMPARISON_ORDER, CURVATURES,EDGE_SCORED,GRAPH_TYPES, HFRAC_BINS, JOINT_COND, MAX_FILES, MULT_BINS, N_WORKERS, NH_BINS, NHEAVY_BINS,NODE_SCORED, OUT_DIR,  SHARD_GLOB, SKIP_SENSITIVITY
from experiments.qm9.constructions import  STYLING, register_duplicates_from_table,register_duplicates_from_values
from experiments.qm9.distributions import analyze_across_construction
from experiments.qm9.loading import load_qm9
from experiments.qm9.scores import run_conditioned_scores, score_slice
from plotting.conditioned import plot_share
from utils.tables import SEP_COL, save_csv, share_table

CONDITIONING_PASSES = [
    ("n", MULT_BINS),
    ("n_H", NH_BINS),
    ("n_heavy", NHEAVY_BINS),
    ("h_frac", HFRAC_BINS),
    (JOINT_COND, (NHEAVY_BINS, NH_BINS)),
]


def main():
    # Line buffering sends prints to the log file as they happen.
    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    data_dict = load_qm9(GRAPH_TYPES, SHARD_GLOB, N_WORKERS, MAX_FILES)
    print(f"loaded in {round(time.time() - start_time, 1)} s")

    tables = analyze_across_construction(data_dict, GRAPH_TYPES)
    across_graphs_nodes, across_graphs_edges, across_subsets_nodes, across_subsets_edges = tables

    if SKIP_SENSITIVITY:
        register_duplicates_from_values(data_dict, GRAPH_TYPES)
    else:
        register_duplicates_from_table(across_graphs_nodes)
        save_csv(across_graphs_nodes, OUT_DIR, "w1_across_graphs_nodes.csv")
        save_csv(across_graphs_edges, OUT_DIR, "w1_across_graphs_edges.csv")

    slice_inputs = [
        ("nodes", NODE_SCORED, across_subsets_nodes, across_graphs_nodes),
        ("edges", EDGE_SCORED, across_subsets_edges, across_graphs_edges),
    ]
    all_rows = []
    for table_kind, comparisons, df_subsets, df_graphs in slice_inputs:
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = score_slice(df_subsets, df_graphs, GRAPH_TYPES, comparison, curvature, aggregation)
            print(f"\n{table_kind.upper()}: {comparison} {curvature} {aggregation}")
            top_rows = df_slice.sort_values(SEP_COL, ascending=False)
            print(top_rows.head(10).to_string(index=False))
            file_name = f"scores_{table_kind}_{comparison}_{curvature}_{aggregation}.csv"
            save_csv(df_slice, OUT_DIR, file_name.replace("/", "_"))
            all_rows.append(df_slice)

    all_scores_df = pd.concat(all_rows, ignore_index=True)
    save_csv(all_scores_df, OUT_DIR, "scores_all_slices.csv")

    share_frames = []
    for cond_var, bins in CONDITIONING_PASSES:
        weighted = run_conditioned_scores(data_dict, GRAPH_TYPES, cond_var, bins, OUT_DIR)
        if weighted is None:
            continue
        share = share_table(all_scores_df, weighted, cond_var, "mols", ["n_bins_sep"])
        save_csv(share, OUT_DIR, f"share_{cond_var}.csv")
        print(f"\nSHARE (conditioned on {cond_var}; denominator on the same molecules,coverage reported):")
        print(share.to_string(index=False))
        plot_share(share, cond_var, OUT_DIR, STYLING, COMPARISON_ORDER, {}, AGGREGATIONS, "per-mol-mean", "molecules")
        share_frames.append(share)

    if share_frames:
        save_csv(pd.concat(share_frames, ignore_index=True), OUT_DIR, "share_all_condvars.csv")

    print(f"done in {round(time.time() - start_time, 1)} s")


if __name__ == "__main__":
    main()

