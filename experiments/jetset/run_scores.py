"""
Separation scores of the JetSet constructions: the unconditioned S_sep of
every slice, then the same conditioned on multiplicity, SV count, SV
fraction and the joint (n, n_sv) grid, with the share of the separation each
one removes. Every table and figure goes to OUT_DIR.

    python -m experiments.jetset.run_scores
"""
import os

# Thread limits and the plotting backend must be set before numpy and
# matplotlib are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import gc
import time
import pandas as pd

from experiments.jetset.config import (
    AGGREGATIONS, COMPARISON_LABELS, COMPARISON_ORDER,COND_EDGE_COMPARISONS, COND_GRAPH_TYPES,COND_NODE_COMPARISONS, CURVATURES,
    DEFAULT_GRAPH_TYPES, FLAVORPAIR_LABELS, FLAVORS,GRAPH_TYPES,
    JOINT_COND,MAX_TRAIN_JETS, MIN_JETS_PER_BIN, MULT_BINS, OUT_DIR, SUBSAMPLE_SEED, SV_BINS, SVFRAC_BINS, SWEEP_GRAPH_TYPES, W1_N_JOBS, WITHIN_COMPARISONS,
)
from experiments.jetset.constructions import STYLING, duplicate_free, register_duplicates
from experiments.jetset.distributions import within_rows
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import drop_graph_types, load_kept_jets
from experiments.jetset.plots import plot_sweep_set
from experiments.jetset.scores import run_conditioned_scores, run_unconditioned_scores, unconditioned_pair_scores
from plotting.conditioned import plot_share
from utils.memory import rss_gb
from utils.tables import save_csv, share_table
from utils.wassertein1_parallel import analyze_across_construction_from_hdf5_parallel, analyze_flavor_separation_edge_from_hdf5_parallel,analyze_flavor_separation_from_hdf5_parallel

# Conditioning passes in run order:
# - "none": Single bin with all jets for unconditioned scores.
# - "n": Second pass (fewest bins, highest coverage) to cross-check the matched denominator (see share_table)
CONDITIONING_PASSES = [
    ("none", [(0, 10 ** 9)]),
    ("n", MULT_BINS),
    ("n_sv", SV_BINS),
    ("sv_frac", SVFRAC_BINS),
    (JOINT_COND, (MULT_BINS, SV_BINS)),
]


def unconditioned_w1_tables(train_dict, save_dir):
    """
    Compute across-flavor W1 tables with added within-flavor PV-vs-SV rows and W1 utilities, saving outputs incrementally

    :param train_dict: Loaded split data
    :param save_dir: Output directory
    :return: tuple of across flavor node and edge tables
    """

    tables = analyze_across_construction_from_hdf5_parallel(train_dict, GRAPH_TYPES, flavors=FLAVORS, n_jobs=W1_N_JOBS)
    across_flavors_nodes, across_flavors_edges, data = tables[2:]

    within_node_rows = []
    for graph_type in GRAPH_TYPES:
        if graph_type not in data:
            continue
        for comparison, flavor in WITHIN_COMPARISONS.items():
            base = {
                "Graph": graph_type.upper(),
                "Flavor_A": f"Flavor {flavor}",
                "Flavor_B": f"Flavor {flavor}",
            }
            within_node_rows.extend(within_rows(base, comparison, data[graph_type][flavor], 1))
    if within_node_rows:
        across_flavors_nodes = pd.concat([across_flavors_nodes, pd.DataFrame(within_node_rows)], ignore_index=True)
    print(f"[mem] RSS after unconditioned W1 tables: {rss_gb():.1f} GB", flush=True)

    # Checkpoint right away so that a later failure does not lose this stage.
    save_csv(across_flavors_nodes, save_dir, "w1_uncond_across_flavors_nodes.csv")
    save_csv(across_flavors_edges, save_dir, "w1_uncond_across_flavors_edges.csv")
    print("[ckpt] unconditioned W1 tables written", flush=True)

    within_node_orc, within_node_frc = analyze_flavor_separation_from_hdf5_parallel(train_dict, FLAVORS, n_jobs=W1_N_JOBS, data=data)
    within_edge_orc, within_edge_frc = analyze_flavor_separation_edge_from_hdf5_parallel(train_dict, FLAVORS, n_jobs=W1_N_JOBS, data=data)
    within_outputs = {
        "within_node_orc": within_node_orc,
        "within_node_frc": within_node_frc,
        "within_edge_orc": within_edge_orc,
        "within_edge_frc": within_edge_frc,
    }
    for name, table in within_outputs.items():
        if table is None:
            continue
        save_csv(table, save_dir, f"w1_{name}.csv")
        print(f"\nWITHIN-FLAVOR ({name}): {len(table)} rows")
        print(table.head(20).to_string(index=False))

    del data, tables
    gc.collect()
    print(f"[mem] RSS after unconditioned tables: {rss_gb():.1f} GB", flush=True)
    return across_flavors_nodes, across_flavors_edges


def main():
    save_dir = OUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()

    kept_ids = get_kept_ids("train", MAX_TRAIN_JETS, SUBSAMPLE_SEED)
    train_dict = load_kept_jets(kept_ids)
    print(f"loaded in {time.time() - start:.1f} s")
    print(f"[mem] RSS after load: {rss_gb():.1f} GB")

    across_flavors_nodes, across_flavors_edges = unconditioned_w1_tables(train_dict, save_dir)
    register_duplicates(train_dict, GRAPH_TYPES)

    all_scores_df = run_unconditioned_scores(across_flavors_nodes, across_flavors_edges, save_dir)
    print(f"[plot] {len(DEFAULT_GRAPH_TYPES)} defaults, {len(SWEEP_GRAPH_TYPES)} sweep constructions available for the figures", flush=True)
    plot_sweep_set(all_scores_df, "scatter_uncond", "", save_dir)
    unconditioned_pair_scores(across_flavors_nodes, across_flavors_edges, save_dir)

    # Duplicates are skipped at scoring time anyway, so there is no point in
    # carrying them through the conditioned passes.
    cond_types = duplicate_free(COND_GRAPH_TYPES)
    print(f"[cond] {len(cond_types)} constructions after dropping duplicates", flush=True)
    train_dict = drop_graph_types(train_dict, cond_types)

    share_frames = []
    pair_frames = []
    for cond_var, bins in CONDITIONING_PASSES:
        weighted, pair_summary = run_conditioned_scores(
            data_dict=train_dict, graph_types=cond_types, flavors=FLAVORS,
            flavorpair_labels=FLAVORPAIR_LABELS, node_comparisons=COND_NODE_COMPARISONS,
            edge_comparisons=COND_EDGE_COMPARISONS,
            within_comparisons=list(WITHIN_COMPARISONS), curvatures=CURVATURES,
            aggregations=AGGREGATIONS, save_dir=save_dir, cond_var=cond_var, bins=bins, min_jets=MIN_JETS_PER_BIN)

        if pair_summary is not None and not pair_summary.empty:
            pair_frames.append(pair_summary)
        if weighted is None:
            continue

        share = share_table(all_scores_df, weighted, cond_var, "jets", ["n_pairs"])
        save_csv(share, save_dir, f"share_{cond_var}.csv")
        print(f"\nSHARE (conditioned on {cond_var}; denominator on the same jets, coverage reported):")
        print(share.to_string(index=False))
        share_frames.append(share)
        plot_share(share, cond_var, save_dir, STYLING, COMPARISON_ORDER, COMPARISON_LABELS, AGGREGATIONS, "per-jet-mean", "jets")

        plot_sweep_set(weighted, f"scatter_weighted_{cond_var}", f", conditioned on {cond_var}",save_dir)

    if share_frames:
        save_csv(pd.concat(share_frames, ignore_index=True), save_dir, "share_all_condvars.csv")
    if pair_frames:
        save_csv(pd.concat(pair_frames, ignore_index=True), save_dir, "share_per_pair_all_condvars.csv")

    print(f"done in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()

