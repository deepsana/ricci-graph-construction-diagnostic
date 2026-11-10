import os

# Thread limits and the plotting backend must be set before numpy and
# matplotlib are imported.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import time
import pandas as pd

from experiments.jetset.config import (
    AGGREGATIONS, COMPARISON_LABELS, COMPARISON_ORDER, COND_GRAPH_TYPES, CURVATURES, DEFAULT_GRAPH_TYPES,
    GRAPH_TYPES, JOINT_COND, MAX_TRAIN_JETS, MULT_BINS, OUT_DIR, SUBSAMPLE_SEED, SV_BINS, SVFRAC_BINS,
    SWEEP_GRAPH_TYPES,
)
from experiments.jetset.constructions import STYLING, duplicate_free, register_duplicates
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import drop_graph_types, load_kept_jets
from experiments.jetset.plots import plot_sweep_set
from experiments.jetset.scores import run_conditioned_scores, unconditioned_tables_from_none_pass
from plotting.conditioned import plot_share
from utils.graph_descriptors import DESCRIPTORS, add_graph_descriptors, degree_vs_curvature_table
from utils.memory import rss_gb
from utils.tables import save_csv, share_table

# Conditioning passes in run order:
# - "none": one bin with all jets, which gives the unconditioned scores.
# - "n": fewest bins and highest coverage, a cross-check on the matched denominator (see share_table).
CONDITIONING_PASSES = [
    ("none", [(0, 10 ** 9)]),
    ("n", MULT_BINS),
    ("n_sv", SV_BINS),
    ("sv_frac", SVFRAC_BINS),
    (JOINT_COND, (MULT_BINS, SV_BINS)),
]


def main():
    save_dir = OUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    print(f"[cfg] descriptors {CURVATURES}; output {save_dir}", flush=True)

    kept_ids = get_kept_ids("train", MAX_TRAIN_JETS, SUBSAMPLE_SEED)
    train_dict = load_kept_jets(kept_ids)
    print(f"loaded in {time.time() - start:.1f} s; RSS {rss_gb():.1f} GB", flush=True)

    # Read off the cached edge tables, so the cache itself is unchanged.
    if any(descriptor in CURVATURES for descriptor in DESCRIPTORS):
        train_dict = add_graph_descriptors(train_dict, GRAPH_TYPES, "jet_id")

    # The duplicate check compares ORC and FRC on every node.
    register_duplicates(train_dict, GRAPH_TYPES)
    cond_types = duplicate_free(COND_GRAPH_TYPES)
    print(f"[cond] {len(cond_types)} constructions after dropping duplicates", flush=True)
    train_dict = drop_graph_types(train_dict, cond_types)
    print(f"[plot] {len(DEFAULT_GRAPH_TYPES)} defaults, {len(SWEEP_GRAPH_TYPES)} sweep constructions available for the figures", flush=True)

    all_scores_df = None
    share_frames = []
    pair_frames = []
    for cond_var, bins in CONDITIONING_PASSES:
        weighted, pair_summary = run_conditioned_scores(train_dict, cond_types, cond_var, bins, save_dir)

        if pair_summary is not None and not pair_summary.empty:
            pair_frames.append(pair_summary)
        if weighted is None:
            if cond_var == "none":
                raise SystemExit("FATAL: the unconditioned (none) pass produced no scores")
            continue

        if cond_var == "none":
            all_scores_df, _ = unconditioned_tables_from_none_pass(weighted, pair_summary, save_dir)
            degree_vs_curvature_table(all_scores_df, CURVATURES, save_dir)
            plot_sweep_set(all_scores_df, "scatter_uncond", "", save_dir)
            print(f"[ckpt] unconditioned tables written; RSS {rss_gb():.1f} GB", flush=True)
            continue

        share = share_table(all_scores_df, weighted, cond_var, "jets", ["n_pairs"])
        save_csv(share, save_dir, f"share_{cond_var}.csv")
        print(f"\nSHARE (conditioned on {cond_var}; denominator on the same jets, coverage reported):")
        print(share.to_string(index=False))
        share_frames.append(share)
        plot_share(share, cond_var, save_dir, STYLING, COMPARISON_ORDER, COMPARISON_LABELS, AGGREGATIONS, "per-jet-mean", "jets")
        plot_sweep_set(weighted, f"scatter_weighted_{cond_var}", f", conditioned on {cond_var}", save_dir)

    if share_frames:
        save_csv(pd.concat(share_frames, ignore_index=True), save_dir, "share_all_condvars.csv")
    if pair_frames:
        save_csv(pd.concat(pair_frames, ignore_index=True), save_dir, "share_per_pair_all_condvars.csv")

    print(f"done in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()

