import os
import time

from experiments.jetset.config import (COND_GRAPH_TYPES, EDGE_SPECS, FLAVORS, GRAPH_TYPES, JOINT_COND, MAX_TRAIN_JETS, MIN_JETS_PER_BIN,
    MIN_VALUES_PER_SIDE, MULT_BINS, NODE_SPECS, OUT_DIR, PAIR_FLAVORS, SUBSAMPLE_SEED, SV_BINS, SVFRAC_BINS, WITHIN_COMPARISONS, WITHIN_PAIR_LABEL,)
from experiments.jetset.constructions import duplicate_free, register_duplicates
from experiments.jetset.distributions import collect_binned_distributions, cond_bin_labels, flavor_by_jet, per_jet_bin
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import drop_graph_types, load_kept_jets
from utils import bootstrap as bs
from utils.tables import SCORE_KEYS, SEP_COL

N_BOOT = 500
SEED = 0
LEVEL = 95.0
N_THREADS = 8  # Numba kernel threads (ignored if numba is absent)

# Unconditioned score plus conditioning variables from run_scores.py
PASSES = ["uncond", "n", "n_sv", "sv_frac", "n_x_n_sv"]

# None = all scored constructions , list = subset for trial runs (skips score CSVs)
GRAPHS = None

ATTACH = True  # Append interval columns to score CSVs in OUT_DIR

PASS_DEFINITIONS = {
    "uncond": ("n", [(0, 10 ** 9)], 0), # # Single bin holding all jets (matches "none" pass)
    "n": ("n", MULT_BINS, MIN_JETS_PER_BIN),
    "n_sv": ("n_sv", SV_BINS, MIN_JETS_PER_BIN),
    "sv_frac": ("sv_frac", SVFRAC_BINS, MIN_JETS_PER_BIN),
    JOINT_COND: (JOINT_COND, (MULT_BINS, SV_BINS), MIN_JETS_PER_BIN),
}


def jetset_terms(specs, pair_flavors, within_comparisons, within_pair_label):
    """
    Every (slice, pair) term of the jets analysis.

    :param specs: (comparison, distribution key, curvature, aggregation) tuples
    :param pair_flavors: {pair label: (flavor A, flavor B)}
    :param within_comparisons: {comparison name: flavor} for PV-vs-SV subsets
    :param within_pair_label: pair label of the within-flavor subsets
    :return: list of term dicts with Comparison, Curvature, Aggregation,
        FlavorPair, side_a and side_b, each side a (population, key) tuple
    """
    key_of = {}
    terms = []
    for comparison, key, curvature, aggregation in specs:
        key_of[(comparison, curvature, aggregation)] = key
        for pair, flavors in pair_flavors.items():
            terms.append({
                "Comparison": comparison, "Curvature": curvature,
                "Aggregation": aggregation, "FlavorPair": pair,
                "side_a": (flavors[0], key), "side_b": (flavors[1], key),
            })
    for comparison, flavor in within_comparisons.items():
        for (spec_comparison, curvature, aggregation), key in key_of.items():
            if spec_comparison != "PV_nodes":
                continue
            terms.append({
                "Comparison": comparison, "Curvature": curvature,
                "Aggregation": aggregation, "FlavorPair": within_pair_label,
                "side_a": (flavor, key),
                "side_b": (flavor, key_of[("SV_nodes", curvature, aggregation)]),
            })
    return terms


def attach_jobs(name):
    """
    Determine which interval columns apply to the score CSVs for a given pass.
    scores_all_slices.csv lacks all_nodes/all_edges rows, so the unconditioned pass
    also populates the "none" tables from run_scores.py (means from none_weighted, pairs from none_per_pair_matched). 
    With a single bin, S_sep_cond equals the unconditioned score. 
    Columns re named "Sep ..." to match plotting script expectations.

    :param name: pass name
    :return: list of jobs for bs.check_and_attach
    """
    pair_keys = SCORE_KEYS + ["FlavorPair"]
    if name == "uncond":
        return [
            ("scores_all_slices.csv", "per_construction", SCORE_KEYS,
             SEP_COL, "S_sep_kept point", bs.interval_columns("S_sep_kept", "Sep")),
            ("scores_per_pair_uncond.csv", "per_pair", pair_keys,
             SEP_COL, "S_sep_kept point", bs.interval_columns("S_sep_kept", "Sep")),
            ("scores_conditioned_none_all_slices.csv", "per_bin", SCORE_KEYS + ["Bin"],
             SEP_COL, "Sep point", bs.interval_columns("Sep", "Sep")),
            ("scores_conditioned_none_per_pair_bins.csv", "per_pair_bins",
             pair_keys + ["Bin"], SEP_COL, "Sep point", bs.interval_columns("Sep", "Sep")),
            ("scores_conditioned_none_weighted.csv", "per_construction", SCORE_KEYS,
             SEP_COL, "S_sep_cond point", bs.interval_columns("S_sep_cond", "Sep")),
            ("scores_conditioned_none_per_pair_matched.csv", "per_pair", pair_keys,
             "S_sep_cond", "S_sep_cond point", bs.interval_columns("S_sep_cond", "Sep")),
        ]
    share_columns = {}
    for quantity in ("share", "S_sep_cond", "S_sep_kept"):
        share_columns.update(bs.interval_columns(quantity, quantity))
    return [
        (f"scores_conditioned_{name}_all_slices.csv", "per_bin", SCORE_KEYS + ["Bin"],
         SEP_COL, "Sep point", bs.interval_columns("Sep", "Sep")),
        (f"scores_conditioned_{name}_per_pair_bins.csv", "per_pair_bins",
         pair_keys + ["Bin"], SEP_COL, "Sep point", bs.interval_columns("Sep", "Sep")),
        (f"scores_conditioned_{name}_weighted.csv", "per_construction", SCORE_KEYS,
         "share", "share point", share_columns),
        (f"share_{name}.csv", "per_construction", SCORE_KEYS,
         "share", "share point", share_columns),
        (f"scores_conditioned_{name}_per_pair_matched.csv", "per_pair", pair_keys,
         "share", "share point", share_columns),
    ]


def main():
    start = time.time()
    boot_dir = os.path.join(OUT_DIR, "bootstrap")
    os.makedirs(boot_dir, exist_ok=True)
    print(f"[cfg] score CSVs are read from and updated in {OUT_DIR}", flush=True)
    bs.configure_numba(N_THREADS)

    kept_ids = get_kept_ids("train", MAX_TRAIN_JETS, SUBSAMPLE_SEED)
    train_dict = load_kept_jets(kept_ids)
    # The duplicate check reads the curvatures, so the jets come first.
    register_duplicates(train_dict, GRAPH_TYPES)
    graphs = duplicate_free(COND_GRAPH_TYPES)
    attach = ATTACH
    if GRAPHS is not None:
        graphs = bs.select_graphs(graphs, GRAPHS)
        attach = False
        print("[cfg] GRAPHS is set: trial run, the score CSVs are left alone", flush=True)
    if not graphs:
        raise SystemExit("no construction to bootstrap; check GRAPHS against COND_GRAPH_TYPES")
    print(f"[cfg] B={N_BOOT}, seed={SEED}, {len(graphs)} constructions, passes={PASSES}", flush=True)

    train_dict = drop_graph_types(train_dict, graphs)
    graphs = bs.select_graphs(graphs, train_dict)
    print(f"[load] {len(graphs)} constructions in {time.time() - start:.0f} s", flush=True)

    # Every construction is built on the same jets, so the first one gives the
    # jet ids, flavors and bins for all of them.
    first_nodes = train_dict[graphs[0]]["nodes"]
    jet_flavor = flavor_by_jet(first_nodes).sort_index()
    jet_ids = jet_flavor.index.to_numpy()
    counts = bs.draw_counts(jet_flavor.to_numpy(), N_BOOT, SEED)
    print(f"[boot] {counts.shape[0]} resamples of {counts.shape[1]} jets, drawn within flavor; {counts.nbytes / 1e6:.0f} MB", flush=True)

    terms = jetset_terms(NODE_SPECS + EDGE_SPECS, PAIR_FLAVORS, WITHIN_COMPARISONS, WITHIN_PAIR_LABEL)
    n_val = max(1, MIN_VALUES_PER_SIDE)

    for name in PASSES:
        if name not in PASS_DEFINITIONS:
            print(f"[warn] unknown pass '{name}', skipped", flush=True)
            continue
        cond_var, bins, min_jets = PASS_DEFINITIONS[name]
        labels = cond_bin_labels(cond_var, bins)
        print(f"\n bootstrap: {name} ({len(labels)} bins)", flush=True)

        jet_bin = per_jet_bin(first_nodes, cond_var, bins).reindex(jet_ids)
        weights = bs.cell_weight_replicates(jet_flavor.to_numpy(), jet_bin.to_numpy(), counts)

        def cells_of(graph_type):
            binned, _ = collect_binned_distributions(train_dict, [graph_type], FLAVORS, cond_var, bins, min_jets=min_jets, keep_ids=True)
            return binned[graph_type]

        tables = bs.bootstrap_pass(name, graphs, cells_of, terms, labels, weights, jet_ids, counts, n_val, LEVEL, boot_dir)
        bs.check_and_attach(OUT_DIR, attach_jobs(name), tables, attach)

    print(f"\ndone in {(time.time() - start) / 3600:.2f} h; tables in {boot_dir}", flush=True)


if __name__ == "__main__":
    main()

