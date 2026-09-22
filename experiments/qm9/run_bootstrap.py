import os
import time
import numpy as np
import pandas as pd

from experiments.qm9.config import EDGE_SPECS, GRAPH_TYPES, HFRAC_BINS,JOINT_COND, MAX_FILES,MIN_MOLS_PER_BIN, MIN_VALUES_PER_DIST, MULT_BINS, N_WORKERS, NH_BINS, NHEAVY_BINS, NODE_SPECS, OUT_DIR, SEP_POOL, SHARD_GLOB, SUBSET_PAIRS
from experiments.qm9.distributions import collect_binned_distributions, cond_bin_labels, per_mol_bin
from experiments.qm9.loading import load_qm9
from utils import bootstrap as bs
from utils.tables import SCORE_KEYS, SEP_COL

N_BOOT = 500
SEED = 0
LEVEL = 95.0
# Threads of the numba kernel. Ignored when numba is not installed.
N_THREADS = 8

# "uncond" is the unconditioned score; the rest are the conditioning
# variables of run_scores.py.
PASSES = ["uncond", "n", "n_H", "n_heavy", "h_frac", "n_heavy_x_n_H"]

# None = every construction the analysis scored. A list of names, such as
# ["knn_3", "radius_bond"], runs only those and never touches the score CSVs,
# so it is safe for a trial run.
GRAPHS = None

# Add the interval columns to the score CSVs in OUT_DIR.
ATTACH = True

# The single population every molecule belongs to.
POPULATION = "all"

# "uncond" is the single bin "all" with no thin-bin cut, for which
# S_sep_kept is the unconditioned separation.
PASS_DEFINITIONS = {
    "uncond": (None, None, 0),
    "n": ("n", MULT_BINS, MIN_MOLS_PER_BIN),
    "n_H": ("n_H", NH_BINS, MIN_MOLS_PER_BIN),
    "n_heavy": ("n_heavy", NHEAVY_BINS, MIN_MOLS_PER_BIN),
    "h_frac": ("h_frac", HFRAC_BINS, MIN_MOLS_PER_BIN),
    JOINT_COND: (JOINT_COND, (NHEAVY_BINS, NH_BINS), MIN_MOLS_PER_BIN),
}


def scored_constructions(out_dir):
    """
    Constructions in scores_all_slices.csv, in GRAPH_TYPES order, or all of
    GRAPH_TYPES when that file is missing.

    :param out_dir: results directory of run_scores.py
    :return: list of construction names as the config spells them
    """
    path = os.path.join(out_dir, "scores_all_slices.csv")
    if not os.path.exists(path):
        print(f"[warn] {os.path.basename(path)} not found, so every construction in "
              f"GRAPH_TYPES is bootstrapped", flush=True)
        return list(GRAPH_TYPES)
    scored = set(pd.read_csv(path, usecols=["Graph"])["Graph"].astype(str).str.upper())
    graphs = []
    for graph_type in GRAPH_TYPES:
        if graph_type.upper() in scored:
            graphs.append(graph_type)
    return graphs


def qm9_terms():
    """
    Every (slice, pair) term: one per comparison, curvature and aggregation,
    with both sides on the single population, plus the pooled comparisons of
    SEP_POOL, which repeat the terms of their members under the pooled name.

    :return: list of term dicts in the shape bs.bootstrap_construction takes
    """
    key_of = {}
    for subset, key, curvature, aggregation in NODE_SPECS + EDGE_SPECS:
        key_of[(subset, curvature, aggregation)] = key

    terms = []
    for comparison, (subset_a, subset_b) in SUBSET_PAIRS.items():
        for (subset, curvature, aggregation), key_a in key_of.items():
            if subset != subset_a:
                continue
            key_b = key_of.get((subset_b, curvature, aggregation))
            if key_b is None:
                continue
            terms.append({
                "Comparison": comparison, "Curvature": curvature,
                "Aggregation": aggregation,
                "FlavorPair": f"{subset_a}_vs_{subset_b}",
                "side_a": (POPULATION, key_a),
                "side_b": (POPULATION, key_b),
            })
    if not terms:
        raise SystemExit("no term built: SUBSET_PAIRS names no subset that appears in NODE_SPECS + EDGE_SPECS")
    pooled = []
    for pooled_name, members in SEP_POOL.items():
        for term in terms:
            if term["Comparison"] in members:
                pooled.append(dict(term, Comparison=pooled_name))
    return terms + pooled


def attach_jobs(name):
    """
    Which score CSVs of one pass get which interval columns.

    :param name: pass name
    :return: list of jobs for bs.check_and_attach
    """
    if name == "uncond":
        return [ ("scores_all_slices.csv", "per_construction", SCORE_KEYS, SEP_COL, "S_sep_kept point", bs.interval_columns("S_sep_kept", "Sep")) ]
    share_columns = {}
    for quantity in ("share", "S_sep_cond", "S_sep_kept"):
        share_columns.update(bs.interval_columns(quantity, quantity))
    return [
        (f"scores_conditioned_{name}_all_slices.csv", "per_bin", SCORE_KEYS + ["Bin"],
         SEP_COL, "Sep point", bs.interval_columns("Sep", "Sep")),
        (f"scores_conditioned_{name}_weighted.csv", "per_construction", SCORE_KEYS,
         "share", "share point", share_columns),
        (f"share_{name}.csv", "per_construction", SCORE_KEYS,
         "share", "share point", share_columns),
    ]


def main():
    start = time.time()
    boot_dir = os.path.join(OUT_DIR, "bootstrap")
    os.makedirs(boot_dir, exist_ok=True)
    bs.configure_numba(N_THREADS)

    graphs = scored_constructions(OUT_DIR)
    attach = ATTACH
    if GRAPHS is not None:
        graphs = bs.select_graphs(graphs, GRAPHS)
        attach = False
        print("[cfg] GRAPHS is set: trial run, the score CSVs are left alone", flush=True)
    if not graphs:
        raise SystemExit("no construction to bootstrap; check GRAPHS against GRAPH_TYPES")
    print(f"[cfg] B={N_BOOT}, seed={SEED}, {len(graphs)} constructions, passes={PASSES}", flush=True)

    data_dict = load_qm9(graphs, SHARD_GLOB, N_WORKERS, MAX_FILES)
    graphs = bs.select_graphs(graphs, data_dict)
    print(f"[load] {len(graphs)} constructions in {time.time() - start:.0f} s", flush=True)

    # Every construction is built on the same molecules, so the first one
    # gives the molecule ids and the bins for all of them. One stratum: the
    # two sides of a comparison are subsets of the same molecules.
    first_nodes = data_dict[graphs[0]]["nodes"]
    mol_ids = np.unique(first_nodes["mol_id"].to_numpy())
    counts = bs.draw_counts(np.zeros(mol_ids.size, dtype=np.int64), N_BOOT, SEED)
    population = np.full(mol_ids.size, POPULATION, dtype=object)
    print(f"[boot] {counts.shape[0]} resamples of {counts.shape[1]} molecules, one group; {counts.nbytes / 1e6:.0f} MB", flush=True)
    terms = qm9_terms()
    n_val = max(1, MIN_VALUES_PER_DIST)

    for name in PASSES:
        if name not in PASS_DEFINITIONS:
            print(f"[warn] unknown pass '{name}', skipped", flush=True)
            continue
        cond_var, bins, min_mols = PASS_DEFINITIONS[name]
        labels = cond_bin_labels(cond_var, bins)
        print(f"\n bootstrap: {name} ({len(labels)} bins)", flush=True)

        mol_bin = per_mol_bin(first_nodes, cond_var, bins).reindex(mol_ids)
        weights = bs.cell_weight_replicates(population, mol_bin.to_numpy(), counts)

        def cells_of(graph_type):
            binned, _ = collect_binned_distributions(
                data_dict, [graph_type], cond_var, bins, min_mols=min_mols, keep_ids=True)
            return {POPULATION: binned[graph_type]}

        tables = bs.bootstrap_pass(name, graphs, cells_of, terms, labels, weights, mol_ids,counts, n_val, LEVEL, boot_dir)
        bs.check_and_attach(OUT_DIR, attach_jobs(name), tables, attach)

    print(f"\ndone in {(time.time() - start) / 3600:.2f} h; tables in {boot_dir}", flush=True)


if __name__ == "__main__":
    main()

