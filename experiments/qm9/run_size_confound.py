#For each molecule we flip a weighted coin to decide whether to keep its
#  hydrogens, and another coin for its heavy atoms.
#    - big molecules are more likely to keep their hydrogens
#    - small molecules are more likely to keep their heavy atoms
#  So hydrogens end up coming mostly from big molecules and heavy atoms mostly
#  from small ones. Medium molecules often keep both so every size bin still
#  has some of each.

#  The bonds do not change, so we can ask:
#   without size control: does the ranking of graph constructions stop
#                          matching bond recovery, and does the complete
#                          graph suddenly look like it separates the atoms?
#    with size control:    does the ranking go back to matching bond recovery?

import argparse
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.qm9 import config, distributions
from experiments.qm9 import run_scores as base
from utils.tables import SEP_COL

EDGE_RECOVERY_SUMMARY = os.path.join(config.PATHS.RESULTS_ROOT, "qm9", "edge_recovery_summary.csv")
DEFAULT_OUT = os.path.join(config.PATHS.RESULTS_ROOT, "qm9", "size-confound")
SEED = 0
FC_KEY = config.FC_GRAPH.lower()


def keep_probability(n_atoms, center, steepness):
    """
    Chance that a molecule keeps its hydrogens.
    Its heavy atoms are kept with chance 1 minus this.

    This is an S-shaped (sigmoid) curve:
      - at size == center the chance is 0.5
      - bigger molecules get a chance closer to 1
      - smaller molecules get a chance closer to 0
      - steepness 0 gives 0.5 for every molecule, so no size effect

    :param n_atoms: molecule sizes (number of atoms)
    :param center: size where hydrogens and heavy atoms are equally likely to be kept
    :param steepness: how fast the chance changes per extra atom
    :return: array of probabilities between 0 and 1
    """
    sizes = np.asarray(n_atoms, dtype=float)
    return 1.0 / (1.0 + np.exp(-steepness * (sizes - center)))


def molecule_draws(mol_ids):
    """
    Give each molecule two "random" numbers between 0 and 1: one for its
    hydrogens and one for its heavy atoms.

    The numbers come from hashing the molecule id with crc32, so the same
    molecule always gets the same numbers, in every run and for every steepness.

    :param mol_ids: array of molecule ids
    :return: (u_hydrogen, u_heavy), two arrays of numbers in [0, 1)
    """
    u_h_list = []
    u_x_list = []
    for m in mol_ids:
        u_h_list.append(zlib.crc32(f"{SEED}-H-{m}".encode()))
        u_x_list.append(zlib.crc32(f"{SEED}-X-{m}".encode()))

    # crc32 gives a whole number from 0 to 2**32 - 1, so dividing by 2**32 gives [0, 1)
    u_h = np.array(u_h_list, dtype=np.float64) / 2 ** 32
    u_x = np.array(u_x_list, dtype=np.float64) / 2 ** 32
    return u_h, u_x


def molecule_sizes(data_dict):
    """
    Number of atoms in every molecule.

    Every construction should have every atom, so the counts should be the
    same for all constructions. We check this instead of assuming it. If they
    differ, we use the sizes from the complete graph (which has every atom)
    and print the constructions that disagreed.

    :param data_dict: {graph_type: {"nodes": DataFrame, ...}} from base.load_qm9
    :return: Series indexed by mol_id with the number of atoms
    """
    graph_types = list(data_dict)
    first = graph_types[0]
    sizes = data_dict[first]["nodes"].groupby("mol_id").size().sort_index()

    disagree = []
    for graph_type in graph_types[1:]:
        other = data_dict[graph_type]["nodes"].groupby("mol_id").size().sort_index()
        if not other.equals(sizes):
            n_diff = int((other.reindex(sizes.index) != sizes).sum())
            disagree.append(f"{graph_type} ({n_diff} molecules)")

    if not disagree:
        return sizes

    print(f"WARNING: molecule sizes differ between constructions, against '{first}': "
          f"{', '.join(disagree)}")
    if config.FC_GRAPH not in data_dict:
        raise SystemExit(f"sizes differ and '{config.FC_GRAPH}' is not loaded, so there is no "
                         f"construction that holds every atom; stopping")
    print(f"         using the sizes from '{config.FC_GRAPH}', which holds every atom")
    return data_dict[config.FC_GRAPH]["nodes"].groupby("mol_id").size().sort_index()


def make_keep_table(sizes, center, steepness):
    """
    Make the coin flips, once per molecule.

    :param sizes: Series indexed by mol_id with the number of atoms
    :param center: passed to keep_probability
    :param steepness: passed to keep_probability
    :return: DataFrame indexed by mol_id with n_atoms, p_keep_h, keep_h, keep_x
    """
    mol_ids = sizes.index.to_numpy()
    u_h, u_x = molecule_draws(mol_ids)
    p_h = keep_probability(sizes.to_numpy(), center, steepness)

    keep_h = u_h < p_h          # keep this molecule's hydrogens?
    keep_x = u_x < 1.0 - p_h    # keep this molecule's heavy atoms?

    return pd.DataFrame({"n_atoms": sizes.to_numpy(),
                         "p_keep_h": p_h,
                         "keep_h": keep_h,
                         "keep_x": keep_x},
                        index=pd.Index(mol_ids, name="mol_id"))


def print_keep_table(keep):
    """Print a few numbers that show how strong the size effect is."""
    print(f"molecules keeping H: {keep.keep_h.mean():.3f}, keeping heavy: {keep.keep_x.mean():.3f}, "
          f"both: {(keep.keep_h & keep.keep_x).mean():.3f}")
    print(f"mean size of molecules giving H: {keep.n_atoms[keep.keep_h].mean():.2f} atoms, "
          f"giving heavy atoms: {keep.n_atoms[keep.keep_x].mean():.2f} atoms "
          f"(all molecules: {keep.n_atoms.mean():.2f})")


def make_confounded_masks(original, state):
    """
    Take the original node_subset_masks function and return a new version that:
      - keeps "H_nodes" only for hydrogens of molecules where keep_h is True
      - keeps "heavy_nodes" only for heavy atoms of molecules where keep_x is True

    The decisions are read from state["keep"] (filled in after loading), so
    every construction and every size bin uses the same decisions.

    :param original: the original node_subset_masks function
    :param state: dict with "keep" (the keep table) and "calls" (a counter)
    :return: a new function that works just like `original`
    """
    def confounded_masks(nodes):
        masks = original(nodes)
        keep = state["keep"]

        mol_ids = nodes["mol_id"].to_numpy()
        missing = np.setdiff1d(np.unique(mol_ids), keep.index.to_numpy())
        if missing.size:
            raise SystemExit(f"{missing.size} molecules are not in the keep table, e.g. {missing[:5]}")

        keep_h = keep["keep_h"].reindex(mol_ids).to_numpy()
        keep_x = keep["keep_x"].reindex(mol_ids).to_numpy()

        # an atom stays in a group only if it was already in it and its molecule was kept
        old_h = masks["H_nodes"].to_numpy(dtype=bool)
        old_x = masks["heavy_nodes"].to_numpy(dtype=bool)
        masks["H_nodes"] = pd.Series(old_h & keep_h, index=nodes.index)
        masks["heavy_nodes"] = pd.Series(old_x & keep_x, index=nodes.index)

        state["calls"] += 1
        return masks

    return confounded_masks


def make_loader(original, state, center, steepness, out_dir):
    """
    Wrap base.load_qm9 so that the keep table is made right after loading,
    from the molecules that were actually loaded.

    :return: a function with the same inputs as base.load_qm9
    """
    def load_and_decide(graph_types, shard_glob, n_workers=1, max_files=None):
        data_dict = original(graph_types, shard_glob, n_workers, max_files)

        # if data is loaded again reuse the first decisions so the same atoms are kept
        if state["keep"] is not None:
            return data_dict

        sizes = molecule_sizes(data_dict)
        if center is None:
            mid = float(sizes.median())
        else:
            mid = center

        keep = make_keep_table(sizes, mid, steepness)
        print(f"size confound: center {mid:g} atoms, steepness {steepness:g}; output {out_dir}")
        print_keep_table(keep)
        keep.to_csv(os.path.join(out_dir, "keep_table.csv"))
        state["keep"] = keep
        return data_dict

    return load_and_decide


def null_columns(null, matched, curvature, aggregation):
    """
    Extra columns from the permutation null, if its file is in the run's folder.

    Labels are only shuffled inside molecules, so the null keeps which molecules
    give hydrogens and which give heavy atoms. So the score without size control
    minus its null is a size-controlled number that does not need bins.

    :param null: qm9_construction_null_<cond>.csv as a DataFrame, or None
    :param matched: share rows of this slice, merged with F1, with column key
    :return: dict of extra columns (empty when there is no null file)
    """
    if null is None:
        return {}

    part = null[(null["descriptor"] == curvature) & (null["aggregation"] == aggregation)].copy()
    part["key"] = part["construction"].str.lower()
    d = matched.merge(part[["key", "null_kept_mean", "null_cond_mean"]], on="key", how="inner")
    if len(d) < 4:
        return {}

    excess = d["S_sep_kept"] - d["null_kept_mean"]
    fc = d[d["key"] == FC_KEY]
    if len(fc):
        fc_null_before = float(fc["null_kept_mean"].iloc[0])
        fc_null_after = float(fc["null_cond_mean"].iloc[0])
    else:
        fc_null_before = np.nan
        fc_null_after = np.nan

    return {
        "rho with F1, before minus null": spearmanr(excess, d["f1"])[0],
        "complete graph, null before": fc_null_before,
        "complete graph, null after": fc_null_after,
    }


def summarize(out_dirs, edge_recovery_path, labels=None):
    """
    For hydrogen vs heavy, print how well the separation scores agree with bond
    recovery, before and after size control. Several result folders (for example
    steepness 0, 0.25 and 0.5) are put into one table, which is saved as
    confound_summary.csv in the first folder.

    "Agreement" is the Spearman rank correlation (rho): does the construction
    that separates H from heavy best also recover bonds best, and so on down
    the list?

    :param out_dirs: folders with the share_<condition>.csv files
    :param edge_recovery_path: CSV with the bond-recovery F1 of each construction
    :param labels: optional run names, one per folder (default: folder names)
    """
    f1 = pd.read_csv(edge_recovery_path)[["graph_type", "f1"]]
    f1["key"] = f1["graph_type"].str.lower()

    if not labels:
        labels = []
        for d in out_dirs:
            labels.append(os.path.basename(d.rstrip("/")))

    rows = []
    for label, out_dir in zip(labels, out_dirs):
        for cond in ["n", "n_H", "n_heavy", config.JOINT_COND]:
            path = os.path.join(out_dir, f"share_{cond}.csv")
            if not os.path.exists(path):
                continue
            share = pd.read_csv(path)
            share["key"] = share["Graph"].str.lower()

            # the permutation null if it was run into this folder
            null_path = os.path.join(out_dir, f"qm9_construction_null_{cond}.csv")
            if os.path.exists(null_path):
                null = pd.read_csv(null_path)
            else:
                null = None

            for curvature in config.CURVATURES:
                for aggregation in config.AGGREGATIONS:
                    picked = share[(share.Comparison == "H_vs_heavy")
                                   & (share.Curvature == curvature)
                                   & (share.Aggregation == aggregation)]
                    if picked.empty:
                        continue
                    matched = picked.merge(f1, on="key", how="inner")

                    unmatched = sorted(set(picked["key"]) - set(f1["key"]))
                    if unmatched:
                        print(f"WARNING {label} {cond} {curvature} {aggregation}: "
                              f"no F1 row for {unmatched}")

                    # too few constructions for a useful correlation
                    if len(matched) < 4:
                        continue

                    complete_row = matched[matched.key == config.FC_GRAPH.lower()]
                    if len(complete_row):
                        complete_before = float(complete_row["S_sep_kept"].iloc[0])
                        complete_after = float(complete_row[SEP_COL].iloc[0])
                    else:
                        complete_before = np.nan
                        complete_after = np.nan

                    # before = without size control (S_sep_kept), after = with size control (SEP_COL)
                    row = {
                        "run": label,
                        "conditioned on": cond,
                        "descriptor": curvature,
                        "aggregation": aggregation,
                        "constructions": len(matched),
                        "rho with F1, before": spearmanr(matched["S_sep_kept"], matched["f1"])[0],
                        "rho with F1, after": spearmanr(matched[SEP_COL], matched["f1"])[0],
                        "complete graph, before": complete_before,
                        "complete graph, after": complete_after,
                        "coverage": float(matched["coverage"].median()),
                    }
                    row.update(null_columns(null, matched, curvature, aggregation))
                    rows.append(row)

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 240, "display.float_format", "{:.3f}".format):
        print("\nHydrogen vs heavy with an induced size confound: agreement with bond recovery")
        print(table.to_string(index=False))

        # the rows : ORC, per-molecule mean, conditioned on n
        if not table.empty:
            focus = table[(table["descriptor"] == "orc")
                          & (table["aggregation"] == "per_mol_mean")
                          & (table["conditioned on"] == "n")]
            print("\nORC, per-molecule mean, conditioned on n (one row per run):")
            print(focus.drop(columns=["descriptor", "aggregation", "conditioned on"]).to_string(index=False))

    save_path = os.path.join(out_dirs[0], "confound_summary.csv")
    table.to_csv(save_path, index=False)
    print(f"\nwritten {save_path}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--center", type=float, default=None, help="molecule size at which H and heavy atoms are kept equally (default: median size)")
    parser.add_argument("--steepness", type=float, default=0.5, help="per atom; 0 = no size effect (the control); larger = stronger size effect")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--edge-recovery", default=EDGE_RECOVERY_SUMMARY)
    parser.add_argument("--max-files", type=int, default=None, help="read only this many shard files (smoke test)")
    parser.add_argument("--summary-only", action="store_true", help="just reread the CSVs and print the comparison")
    parser.add_argument("--compare", nargs="*", default=None, help="with --summary-only: result folders to stack into one table")
    args = parser.parse_args()

    # default steepness uses the normal folder any other steepness gets its own
    if args.steepness == 0.5:
        out_dir = args.out
    else:
        out_dir = f"{args.out}-steep{args.steepness:g}"

    if args.summary_only:
        if args.compare:
            summarize(args.compare, args.edge_recovery)
        else:
            summarize([out_dir], args.edge_recovery)
        return

    if os.path.abspath(out_dir) == os.path.abspath(base.OUT_DIR):
        raise SystemExit(f"--out is the folder of the normal run ({base.OUT_DIR}); pick another")
    os.makedirs(out_dir, exist_ok=True)

    state = {"keep": None, "calls": 0}
    base.OUT_DIR = out_dir
    if args.max_files is not None:
        base.MAX_FILES = args.max_files
    base.load_qm9 = make_loader(base.load_qm9, state, args.center, args.steepness, out_dir)
    distributions.node_subset_masks = make_confounded_masks(distributions.node_subset_masks, state)
    base.main()

    if state["calls"] == 0:
        raise SystemExit("node_subset_masks was never called: the size effect was NOT applied")
    if not os.path.exists(os.path.join(out_dir, "share_n.csv")):
        raise SystemExit(f"no share_n.csv in {out_dir}: the results went somewhere else")
    print(f"size effect applied in {state['calls']} calls of node_subset_masks")

    summarize([out_dir], args.edge_recovery)


if __name__ == "__main__":
    main()

