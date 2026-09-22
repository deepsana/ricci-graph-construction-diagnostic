"""
Curvature distributions of the molecules split by subset and by bin of a
conditioning variable, and the W1 tables computed inside every bin.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.qm9.config import AGGREGATIONS, ALL_CONCAT,CURVATURES, EDGE_SPECS,EDGE_SUBSETS,JOINT_COND, MIN_MOLS_PER_BIN, MIN_VALUES_PER_DIST,NODE_COMPARISONS,NODE_SPECS,NODE_SUBSETS, SENS_ONLY_COMPARISONS,SKIP_SENSITIVITY,SUBSET_PAIRS, dist_key
from utils.binning import bin_labels_of, bin_series, joint_bin_labels, joint_bin_series

EMPTY_VALUES = np.empty(0, dtype=np.float64)


def joint_label(heavy_label, hydrogen_label):
    """
    Label of one cell of the joint (n_heavy, n_H) grid.

    :param heavy_label: label of the heavy-atom bin
    :param hydrogen_label: label of the hydrogen bin
    :return: "nheavy=<heavy_label>|nH=<hydrogen_label>"
    """
    return f"nheavy={heavy_label}|nH={hydrogen_label}"


def cond_bin_labels(cond_var, bins):
    """
    Every bin label of a conditioning variable in plotting order.

    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: list of (low, high) ranges, or (NHEAVY_BINS, NH_BINS) for JOINT_COND
    :return: list of labels; ["all"] when cond_var is None, and for
        JOINT_COND the full grid ordered by the heavy-atom bin first
    """
    if cond_var is None:
        return ["all"]
    if cond_var != JOINT_COND:
        return bin_labels_of(bins)
    heavy_bins, hydrogen_bins = bins
    return joint_bin_labels(heavy_bins, hydrogen_bins, joint_label)


def per_mol_counts(nodes):
    """
    Count the atoms of every molecule.

    :param nodes: node table with mol_id and is_hydrogen
    :return: tuple (n, n_H, n_heavy, h_frac) of Series indexed by mol_id
    """
    is_hydrogen = nodes["is_hydrogen"].astype(bool)
    n_atoms = nodes.groupby("mol_id").size()
    n_hydrogen = nodes[is_hydrogen].groupby("mol_id").size()
    n_hydrogen = n_hydrogen.reindex(n_atoms.index, fill_value=0)
    n_heavy = n_atoms - n_hydrogen
    h_frac = n_hydrogen / n_atoms
    return n_atoms, n_hydrogen, n_heavy, h_frac


def per_mol_bin(nodes, cond_var, bins):
    """
    Bin of every molecule for one conditioning variable.

    :param nodes: node table with mol_id and is_hydrogen
    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: list of (low, high) ranges, or (NHEAVY_BINS, NH_BINS) for
        JOINT_COND, where a molecule needs a bin in both variables
    :return: object Series indexed by mol_id with the bin label, or None
        for molecules outside every bin
    """
    n_atoms, n_hydrogen, n_heavy, h_frac = per_mol_counts(nodes)

    if cond_var is None:
        return pd.Series("all", index=n_atoms.index, dtype=object)

    if cond_var == JOINT_COND:
        heavy_bins, hydrogen_bins = bins
        return joint_bin_series(n_heavy, heavy_bins, n_hydrogen, hydrogen_bins, joint_label)

    counts = {"n": n_atoms, "n_H": n_hydrogen, "n_heavy": n_heavy, "h_frac": h_frac}
    if cond_var not in counts:
        raise ValueError(f"cond_var must be one of {list(counts)} or '{JOINT_COND}', got '{cond_var}'")
    return bin_series(counts[cond_var], bins)


def node_subset_masks(nodes):
    """
    Boolean mask for every node subset. Hydrogens and lone heavy atoms
    (heavy_degree -1 and 0) are neither terminal nor branch atoms.

    :param nodes: node table with is_hydrogen, in_ring and heavy_degree
    :return: {subset name: boolean Series aligned to nodes}
    """
    is_hydrogen = nodes["is_hydrogen"].astype(bool)
    in_ring = nodes["in_ring"].astype(bool)
    heavy_degree = nodes["heavy_degree"]
    return {
        "all_nodes": pd.Series(True, index=nodes.index),
        "H_nodes": is_hydrogen,
        "heavy_nodes": ~is_hydrogen,
        "ring_nodes": ~is_hydrogen & in_ring,
        "chain_nodes": ~is_hydrogen & ~in_ring,
        "terminal_nodes": heavy_degree == 1,
        "branch_nodes": heavy_degree >= 2,
    }


def edge_subset_masks(edges):
    """
    Boolean mask for every edge subset.

    :param edges: edge table with is_true_bond, src_is_hydrogen and dst_is_hydrogen
    :return: {subset name: boolean Series aligned to edges}
    """
    src_hydrogen = edges["src_is_hydrogen"].astype(bool)
    dst_hydrogen = edges["dst_is_hydrogen"].astype(bool)
    is_bond = edges["is_true_bond"].astype(bool)
    return {
        "all_edges": pd.Series(True, index=edges.index),
        "bond_edges": is_bond,
        "nonbond_edges": ~is_bond,
        "HH_edges": src_hydrogen & dst_hydrogen,
        "Hheavy_edges": src_hydrogen != dst_hydrogen,
        "heavyheavy_edges": ~src_hydrogen & ~dst_hydrogen,
    }


def fill_subset(cell, subset, rows, orc_col, frc_col, means_only=False, keep_ids=False):
    """
    Store the concatenated ORC and FRC values of one subset in one bin and
    their per-molecule means. With keep_ids, the molecule id of every value
    is stored next to it (<subset>_ids and <subset>_ids_mean), which the
    bootstrap needs to resample by molecule.

    :param cell: data_binned[graph][bin]
    :param subset: subset name
    :param rows: node or edge rows of this subset and bin
    :param orc_col: name of the ORC column in rows
    :param frc_col: name of the FRC column in rows
    :param means_only: store only the per-molecule means
    :param keep_ids: also store the molecule ids
    :return: None; cell is modified in place
    """
    mol_means = rows.groupby("mol_id")[[orc_col, frc_col]].mean()
    for curvature, col in (("orc", orc_col), ("frc", frc_col)):
        if not means_only:
            concat_key = dist_key(subset, curvature, "union")
            cell[concat_key] = rows[col].to_numpy(dtype=np.float64)
        mean_key = dist_key(subset, curvature, "per_mol_mean")
        cell[mean_key] = mol_means[col].to_numpy(dtype=np.float64)
    if keep_ids:
        ids = rows["mol_id"].to_numpy()
        if means_only:
            ids = np.empty(0, dtype=ids.dtype)
        cell[f"{subset}_ids"] = ids
        cell[f"{subset}_ids_mean"] = mol_means.index.to_numpy()


def collect_binned_distributions(data_dict, graph_types, cond_var, bins, min_mols=MIN_MOLS_PER_BIN, keep_ids=False):
    """
    Split every curvature distribution by the bins of a conditioning
    variable. The minimum count applies to the molecules in a bin, since
    both subsets of a comparison come from the same molecules.

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :param graph_types: constructions to process
    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges for cond_var, or None
    :param min_mols: bins with fewer molecules than this are left empty
    :param keep_ids: store the molecule id of every value next to it (see fill_subset)
    :return: tuple (data_binned, mols_per_bin), where data_binned is
        {graph: {bin: {dist_key: array}}}
    """
    bin_labels = cond_bin_labels(cond_var, bins)
    cond_name = cond_var
    if cond_var is None:
        cond_name = "none"

    data_binned = {}
    for graph_type in graph_types:
        data_binned[graph_type] = {}
        for label in bin_labels:
            cell = {}
            for spec in NODE_SPECS + EDGE_SPECS:
                cell[spec[1]] = EMPTY_VALUES
            data_binned[graph_type][label] = cell

    mols_per_bin_rows = []
    ref_mol_bin = None

    for graph_type in graph_types:
        if graph_type not in data_dict:
            print(f"WARNING: '{graph_type}' not loaded; skipping.")
            continue
        nodes = data_dict[graph_type]["nodes"]
        edges = data_dict[graph_type]["edges"]
        if nodes.empty:
            continue

        # All constructions share the molecules, so the maps must agree.
        mol_bin = per_mol_bin(nodes, cond_var, bins)
        if ref_mol_bin is None:
            ref_mol_bin = mol_bin
        else:
            reference = ref_mol_bin.reindex(mol_bin.index).fillna("__none__")
            current = mol_bin.fillna("__none__")
            if not reference.equals(current):
                n_different = int((reference != current).sum())
                print(f"WARNING: molecule->bin map of '{graph_type}' differs from the first graph for {n_different} molecules (cond_var={cond_name}); shards misaligned?")

        nodes = nodes.copy()
        nodes["bin"] = nodes["mol_id"].map(mol_bin)

        molecules = nodes.drop_duplicates("mol_id")
        bin_ok = {}
        parts = []
        for label in bin_labels:
            n_in_bin = int((molecules["bin"] == label).sum())
            bin_ok[label] = n_in_bin >= min_mols
            mols_per_bin_rows.append({
                "Graph": graph_type.upper(),
                "CondVar": cond_name,
                "Bin": label,
                "n_mols": n_in_bin,
                "used": bin_ok[label],
            })
            if bin_ok[label]:
                parts.append(f"{label}: {n_in_bin}")
            else:
                parts.append(f"{label}: {n_in_bin}(skip)")
        n_outside = int(molecules["bin"].isna().sum())
        summary = ",  ".join(parts)
        print(f"[{graph_type:30s}] molecules per {cond_name}-bin  {summary},"
              f"  outside bins: {n_outside}")

        binned_nodes = nodes[nodes["bin"].notna()]
        node_masks = node_subset_masks(binned_nodes)
        for label in bin_labels:
            if not bin_ok[label]:
                continue
            in_bin = binned_nodes["bin"] == label
            for subset in NODE_SUBSETS:
                subset_rows = binned_nodes[node_masks[subset] & in_bin]
                means_only = subset in SENS_ONLY_COMPARISONS and not ALL_CONCAT
                fill_subset(data_binned[graph_type][label], subset, subset_rows, "orc_curvature", "frc_curvature", means_only=means_only, keep_ids=keep_ids)

        if edges.empty:
            continue

        node_lookup = nodes[["mol_id", "node_id", "is_hydrogen"]]
        src_lookup = node_lookup.rename(columns={"node_id": "src", "is_hydrogen": "src_is_hydrogen"})
        dst_lookup = node_lookup.rename(columns={"node_id": "dst", "is_hydrogen": "dst_is_hydrogen"})
        edge_rows = edges.merge(src_lookup, on=["mol_id", "src"], how="left")
        edge_rows = edge_rows.merge(dst_lookup, on=["mol_id", "dst"], how="left")
        edge_rows["bin"] = edge_rows["mol_id"].map(mol_bin)
        # Missing endpoints are dropped first, because astype(bool) in the
        # masks would turn NaN into True.
        edge_rows = edge_rows[edge_rows["bin"].notna()]
        edge_rows = edge_rows.dropna(subset=["src_is_hydrogen", "dst_is_hydrogen"])
        edge_masks = edge_subset_masks(edge_rows)
        for label in bin_labels:
            if not bin_ok[label]:
                continue
            in_bin = edge_rows["bin"] == label
            for subset in EDGE_SUBSETS:
                subset_rows = edge_rows[edge_masks[subset] & in_bin]
                means_only = subset in SENS_ONLY_COMPARISONS and not ALL_CONCAT
                fill_subset(data_binned[graph_type][label], subset, subset_rows, "orc_edge_curvature", "frc_edge_curvature", means_only=means_only, keep_ids=keep_ids)

    mols_per_bin = pd.DataFrame(mols_per_bin_rows)
    return data_binned, mols_per_bin


def compute_binned_w1_tables(data_binned, graph_types, cond_var, bins,
                             min_values=MIN_VALUES_PER_DIST, across_graphs=True):
    """
    Pairwise W1 tables inside every bin. A pair is skipped when either
    distribution has fewer than min_values entries.

    :param data_binned: first output of collect_binned_distributions
    :param graph_types: constructions; every unordered pair is compared
    :param cond_var: conditioning variable name, or None
    :param bins: bins for cond_var, or None
    :param min_values: minimum values per distribution for a W1
    :param across_graphs: also compare every pair of constructions (the S_sens stage)
    :return: tuple of DataFrames (across-graph nodes, across-graph edges,
        across-subset nodes, across-subset edges); the first two feed
        S_sens and the last two feed S_sep
    """
    bin_labels = cond_bin_labels(cond_var, bins)
    cond_name = cond_var
    if cond_var is None:
        cond_name = "none"

    graph_node_rows = []
    graph_edge_rows = []
    subset_node_rows = []
    subset_edge_rows = []

    for label in bin_labels:
        graph_pairs = ()
        if across_graphs:
            graph_pairs = combinations(graph_types, 2)
        for graph_a, graph_b in graph_pairs:
            dist_a = data_binned[graph_a][label]
            dist_b = data_binned[graph_b][label]
            for subset, key, curvature, aggregation in NODE_SPECS + EDGE_SPECS:
                values_a = dist_a[key]
                values_b = dist_b[key]
                if len(values_a) < min_values or len(values_b) < min_values:
                    continue
                row = {
                    "Graph_A": graph_a.upper(),
                    "Graph_B": graph_b.upper(),
                    "CondVar": cond_name,
                    "Bin": label,
                    "Subset": subset,
                    "Curvature": curvature,
                    "Aggregation": aggregation,
                    "n_A": len(values_a),
                    "n_B": len(values_b),
                    "W_Distance": wasserstein_distance(values_a, values_b),
                }
                if subset in NODE_SUBSETS:
                    graph_node_rows.append(row)
                else:
                    graph_edge_rows.append(row)

        for graph_type in graph_types:
            cell = data_binned[graph_type][label]
            for comparison in SUBSET_PAIRS:
                subset_a, subset_b = SUBSET_PAIRS[comparison]
                for curvature in CURVATURES:
                    for aggregation in AGGREGATIONS:
                        values_a = cell[dist_key(subset_a, curvature, aggregation)]
                        values_b = cell[dist_key(subset_b, curvature, aggregation)]
                        if len(values_a) < min_values or len(values_b) < min_values:
                            continue
                        row = {
                            "Graph": graph_type.upper(),
                            "Comparison": comparison,
                            "Subset_A": subset_a,
                            "Subset_B": subset_b,
                            "CondVar": cond_name,
                            "Bin": label,
                            "Curvature": curvature,
                            "Aggregation": aggregation,
                            "n_A": len(values_a),
                            "n_B": len(values_b),
                            "W_Distance": wasserstein_distance(values_a, values_b),
                        }
                        if comparison in NODE_COMPARISONS:
                            subset_node_rows.append(row)
                        else:
                            subset_edge_rows.append(row)

    return (pd.DataFrame(graph_node_rows), pd.DataFrame(graph_edge_rows), pd.DataFrame(subset_node_rows), pd.DataFrame(subset_edge_rows))


def analyze_across_construction(data_dict, graph_types):
    """
    Unconditioned W1 tables, from the binned pipeline with a single bin
    "all" and no minimum molecule count.

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :param graph_types: constructions to process
    :return: tuple of DataFrames (across-graph nodes, across-graph edges,
        across-subset nodes, across-subset edges)
    """
    data_binned = collect_binned_distributions(data_dict, graph_types, None, None, min_mols=0)[0]
    return compute_binned_w1_tables(data_binned, graph_types, None, None, across_graphs=not SKIP_SENSITIVITY)

