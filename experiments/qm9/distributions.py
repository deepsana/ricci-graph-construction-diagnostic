from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.qm9.config import AGGREGATIONS, ALL_CONCAT,CURVATURES, EDGE_SPECS,EDGE_SUBSETS,JOINT_COND, MIN_MOLS_PER_BIN, MIN_VALUES_PER_DIST,NODE_COMPARISONS,NODE_SPECS,NODE_SUBSETS, SENS_ONLY_COMPARISONS,SKIP_SENSITIVITY,SUBSET_PAIRS, dist_key
from utils.binning import bin_labels_of, bin_series, joint_bin_labels, joint_bin_series

EMPTY_VALUES = np.empty(0, dtype=np.float64)
# Stand in for "no bin" when comparing two molecule->bin maps, since NaN != NaN
NO_BIN = "__none__"


def cond_name(cond_var):
    """
    Name of a conditioning variable in the tables; the unconditioned pass is "none"
    """
    return "none" if cond_var is None else cond_var


def joint_label(heavy_label, hydrogen_label):
    """
    Cell label of the joint (n_heavy, n_H) grid, e.g. "nheavy=5-6|nH=3-5"
    """
    return f"nheavy={heavy_label}|nH={hydrogen_label}"


def cond_bin_labels(cond_var, bins):
    """
    Every bin label of a conditioning variable in plotting order

    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: list of (low, high) ranges, or (NHEAVY_BINS, NH_BINS) for JOINT_COND
    :return: ["all"] when cond_var is None; heavy-atom major for JOINT_COND
    """
    if cond_var is None:
        return ["all"]
    if cond_var != JOINT_COND:
        return bin_labels_of(bins)
    heavy_bins, hydrogen_bins = bins
    return joint_bin_labels(heavy_bins, hydrogen_bins, joint_label)


def per_mol_counts(nodes):
    """
    Atom counts of every molecule.

    :return: tuple (n, n_H, n_heavy, h_frac) of Series indexed by mol_id
    """
    is_hydrogen = nodes["is_hydrogen"].astype(bool)
    n_atoms = nodes.groupby("mol_id").size()
    n_hydrogen = nodes[is_hydrogen].groupby("mol_id").size()
    n_hydrogen = n_hydrogen.reindex(n_atoms.index, fill_value=0)
    n_heavy = n_atoms - n_hydrogen
    return n_atoms, n_hydrogen, n_heavy, n_hydrogen / n_atoms


def per_mol_bin(nodes, cond_var, bins):
    """
    Bin of every molecule for one conditioning variable. For JOINT_COND a molecule needs a bin in both variables

    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :return: object Series indexed by mol_id, None for molecules outside every bin
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
    Boolean mask of every node subset. Hydrogens and lone heavy atoms (heavy_degree -1 and 0) are neither terminal nor branch atoms

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
    Boolean mask of every edge subset.

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
    Store the distributions of one subset in one bin, in place: every ORC and FRC value, and one mean per molecule of each.
     With keep_ids, the molecule id of every value is stored next to it (<subset>_ids, <subset>_ids_mean)
    so the bootstrap can resample by molecule.

    :param cell: data_binned[graph][bin]
    :param rows: node or edge rows of this subset and bin
    :param means_only: store only the per-molecule means
    """
    mol_means = rows.groupby("mol_id")[[orc_col, frc_col]].mean()
    for curvature, col in (("orc", orc_col), ("frc", frc_col)):
        if not means_only:
            cell[dist_key(subset, curvature, "union")] = rows[col].to_numpy(dtype=np.float64)
        cell[dist_key(subset, curvature, "per_mol_mean")] = mol_means[col].to_numpy(dtype=np.float64)
    if keep_ids:
        ids = rows["mol_id"].to_numpy()
        if means_only:
            ids = np.empty(0, dtype=ids.dtype)
        cell[f"{subset}_ids"] = ids
        cell[f"{subset}_ids_mean"] = mol_means.index.to_numpy()


def empty_binned(graph_types, bin_labels):
    """data_binned with an empty array under every key of every bin."""
    keys = [spec[1] for spec in NODE_SPECS + EDGE_SPECS]
    return {graph_type: {label:
                             dict.fromkeys(keys, EMPTY_VALUES) for label in bin_labels}
            for graph_type in graph_types}


def warn_if_bins_differ(reference, mol_bin, graph_type, cond_var):
    """
    All constructions share the molecules, so their molecule->bin maps must agree
    """
    reference = reference.reindex(mol_bin.index).fillna(NO_BIN)
    current = mol_bin.fillna(NO_BIN)
    if not reference.equals(current):
        n_different = int((reference != current).sum())
        print(f"WARNING: molecule->bin map of '{graph_type}' differs from the first "
              f"graph for {n_different} molecules (cond_var={cond_name(cond_var)}); shards misaligned?")


def usable_bins(nodes, graph_type, bin_labels, cond_var, min_mols):
    """
    Which bins hold at least min_mols molecules, printed with the counts.

    :param nodes: node table with a bin column
    :return: tuple ({bin label: usable}, rows for the mols_per_bin table)
    """
    molecules = nodes.drop_duplicates("mol_id")
    bin_ok = {}
    table_rows = []
    parts = []
    for label in bin_labels:
        n_in_bin = int((molecules["bin"] == label).sum())
        bin_ok[label] = n_in_bin >= min_mols
        table_rows.append({
            "Graph": graph_type.upper(),
            "CondVar": cond_name(cond_var),
            "Bin": label,
            "n_mols": n_in_bin,
            "used": bin_ok[label],
        })
        parts.append(f"{label}: {n_in_bin}" + ("" if bin_ok[label] else "(skip)"))
    n_outside = int(molecules["bin"].isna().sum())
    print(f"[{graph_type:30s}] molecules per {cond_name(cond_var)}-bin  {',  '.join(parts)}, outside bins: {n_outside}")
    return bin_ok, table_rows


def edges_with_endpoints(nodes, edges, mol_bin):
    """
    Binned edges with the hydrogen flag of both endpoints. Edges with a missing endpoint are dropped here, because astype(bool) in
    edge_subset_masks would turn NaN into True.
    """
    node_lookup = nodes[["mol_id", "node_id", "is_hydrogen"]]
    src_lookup = node_lookup.rename(columns={"node_id": "src", "is_hydrogen": "src_is_hydrogen"})
    dst_lookup = node_lookup.rename(columns={"node_id": "dst", "is_hydrogen": "dst_is_hydrogen"})
    edge_rows = edges.merge(src_lookup, on=["mol_id", "src"], how="left")
    edge_rows = edge_rows.merge(dst_lookup, on=["mol_id", "dst"], how="left")
    edge_rows["bin"] = edge_rows["mol_id"].map(mol_bin)
    edge_rows = edge_rows[edge_rows["bin"].notna()]
    return edge_rows.dropna(subset=["src_is_hydrogen", "dst_is_hydrogen"])


def fill_subsets(graph_cells, rows, masks, subsets, bin_labels, bin_ok, orc_col, frc_col,keep_ids):
    """
    Fill every subset of every usable bin of one construction
    """
    for label in bin_labels:
        if not bin_ok[label]:
            continue
        in_bin = rows["bin"] == label
        for subset in subsets:
            # The all-atom / all-edge subsets store only their means unless ALL_CONCAT
            means_only = subset in SENS_ONLY_COMPARISONS and not ALL_CONCAT
            fill_subset(graph_cells[label], subset, rows[masks[subset] & in_bin], orc_col, frc_col, means_only=means_only, keep_ids=keep_ids)


def collect_binned_distributions(data_dict, graph_types, cond_var, bins, min_mols=MIN_MOLS_PER_BIN, keep_ids=False):
    """
    Split every curvature distribution by subset and by bin of a conditioning variable.
    The minimum count applies to the molecules in a bin, since both subsets of a comparison come from the same molecules.

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :param cond_var: None, "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges for cond_var, or None
    :param min_mols: bins with fewer molecules than this are left empty
    :param keep_ids: store the molecule id of every value next to it (see fill_subset)
    :return: tuple (data_binned, mols_per_bin DataFrame)
    """
    bin_labels = cond_bin_labels(cond_var, bins)
    data_binned = empty_binned(graph_types, bin_labels)
    mols_per_bin_rows = []
    reference_bin = None

    for graph_type in graph_types:
        if graph_type not in data_dict:
            print(f"WARNING: '{graph_type}' not loaded; skipping.")
            continue
        nodes = data_dict[graph_type]["nodes"]
        edges = data_dict[graph_type]["edges"]
        if nodes.empty:
            continue

        mol_bin = per_mol_bin(nodes, cond_var, bins)
        if reference_bin is None:
            reference_bin = mol_bin
        else:
            warn_if_bins_differ(reference_bin, mol_bin, graph_type, cond_var)

        nodes = nodes.copy()
        nodes["bin"] = nodes["mol_id"].map(mol_bin)
        bin_ok, table_rows = usable_bins(nodes, graph_type, bin_labels, cond_var, min_mols)
        mols_per_bin_rows.extend(table_rows)

        binned_nodes = nodes[nodes["bin"].notna()]
        fill_subsets(data_binned[graph_type], binned_nodes, node_subset_masks(binned_nodes), NODE_SUBSETS, bin_labels, bin_ok, "orc_curvature", "frc_curvature", keep_ids)

        if edges.empty:
            continue
        edge_rows = edges_with_endpoints(nodes, edges, mol_bin)
        fill_subsets(data_binned[graph_type], edge_rows, edge_subset_masks(edge_rows), EDGE_SUBSETS, bin_labels, bin_ok, "orc_edge_curvature", "frc_edge_curvature", keep_ids)

    return data_binned, pd.DataFrame(mols_per_bin_rows)


def across_graph_rows(data_binned, graph_types, label, cond_var):
    """
    W1 of the same subset between every pair of constructions in one bin (the S_sens stage).

    :return: tuple (node rows, edge rows)
    """
    node_rows, edge_rows = [], []
    for graph_a, graph_b in combinations(graph_types, 2):
        dist_a = data_binned[graph_a][label]
        dist_b = data_binned[graph_b][label]
        for subset, key, curvature, aggregation in NODE_SPECS + EDGE_SPECS:
            values_a = dist_a[key]
            values_b = dist_b[key]
            if len(values_a) < MIN_VALUES_PER_DIST or len(values_b) < MIN_VALUES_PER_DIST:
                continue
            row = {
                "Graph_A": graph_a.upper(),
                "Graph_B": graph_b.upper(),
                "CondVar": cond_name(cond_var),
                "Bin": label,
                "Subset": subset,
                "Curvature": curvature,
                "Aggregation": aggregation,
                "n_A": len(values_a),
                "n_B": len(values_b),
                "W_Distance": wasserstein_distance(values_a, values_b),
            }
            (node_rows if subset in NODE_SUBSETS else edge_rows).append(row)
    return node_rows, edge_rows


def across_subset_rows(data_binned, graph_types, label, cond_var):
    """
    W1 between the two subsets of every comparison, within each construction
    in one bin (the S_sep stage).

    :return: tuple (node rows, edge rows)
    """
    node_rows, edge_rows = [], []
    for graph_type in graph_types:
        cell = data_binned[graph_type][label]
        for comparison, (subset_a, subset_b) in SUBSET_PAIRS.items():
            for curvature in CURVATURES:
                for aggregation in AGGREGATIONS:
                    values_a = cell[dist_key(subset_a, curvature, aggregation)]
                    values_b = cell[dist_key(subset_b, curvature, aggregation)]
                    if len(values_a) < MIN_VALUES_PER_DIST or len(values_b) < MIN_VALUES_PER_DIST:
                        continue
                    row = {
                        "Graph": graph_type.upper(),
                        "Comparison": comparison,
                        "Subset_A": subset_a,
                        "Subset_B": subset_b,
                        "CondVar": cond_name(cond_var),
                        "Bin": label,
                        "Curvature": curvature,
                        "Aggregation": aggregation,
                        "n_A": len(values_a),
                        "n_B": len(values_b),
                        "W_Distance": wasserstein_distance(values_a, values_b),
                    }
                    (node_rows if comparison in NODE_COMPARISONS else edge_rows).append(row)
    return node_rows, edge_rows


def compute_binned_w1_tables(data_binned, graph_types, cond_var, bins, across_graphs=True):
    """
    Pairwise W1 tables inside every bin. A pair is skipped when either
    distribution has fewer than MIN_VALUES_PER_DIST values.

    :param data_binned: first output of collect_binned_distributions
    :param across_graphs: also compare every pair of constructions (the S_sens stage)
    :return: tuple of DataFrames (across-graph nodes, across-graph edges,
     across-subset nodes, across-subset edges); the first two feed S_sens and the last two S_sep
    """
    graph_node_rows, graph_edge_rows = [], []
    subset_node_rows, subset_edge_rows = [], []
    for label in cond_bin_labels(cond_var, bins):
        if across_graphs:
            node_rows, edge_rows = across_graph_rows(data_binned, graph_types, label, cond_var)
            graph_node_rows.extend(node_rows)
            graph_edge_rows.extend(edge_rows)
        node_rows, edge_rows = across_subset_rows(data_binned, graph_types, label, cond_var)
        subset_node_rows.extend(node_rows)
        subset_edge_rows.extend(edge_rows)

    return (pd.DataFrame(graph_node_rows), pd.DataFrame(graph_edge_rows), pd.DataFrame(subset_node_rows), pd.DataFrame(subset_edge_rows))


def analyze_across_construction(data_dict, graph_types):
    """
    Unconditioned W1 tables: the binned pipeline with the single bin "all"
    and no minimum molecule count.

    :return: tuple of DataFrames, as compute_binned_w1_tables
    """
    data_binned, _ = collect_binned_distributions(data_dict, graph_types, None, None, min_mols=0)
    return compute_binned_w1_tables(data_binned, graph_types, None, None, across_graphs=not SKIP_SENSITIVITY)
