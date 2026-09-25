import gc
import os
import re
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.jetset.config import (AGGREGATIONS, ALL_PAIR_LABELS,COMPARISON_LABELS,CURVATURES,EDGE_COMPARISONS,
    FLAVORPAIR_LABELS, GRAPH_TYPES, MIN_JETS_PER_BIN, MIN_VALUES_PER_SIDE, NODE_COMPARISONS, PAIR_FLAVORS, PAIR_LABEL,
    SPEC_KEY, WITHIN_COMPARISONS,WITHIN_PAIR_LABEL)
from experiments.jetset.constructions import DUPLICATE_GRAPHS, STYLING, pair_style
from experiments.jetset.distributions import collect_binned_distributions, compute_binned_w1_tables, cond_bin_labels, jets_per_flavor
from plotting.conditioned import plot_pair_scores_vs_bin, plot_pair_scores_vs_bin_by_graph, plot_scores_vs_bin
from utils.binning import order_bins
from utils.memory import rss_gb
from utils.tables import SCORE_KEYS, SEP_COL, apply_value_guard, safe_divide, save_csv, upper_names


PAIR_KEYS = ["Graph", "FlavorPair", "Comparison", "Curvature", "Aggregation"]
PAIR_SORT = ["Comparison", "Curvature", "Aggregation", "FlavorPair", "Graph"]


def extract_int_flavor(label):
    """
    Integer flavor from a label, so "Flavor 5" and 5 both give 5
    """
    match = re.search(r"(\d+)", str(label))
    if not match:
        raise ValueError(f"Could not parse flavor int from: {label}")
    return int(match.group(1))


def pair_labels_for(comparison):
    """Population pairs of a subset: (PV, SV) within one flavor, else the light-vs-heavy pairs."""
    if comparison in WITHIN_COMPARISONS:
        return [WITHIN_PAIR_LABEL]
    return FLAVORPAIR_LABELS


def add_canonical_flavorpair(df):
    """
    Copy of a W1 table with an unordered FlavorPair label such as
    "b_vs_light" on every row. Pairs missing from PAIR_LABEL get the string
    of the sorted flavor tuple; within-flavor rows get WITHIN_PAIR_LABEL.
    """
    df = df.copy()
    if df.empty:
        df["FlavorPair"] = pd.Series(dtype=object)
        return df

    flavor_a = df["Flavor_A"].map(extract_int_flavor).to_numpy()
    flavor_b = df["Flavor_B"].map(extract_int_flavor).to_numpy()
    pairs = pd.Series(list(zip(np.minimum(flavor_a, flavor_b), np.maximum(flavor_a, flavor_b))), index=df.index)
    df["FlavorPair"] = pairs.map(PAIR_LABEL).fillna(pairs.astype(str))
    if "Comparison" in df.columns:
        df.loc[df["Comparison"].isin(WITHIN_COMPARISONS), "FlavorPair"] = WITHIN_PAIR_LABEL
    return df


def scored_pair_rows(table, flavorpair_labels):
    """
    The W1 rows that enter a score: enough values on both sides, not a
    duplicate construction, one of the requested pairs, and a finite W1.
    """
    rows = apply_value_guard(table, MIN_VALUES_PER_SIDE)
    rows = rows[~rows["Graph"].isin(upper_names(DUPLICATE_GRAPHS))]
    rows = add_canonical_flavorpair(rows)
    rows = rows[rows["FlavorPair"].isin(flavorpair_labels)]
    return rows.dropna(subset=["W_Distance"])


def compute_separation_score(df_flavor_pairs, graph_types, comparison, curvature, aggregation,
                             flavorpair_labels, cond_bin=None):
    """
    S_sep(g): mean W1 over the requested population pairs, within one
    construction. Higher means the populations are easier to tell apart.

    :param df_flavor_pairs: across-flavor W1 table
    :param flavorpair_labels: pair labels to average over
    :param cond_bin: bin label to restrict to, or None for the unconditioned score
    :return: DataFrame indexed by upper-cased graph with columns SEP_COL and n_terms
    """
    graph_index = upper_names(graph_types)
    in_slice = ((df_flavor_pairs["Comparison"] == comparison) & (df_flavor_pairs["Curvature"] == curvature) & (df_flavor_pairs["Aggregation"] == aggregation))
    rows = df_flavor_pairs[in_slice]
    if cond_bin is not None:
        rows = rows[rows["Bin"] == cond_bin]
    rows = scored_pair_rows(rows, flavorpair_labels)

    if rows.empty:
        return pd.DataFrame({SEP_COL: np.nan, "n_terms": 0}, index=graph_index)

    grouped = rows.groupby("Graph")
    result = pd.DataFrame({
        SEP_COL: grouped["W_Distance"].mean(),
        "n_terms": grouped["FlavorPair"].nunique(),
    }).reindex(graph_index)
    result["n_terms"] = result["n_terms"].fillna(0).astype(int)
    return result


def pair_scores(df_across_flavors, graph_types, flavorpair_labels):
    """
    S_sep(g, pair): the W1 of every requested pair, before averaging over
    pairs. CondVar and Bin are kept when the table is a binned one.

    :return: DataFrame with Graph, [CondVar, Bin,] FlavorPair, the slice
        columns, SEP_COL, value counts, and per-side mean and std
    """
    if df_across_flavors is None or df_across_flavors.empty:
        return pd.DataFrame()

    rows = df_across_flavors[df_across_flavors["Graph"].isin(upper_names(graph_types))]
    rows = scored_pair_rows(rows, flavorpair_labels)
    if rows.empty:
        return pd.DataFrame()

    wanted = ["Graph", "CondVar", "Bin", "FlavorPair", "Comparison", "Curvature", "Aggregation",
              "W_Distance", "n_A", "n_B", "mean_A", "std_A", "mean_B", "std_B"]
    cols = [col for col in wanted if col in rows.columns]
    return rows[cols].rename(columns={"W_Distance": SEP_COL}).reset_index(drop=True)


def cell_jets_from_table(jets_per_bin):
    """
    {(flavor, bin label): number of jets} from jets_per_bin. The counts are
    the same for every construction, so the first one is read.
    """
    first_rows = jets_per_bin[jets_per_bin["Graph"] == jets_per_bin["Graph"].iloc[0]]
    return {(extract_int_flavor(flavor), str(label)): int(n_jets)
            for flavor, label, n_jets
            in zip(first_rows["Flavor"], first_rows["Bin"], first_rows["n_jets"])}


def pair_sides(pair, comparison, curvature, aggregation):
    """
    What the two sides of a pair are.

    :return: tuple (flavors of the pair, key of side A, key of side B), or
        None for a pair this analysis does not define. A within-flavor pair
        has one flavor and compares its PV and SV nodes.
    """
    if comparison in WITHIN_COMPARISONS:
        return ([WITHIN_COMPARISONS[comparison]],  SPEC_KEY[("PV_nodes", curvature, aggregation)], SPEC_KEY[("SV_nodes", curvature, aggregation)])
    if pair in PAIR_FLAVORS:
        key = SPEC_KEY[(comparison, curvature, aggregation)]
        return list(PAIR_FLAVORS[pair]), key, key
    return None


def weighted_bins(pair_bins, bin_labels, pair_flavors, cell_jets):
    """
    Bins that enter S_sep_cond, with their weights and W1.

    w_b is the number of jets of the pair's flavors in bin b; a bin is used
    when it has a finite W1 and a positive weight.

    :param pair_bins: per-bin rows of one (graph, pair, slice)
    :return: tuple (used labels, weights, W1 values)
    """
    w1_by_bin = pair_bins.set_index(pair_bins["Bin"].astype(str))[SEP_COL]
    used_bins, weights, w1_values = [], [], []
    for label in bin_labels:
        if label not in w1_by_bin.index:
            continue
        weight = sum(cell_jets.get((flavor, label), 0) for flavor in pair_flavors)
        w1 = float(w1_by_bin[label])
        if np.isfinite(w1) and weight > 0:
            used_bins.append(label)
            weights.append(weight)
            w1_values.append(w1)
    return used_bins, np.array(weights, dtype=float), np.array(w1_values, dtype=float)


def matched_pair_summary(data_binned, pair_bins_df, graph_types, jets_per_bin, jet_totals,
                         cond_var, bins):
    """
    Per (graph, pair, slice): the conditioned separation, the unconditioned
    separation on the same jets, and the share.

        S_sep_cond = sum_b w_b W1(A_b, B_b) / sum_b w_b
        S_sep_kept = W1(concat_b A_b, concat_b B_b)
        preserved_sep = S_sep_cond / S_sep_kept
        share         = 1 - preserved_sep

    b runs over the bins with a W1 row for this slice (see weighted_bins).
    The bins partition the jets, so S_sep_kept comes straight from
    data_binned. Coverage is the kept jets over all jets of the pair's
    flavors. The _A columns hold the lighter flavor of the pair.

    :param data_binned: first output of collect_binned_distributions
    :param pair_bins_df: output of pair_scores on the binned tables
    :param jets_per_bin: second output of collect_binned_distributions
    :param jet_totals: {flavor: number of jets in the sample}
    :return: DataFrame with S_sep_cond, S_sep_kept, share, preserved_sep, the bins used, value counts and coverage
    """
    if pair_bins_df is None or pair_bins_df.empty:
        return pd.DataFrame()

    bin_labels = cond_bin_labels(cond_var, bins)
    cell_jets = cell_jets_from_table(jets_per_bin)
    graph_by_upper = {graph_type.upper(): graph_type for graph_type in graph_types}

    rows = []
    for group_key, group in pair_bins_df.groupby(PAIR_KEYS, sort=False):
        graph_upper, pair, comparison, curvature, aggregation = group_key
        graph_type = graph_by_upper.get(graph_upper)
        sides = pair_sides(pair, comparison, curvature, aggregation)
        if graph_type is None or sides is None:
            continue
        pair_flavors, key_a, key_b = sides
        used_bins, weights, w1_values = weighted_bins(group, bin_labels, pair_flavors, cell_jets)
        if not used_bins:
            continue

        s_sep_cond = float((w1_values * weights).sum() / weights.sum())
        cells = data_binned[graph_type]
        values_a = np.concatenate([cells[pair_flavors[0]][label][key_a] for label in used_bins]).astype(np.float64)
        values_b = np.concatenate([cells[pair_flavors[-1]][label][key_b]  for label in used_bins]).astype(np.float64)
        s_sep_kept = float(wasserstein_distance(values_a, values_b))
        preserved = float(preserved_sep(s_sep_cond, s_sep_kept))

        n_jets_kept = int(weights.sum())
        n_jets_total = sum(int(jet_totals.get(flavor, 0)) for flavor in pair_flavors)
        coverage = n_jets_kept / n_jets_total if n_jets_total > 0 else np.nan

        rows.append({
            "Graph": graph_upper,
            "FlavorPair": pair,
            "Comparison": comparison,
            "Curvature": curvature,
            "Aggregation": aggregation,
            "CondVar": cond_var,
            "S_sep_cond": s_sep_cond,
            "S_sep_kept": s_sep_kept,
            "share": 1.0 - preserved,
            "preserved_sep": preserved,
            "n_bins": len(used_bins),
            "bins": ";".join(used_bins),
            "n_A": int(values_a.size),
            "n_B": int(values_b.size),
            "n_jets_kept": n_jets_kept,
            "n_jets_total": n_jets_total,
            "coverage": coverage,
        })
    return pd.DataFrame(rows)


def run_conditioned_slice(flavor_table, graph_types, comparison, curvature, aggregation,
                          cond_var, bins):
    """
    S_sep of one slice in every bin; a row is kept where the score exists.

    :param flavor_table: binned across-flavor W1 table
    :return: DataFrame with one row per (graph, bin)
    """
    rows = []
    for label in cond_bin_labels(cond_var, bins):
        sep = compute_separation_score(flavor_table, graph_types, comparison, curvature, aggregation, pair_labels_for(comparison), cond_bin=label)
        for graph in sep.index:
            s_sep = sep.loc[graph, SEP_COL]
            if np.isnan(s_sep):
                continue
            rows.append({"Graph": graph,
                         "CondVar": cond_var,
                         "Bin": label,
                         SEP_COL: s_sep,
                         "n_terms_sep": int(sep.loc[graph, "n_terms"]),
                         "Comparison": comparison,
                         "Curvature": curvature,
                         "Aggregation": aggregation
                         })
    return pd.DataFrame(rows)


def weighted_summary(pair_summary):
    """
    Conditioned analogue of the unconditioned S_sep: the mean over pairs of
    the jet-weighted S_sep_cond, with S_sep_kept, the share and preserved_sep
    of those two means, and the pooled coverage.

    :param pair_summary: output of matched_pair_summary
    :return: one row per SCORE_KEYS
    """
    sep_cols = [SEP_COL, "S_sep_kept", "share", "preserved_sep", "coverage", "n_pairs", "n_jets_kept", "n_jets_total"]
    if pair_summary is None or pair_summary.empty:
        return pd.DataFrame(columns=SCORE_KEYS + sep_cols)

    grouped = pair_summary.groupby(SCORE_KEYS)
    sep = pd.DataFrame({
        SEP_COL: grouped["S_sep_cond"].mean(),
        "S_sep_kept": grouped["S_sep_kept"].mean(),
        "n_pairs": grouped["FlavorPair"].nunique(),
        "n_jets_kept": grouped["n_jets_kept"].sum(),
        "n_jets_total": grouped["n_jets_total"].sum(),
    }).reset_index()
    sep["preserved_sep"] = preserved_sep(sep[SEP_COL], sep["S_sep_kept"])
    sep["share"] = 1.0 - sep["preserved_sep"]
    sep["coverage"] = safe_divide(sep["n_jets_kept"], sep["n_jets_total"])
    return sep[SCORE_KEYS + sep_cols].sort_values(SCORE_KEYS).reset_index(drop=True)


def slice_sets(node_comparisons, edge_comparisons, nodes_table, edges_table):
    """
    (heading, table name, comparisons, W1 table) of the node, edge and
    within-flavor slices. Within-flavor subsets are node comparisons.
    """
    return [
        ("NODES", "nodes", node_comparisons, nodes_table),
        ("EDGES", "edges", edge_comparisons, edges_table),
        ("WITHIN", "within", list(WITHIN_COMPARISONS), nodes_table),
    ]


def binned_pair_tables(data_dict, graph_types, first_graph, cond_var, bins, save_dir):
    """
    Bin the distributions and compute everything that needs the binned
    arrays: the across-flavor W1 tables, the per-pair scores per bin, and the
    matched denominators. The binned arrays are freed before returning.

    :param first_graph: a loaded construction; its jets give the flavor totals
    :return: tuple (across-flavor nodes, across-flavor edges, per-pair bins,
        per-pair matched summary)
    """
    pass_start = time.time()
    jet_totals = jets_per_flavor(data_dict[first_graph]["nodes"], FLAVORS)
    totals_text = ", ".join(f"{flavor}: {n_jets}" for flavor, n_jets in jet_totals.items())
    print(f"jets per flavor in the sample: {totals_text}", flush=True)

    data_binned, jets_per_bin = collect_binned_distributions(
        data_dict, graph_types, FLAVORS, cond_var, bins, min_jets=MIN_JETS_PER_BIN)
    save_csv(jets_per_bin, save_dir, f"jets_per_bin_{cond_var}.csv")
    print(f"binned in {(time.time() - pass_start) / 60:.1f} min; RSS {rss_gb():.1f} GB",flush=True)

    across_flavors_nodes, across_flavors_edges = compute_binned_w1_tables(data_binned, graph_types, FLAVORS, cond_var, bins)

    matched_start = time.time()
    pair_labels = FLAVORPAIR_LABELS + [WITHIN_PAIR_LABEL]
    pair_bins = pd.concat([pair_scores(across_flavors_nodes, graph_types, pair_labels),
                           pair_scores(across_flavors_edges, graph_types, pair_labels)
                           ], ignore_index=True)
    pair_summary = matched_pair_summary(data_binned, pair_bins, graph_types, jets_per_bin,
                                        jet_totals, cond_var, bins)
    print(f"matched denominators: {len(pair_summary)} (construction, pair, slice) rows in {(time.time() - matched_start) / 60:.1f} min", flush=True)
    del data_binned
    gc.collect()
    return across_flavors_nodes, across_flavors_edges, pair_bins, pair_summary


def write_pair_tables(pair_bins, pair_summary, cond_var, bin_labels, save_dir):
    """
    Sort and write the per-pair tables of one pass.

    :return: tuple (pair_bins, pair_summary), sorted. Later means over pairs
        are taken in this order, which fixes their floating-point rounding.
    """
    if not pair_bins.empty:
        pair_bins = order_bins(pair_bins, bin_labels)
        pair_bins = pair_bins.sort_values(PAIR_SORT + ["Bin"])
        save_csv(pair_bins, save_dir, f"scores_conditioned_{cond_var}_per_pair_bins.csv")
    if not pair_summary.empty:
        pair_summary = pair_summary.sort_values(PAIR_SORT)
        save_csv(pair_summary, save_dir, f"scores_conditioned_{cond_var}_per_pair_matched.csv")
        print(f"\nPER-PAIR (conditioned on {cond_var}; denominator on the same jets):")
        print(pair_summary.drop(columns=["bins"]).to_string(index=False))
    return pair_bins, pair_summary


def score_conditioned_slices(across_flavors_nodes, across_flavors_edges, graph_types, cond_var, bins, save_dir):
    """
    Per-bin S_sep of every slice, one CSV per slice; returns the list of slice tables
    """
    all_rows = []
    for heading, table_name, comparisons, flavor_table in slice_sets(
            COND_NODE_COMPARISONS, COND_EDGE_COMPARISONS, across_flavors_nodes,
            across_flavors_edges):
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = run_conditioned_slice(flavor_table, graph_types, comparison, curvature, aggregation, cond_var, bins)
            print(f"\n{heading} (conditioned on {cond_var}): {comparison} {curvature} {aggregation}")
            if df_slice.empty:
                print("[warn] empty conditioned slice")
                continue
            print(df_slice.to_string(index=False))
            file_name = (f"scores_conditioned_{cond_var}_{table_name}_{comparison}_{curvature}_{aggregation}.csv")
            save_csv(df_slice, save_dir, file_name.replace("/", "_"))
            all_rows.append(df_slice)
    return all_rows


def run_conditioned_scores(data_dict, graph_types, cond_var, bins, save_dir):
    """
    Full conditioned pass for one conditioning variable: bin, compute the binned W1 tables and matched denominators,
    score every slice in every bin, and write the weighted summary, the CSVs and the figures.
    Every file name contains cond_var.

    :param data_dict: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    :param cond_var: "none", "n", "n_sv", "n_pv", "sv_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges for cond_var
    :return: tuple (weighted, pair_summary); weighted is None if nothing was scored
    """
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = cond_bin_labels(cond_var, bins)
    pass_start = time.time()
    print(f"\nconditioned on {cond_var}, {len(bin_labels)} bins, {len(graph_types)} constructions", flush=True)

    loaded_types = [graph_type for graph_type in graph_types if graph_type in data_dict]
    if not loaded_types:
        print(f"[warn] none of {len(graph_types)} constructions is loaded. Skipping the {cond_var} pass", flush=True)
        return None, pd.DataFrame()

    across_nodes, across_edges, pair_bins, pair_summary = binned_pair_tables(
        data_dict, graph_types, loaded_types[0], cond_var, bins, save_dir)

    save_csv(across_nodes, save_dir, f"w1_binned_{cond_var}_across_flavors_nodes.csv")
    save_csv(across_edges, save_dir, f"w1_binned_{cond_var}_across_flavors_edges.csv")
    pair_bins, pair_summary = write_pair_tables(pair_bins, pair_summary, cond_var, bin_labels, save_dir)

    all_rows = score_conditioned_slices(across_nodes, across_edges, graph_types, cond_var, bins, save_dir)
    if not all_rows:
        print("[warn] no conditioned scores produced")
        return None, pair_summary

    all_cond_df = order_bins(pd.concat(all_rows, ignore_index=True), bin_labels)
    all_cond_df = all_cond_df.sort_values(["Comparison", "Curvature", "Aggregation", "Graph", "Bin"])
    save_csv(all_cond_df, save_dir, f"scores_conditioned_{cond_var}_all_slices.csv")

    weighted = weighted_summary(pair_summary)
    save_csv(weighted, save_dir, f"scores_conditioned_{cond_var}_weighted.csv")
    print(f"\nSUMMARY (jet-weighted over {cond_var} bins. S_sep with matched denominator and share):")
    print(weighted.to_string(index=False))

    pair_labels = FLAVORPAIR_LABELS + [WITHIN_PAIR_LABEL]
    plot_scores_vs_bin(all_cond_df, cond_var, bin_labels, save_dir, STYLING,
                       comparison_labels=COMPARISON_LABELS, legend_ncol=3)
    plot_pair_scores_vs_bin(pair_bins, cond_var, bin_labels, pair_labels, save_dir, STYLING,
                            pair_style, COMPARISON_LABELS)
    plot_pair_scores_vs_bin_by_graph(pair_bins, cond_var, bin_labels, pair_labels, save_dir,
                                     STYLING, pair_style, COMPARISON_LABELS)

    print(f"===== {cond_var} finished in {(time.time() - pass_start) / 60:.1f} min =====",
          flush=True)
    return weighted, pair_summary


def run_slice(flavor_table, comparison, curvature, aggregation):
    """
    Unconditioned S_sep of one slice over GRAPH_TYPES

    :return: DataFrame with one row per construction that has a score
    """
    sep = compute_separation_score(flavor_table, GRAPH_TYPES, comparison, curvature, aggregation, pair_labels_for(comparison))
    if sep[SEP_COL].dropna().empty:
        print(f"[warn] empty slice: {comparison} | {curvature} | {aggregation}")

    df = pd.DataFrame({SEP_COL: sep[SEP_COL], "n_terms_sep": sep["n_terms"]})
    df = df.dropna(subset=[SEP_COL])
    df = df.reset_index().rename(columns={"index": "Graph"})
    df["Comparison"] = comparison
    df["Curvature"] = curvature
    df["Aggregation"] = aggregation
    return df


def run_unconditioned_scores(across_flavors_nodes, across_flavors_edges, save_dir):
    """
    Unconditioned S_sep of every node and edge slice, one CSV per slice plus
    scores_all_slices.csv

    :return: all slices in one DataFrame
    """
    all_rows = []
    for heading, table_name, comparisons, flavor_table in slice_sets(
            NODE_COMPARISONS, EDGE_COMPARISONS, across_flavors_nodes, across_flavors_edges):
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = run_slice(flavor_table, comparison, curvature, aggregation)
            print(f"\n{heading}: {comparison} {curvature} {aggregation}")
            print(df_slice.sort_values(SEP_COL, ascending=False).to_string(index=False))
            file_name = f"scores_{table_name}_{comparison}_{curvature}_{aggregation}.csv"
            save_csv(df_slice, save_dir, file_name.replace("/", "_"))
            all_rows.append(df_slice)

    all_scores_df = pd.concat(all_rows, ignore_index=True)
    save_csv(all_scores_df, save_dir, "scores_all_slices.csv")
    return all_scores_df


def unconditioned_pair_scores(across_flavors_nodes, across_flavors_edges, save_dir):
    """
    S_sep(g, pair) of every pair on the full sample, written to scores_per_pair_uncond.csv.
    """
    pair_uncond = pd.concat([
        pair_scores(across_flavors_nodes, GRAPH_TYPES, ALL_PAIR_LABELS),
        pair_scores(across_flavors_edges, GRAPH_TYPES, ALL_PAIR_LABELS),
    ], ignore_index=True)
    if pair_uncond.empty:
        print("[warn] per-pair unconditioned table is empty; check that FLAVORPAIR_LABELS "
              "matches the labels in PAIR_LABEL", flush=True)
        return pair_uncond
    pair_uncond = pair_uncond.sort_values(PAIR_SORT)
    save_csv(pair_uncond, save_dir, "scores_per_pair_uncond.csv")
    print(f"\nPER-PAIR (unconditioned): {len(pair_uncond)} rows")
    print(pair_uncond.head(20).to_string(index=False))
    return pair_uncond
