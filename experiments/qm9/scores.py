import os
import time
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.qm9.config import AGGREGATIONS, CURVATURES, EDGE_SCORED,EXCLUDE_FC_FROM_SENS_FOR, FC_GRAPH, MIN_MOLS_PER_BIN, MIN_VALUES_PER_DIST, NODE_SCORED, SENS_ONLY_COMPARISONS, SEP_POOL, SKIP_SENSITIVITY, SUBSET_PAIRS, comparison_subsets, dist_key
from experiments.qm9.constructions import DUPLICATE_GRAPHS, STYLING
from experiments.qm9.distributions import collect_binned_distributions,compute_binned_w1_tables, cond_bin_labels
from plotting.conditioned import plot_scores_vs_bin
from utils.binning import order_bins
from utils.tables import SCORE_KEYS,SENS_COL,SEP_COL, SLICE_KEYS,apply_value_guard, save_csv, upper_names


def scores_by_graph(rows, score_col, graph_names):
    """
    Mean W_Distance per construction, reindexed to graph_names.

    :param rows: W1 rows with Graph and W_Distance
    :return: DataFrame indexed by graph with columns score_col and n_terms
    """
    grouped = rows.groupby("Graph")["W_Distance"]
    result = pd.DataFrame({score_col: grouped.mean(), "n_terms": grouped.size()})
    result = result.reindex(graph_names)
    result["n_terms"] = result["n_terms"].fillna(0).astype(int)
    return result


def compute_separation_score(df_subset_pairs, graph_types, comparison, curvature, aggregation, cond_bin=None):
    """
    S_sep of every construction: the W1 between the curvature distributions
    of the two subsets of a comparison, within the same construction. Higher
    means better separation. n_terms is 0 or 1 unless the comparison pools
    several pairs (SEP_POOL), where S_sep is their mean.

    :param df_subset_pairs: across-subset W1 table
    :param comparison: a key of SUBSET_PAIRS or SEP_POOL
    :param cond_bin: bin label to restrict the rows to, or None
    :return: DataFrame indexed by upper-cased graph with columns SEP_COL and n_terms
    """
    graph_names = upper_names(graph_types)
    if comparison in SEP_POOL:
        in_comparison = df_subset_pairs["Comparison"].isin(SEP_POOL[comparison])
    else:
        in_comparison = df_subset_pairs["Comparison"] == comparison
    rows = df_subset_pairs[in_comparison & (df_subset_pairs["Curvature"] == curvature) & (df_subset_pairs["Aggregation"] == aggregation)]
    if cond_bin is not None:
        rows = rows[rows["Bin"] == cond_bin]
    rows = apply_value_guard(rows, MIN_VALUES_PER_DIST)
    rows = rows[~rows["Graph"].isin(upper_names(DUPLICATE_GRAPHS))]
    rows = rows.dropna(subset=["W_Distance"])

    if rows.empty:
        return pd.DataFrame({SEP_COL: np.nan, "n_terms": 0}, index=graph_names)
    return scores_by_graph(rows, SEP_COL, graph_names)


def compute_sensitivity_score(df_across_samesubset, graph_types, comparison, curvature,  aggregation, cond_bin=None):
    """
    S_sens of every construction (not used in the paper, and skipped while
    SKIP_SENSITIVITY is set): how much the curvature distribution of a subset
    changes against the other constructions, averaged over those
    constructions and the comparison's subsets. Lower means the construction
    is more interchangeable with the rest.

    For curvatures in EXCLUDE_FC_FROM_SENS_FOR, the fully connected graph is
    left out of everyone else's average but still gets its own score.

    :param df_across_samesubset: across-construction W1 table
    :param comparison: a key of SUBSET_PAIRS or SENS_ONLY_COMPARISONS
    :param cond_bin: bin label to restrict the rows to, or None
    :return: DataFrame indexed by upper-cased graph with columns SENS_COL and n_terms
    """
    graph_names = upper_names(graph_types)
    if SKIP_SENSITIVITY or df_across_samesubset is None or df_across_samesubset.empty:
        return pd.DataFrame({SENS_COL: np.nan, "n_terms": 0}, index=graph_names)

    table = df_across_samesubset
    rows = table[(table["Curvature"] == curvature) & (table["Aggregation"] == aggregation) & table["Subset"].isin(comparison_subsets(comparison))]
    if cond_bin is not None:
        rows = rows[rows["Bin"] == cond_bin]
    rows = apply_value_guard(rows, MIN_VALUES_PER_DIST)
    duplicates = upper_names(DUPLICATE_GRAPHS)
    rows = rows[~rows["Graph_A"].isin(duplicates) & ~rows["Graph_B"].isin(duplicates)]
    rows = rows.dropna(subset=["W_Distance"])

    if rows.empty:
        return pd.DataFrame({SENS_COL: np.nan, "n_terms": 0}, index=graph_names)

    fc_name = FC_GRAPH.upper()
    if curvature in EXCLUDE_FC_FROM_SENS_FOR:
        involves_fc = (rows["Graph_A"] == fc_name) | (rows["Graph_B"] == fc_name)
        pair_rows = rows[~involves_fc]
        fc_rows = rows[involves_fc]
    else:
        pair_rows = rows
        fc_rows = rows.iloc[0:0]

    # Every pair row counts once for each of its two constructions.
    side_a = pair_rows[["Graph_A", "Subset", "W_Distance"]].rename(columns={"Graph_A": "Graph"})
    side_b = pair_rows[["Graph_B", "Subset", "W_Distance"]].rename(columns={"Graph_B": "Graph"})
    long_table = pd.concat([side_a, side_b], ignore_index=True)
    if not fc_rows.empty:
        fc_table = fc_rows[["Subset", "W_Distance"]].copy()
        fc_table["Graph"] = fc_name
        long_table = pd.concat([long_table, fc_table[["Graph", "Subset", "W_Distance"]]], ignore_index=True)
    return scores_by_graph(long_table, SENS_COL, graph_names)


def molecules_per_bin(mols_per_bin):
    """{bin label: molecules in the bin}. Counts are the same for every construction."""
    first_rows = mols_per_bin[mols_per_bin["Graph"] == mols_per_bin["Graph"].iloc[0]]
    return {str(label): int(n_mols) for label, n_mols in zip(first_rows["Bin"], first_rows["n_mols"])}


def coverage_of(n_kept, n_total_mols):
    return n_kept / n_total_mols if n_total_mols else np.nan


def single_pair_row(group, graph_upper, comparison, curvature, aggregation, cells, bin_labels, mols_in_bin, n_total_mols, cond_var):
    """
    Matched row of one (graph, comparison, slice); None if no bin qualifies.

    :param group: binned across-subset W1 rows of this (graph, comparison, slice)
    :param cells: data_binned[graph_type]
    """
    subset_a, subset_b = SUBSET_PAIRS[comparison]
    key_a = dist_key(subset_a, curvature, aggregation)
    key_b = dist_key(subset_b, curvature, aggregation)
    w1_by_bin = group.set_index(group["Bin"].astype(str))["W_Distance"]

    used_bins = [label for label in bin_labels if label in w1_by_bin.index and mols_in_bin.get(label, 0) > 0]
    if not used_bins:
        return None
    weights = np.array([mols_in_bin[label] for label in used_bins], dtype=float)
    w1_values = np.array([float(w1_by_bin[label]) for label in used_bins], dtype=float)
    s_cond = float((w1_values * weights).sum() / weights.sum())

    values_a = np.concatenate([cells[label][key_a] for label in used_bins])
    values_b = np.concatenate([cells[label][key_b] for label in used_bins])
    s_kept = float(wasserstein_distance(values_a, values_b))
    preserved = float(preserved_sep(s_cond, s_kept))
    n_kept = int(weights.sum())
    return {
        "Graph": graph_upper,
        "Comparison": comparison,
        "Curvature": curvature,
        "Aggregation": aggregation,
        "CondVar": cond_var,
        "S_sep_cond": s_cond,
        "S_sep_kept": s_kept,
        "share": 1.0 - preserved,
        "preserved_sep": preserved,
        "n_bins": len(used_bins),
        "bins": ";".join(used_bins),
        "n_A": int(values_a.size),
        "n_B": int(values_b.size),
        "n_mols_kept": n_kept,
        "n_mols_total": int(n_total_mols),
        "coverage": coverage_of(n_kept, n_total_mols),
    }


def pooled_rows(per_pair, bin_labels, mols_in_bin, n_total_mols, cond_var):
    """
    Rows of the pooled comparisons (all_nodes, all_edges), built like the
    jets pool flavor pairs: the mean over member pairs of S_sep_cond and of
    S_sep_kept, and the share and preserved_sep of those two means. Coverage counts the
    molecules in the union of the bins any member pair used.
    """
    rows = []
    for pooled_name, members in SEP_POOL.items():
        member_rows = per_pair[per_pair["Comparison"].isin(members)]
        for (graph_upper, curvature, aggregation), group in member_rows.groupby(["Graph", "Curvature", "Aggregation"], sort=False):
            s_cond = float(group["S_sep_cond"].mean())
            s_kept = float(group["S_sep_kept"].mean())
            preserved = float(preserved_sep(s_cond, s_kept))
            used = set()
            for bins_text in group["bins"]:
                used.update(bins_text.split(";"))
            used_bins = [label for label in bin_labels if label in used]
            n_kept = sum(mols_in_bin[label] for label in used_bins)
            rows.append({
                "Graph": graph_upper,
                "Comparison": pooled_name,
                "Curvature": curvature,
                "Aggregation": aggregation,
                "CondVar": cond_var,
                "S_sep_cond": s_cond,
                "S_sep_kept": s_kept,
                "share": 1.0 - preserved,
                "preserved_sep": preserved,
                "n_bins": len(used_bins),
                "bins": ";".join(used_bins),
                "n_A": int(group["n_A"].sum()),
                "n_B": int(group["n_B"].sum()),
                "n_mols_kept": n_kept,
                "n_mols_total": int(n_total_mols),
                "coverage": coverage_of(n_kept, n_total_mols),
                "n_pairs": len(group),
            })
    return rows


def matched_denominators(data_binned, subsets_nodes, subsets_edges, graph_types, mols_per_bin, n_total_mols, cond_var, bins):
    """
    For every (graph, comparison, slice), the conditioned separation next to
    the unconditioned separation on the same molecules:

        S_sep_cond = sum_b w_b W1(A_b, B_b) / sum_b w_b
        S_sep_kept = W1(concat_b A_b, concat_b B_b)
        preserved_sep = S_sep_cond / S_sep_kept
        share         = 1 - preserved_sep

    b runs over the bins where the comparison has a W1 row, A_b and B_b are
    the two subset distributions in bin b, and w_b is the number of
    molecules in bin b. The bins partition the molecules, so the kept
    distributions are the concatenated per-bin arrays.

    :param data_binned: first output of collect_binned_distributions
    :param subsets_nodes: binned across-subset node table
    :param subsets_edges: binned across-subset edge table
    :param mols_per_bin: second output of collect_binned_distributions
    :param n_total_mols: molecules in the sample, for the coverage
    :return: DataFrame with one row per SCORE_KEYS, pooled comparisons included
    """
    tables = [table for table in (subsets_nodes, subsets_edges)
              if table is not None and not table.empty]
    if not tables:
        return pd.DataFrame()

    bin_labels = cond_bin_labels(cond_var, bins)
    mols_in_bin = molecules_per_bin(mols_per_bin)
    graph_by_upper = {graph_type.upper(): graph_type for graph_type in graph_types}

    pairs = apply_value_guard(pd.concat(tables, ignore_index=True), MIN_VALUES_PER_DIST)
    pairs = pairs[~pairs["Graph"].isin(upper_names(DUPLICATE_GRAPHS))]
    pairs = pairs.dropna(subset=["W_Distance"])

    rows = []
    for (graph_upper, comparison, curvature, aggregation), group in pairs.groupby(SCORE_KEYS, sort=False):
        graph_type = graph_by_upper.get(graph_upper)
        if graph_type is None or comparison not in SUBSET_PAIRS:
            continue
        row = single_pair_row(group, graph_upper, comparison, curvature, aggregation, data_binned[graph_type], bin_labels, mols_in_bin, n_total_mols, cond_var)
        if row is not None:
            rows.append(row)
    per_pair = pd.DataFrame(rows)
    if per_pair.empty:
        return per_pair
    per_pair["n_pairs"] = 1

    pooled = pooled_rows(per_pair, bin_labels, mols_in_bin, n_total_mols, cond_var)
    if not pooled:
        return per_pair
    return pd.concat([per_pair, pd.DataFrame(pooled)], ignore_index=True)


def score_conditioned_slice(df_subset_pairs_binned, df_across_samesubset_binned, graph_types, comparison, curvature, aggregation, cond_var, bins):
    """
    Score one slice separately in every bin. A row is kept when at least one
    of the two scores exists, since radius_bond, for example, loses S_sep on
    nonbond edges in most bins while S_sens survives.

    :param df_subset_pairs_binned: binned across-subset W1 table
    :param df_across_samesubset_binned: binned across-construction table
    :return: DataFrame with one row per (graph, bin)
    """
    rows = []
    for label in cond_bin_labels(cond_var, bins):
        sep = compute_separation_score(df_subset_pairs_binned, graph_types, comparison, curvature, aggregation, cond_bin=label)
        sens = compute_sensitivity_score(df_across_samesubset_binned, graph_types, comparison, curvature, aggregation, cond_bin=label)
        for graph in sep.index:
            s_sep = sep.loc[graph, SEP_COL]
            s_sens = np.nan
            n_terms_sens = 0
            if graph in sens.index:
                s_sens = sens.loc[graph, SENS_COL]
                n_terms_sens = int(sens.loc[graph, "n_terms"])
            if np.isnan(s_sep) and np.isnan(s_sens):
                continue
            rows.append({
                "Graph": graph,
                "CondVar": cond_var,
                "Bin": label,
                SEP_COL: s_sep,
                SENS_COL: s_sens,
                "n_terms_sep": int(sep.loc[graph, "n_terms"]),
                "n_terms_sens": n_terms_sens,
                "Comparison": comparison,
                "Curvature": curvature,
                "Aggregation": aggregation,
            })
    return pd.DataFrame(rows)


def weighted_mean(rows, col):
    """Mean of col weighted by the "w" column, over rows where col exists and w > 0; NaN if none."""
    valid = rows[col].notna() & (rows["w"] > 0)
    if not valid.any():
        return np.nan
    weights = rows.loc[valid, "w"]
    return float((rows.loc[valid, col] * weights).sum() / weights.sum())


def sensitivity_summary(all_cond_df, mols_per_bin):
    """
    Molecule-weighted mean over bins of the per-bin S_sens, plus S_sep_check:
    the same weighting applied to the per-bin S_sep.

    :return: one row per SCORE_KEYS
    """
    first_rows = mols_per_bin[mols_per_bin["Graph"] == mols_per_bin["Graph"].iloc[0]]
    used_rows = first_rows[first_rows["used"].astype(bool)]
    bin_weights = used_rows.groupby("Bin")["n_mols"].sum()

    scores = all_cond_df.copy()
    scores["w"] = scores["Bin"].astype(str).map(bin_weights).fillna(0.0).astype(float)

    rows = []
    for group_key, group in scores.groupby(SCORE_KEYS):
        mean_terms_sens = np.nan
        if "n_terms_sens" in group.columns:
            mean_terms_sens = float(group["n_terms_sens"].mean())
        valid_sens = group[SENS_COL].notna() & (group["w"] > 0)
        rows.append({
            **dict(zip(SCORE_KEYS, group_key)),
            "S_sep_check": weighted_mean(group, SEP_COL),
            SENS_COL: weighted_mean(group, SENS_COL),
            "n_bins_sens": int(valid_sens.sum()),
            "mean_terms_sens": mean_terms_sens,
        })
    return pd.DataFrame(rows, columns=SCORE_KEYS + ["S_sep_check", SENS_COL, "n_bins_sens","mean_terms_sens"])


def report_sep_check(summary):
    """
    Print the gap between the matched S_sep and S_sep_check; they should agree
    exactly. Pooled comparisons are left out: S_sep_check pools per bin
    while the matched S_sep averages pairs, which differs when the pairs
    cover different bins.
    """
    pooled = summary["Comparison"].isin(list(SEP_POOL))
    both = summary[SEP_COL].notna() & summary["S_sep_check"].notna() & ~pooled
    if both.any():
        differences = (summary.loc[both, SEP_COL] - summary.loc[both, "S_sep_check"]).abs()
        print(f"[check] weighted S_sep from the per-bin table vs matched_denominators: max |diff| = {float(differences.max()):.3e} over {int(both.sum())} rows",
              flush=True)


def weighted_summary(all_cond_df, mols_per_bin, matched):
    """
    Conditioned scores per SCORE_KEYS: S_sep, S_sep_kept, the share,
    preserved_sep and the coverage from matched_denominators, and the molecule-weighted S_sens.

    :param all_cond_df: concatenated per-bin scores
    :param mols_per_bin: second output of collect_binned_distributions
    :param matched: output of matched_denominators
    :return: one row per SCORE_KEYS
    """
    sep_cols = [SEP_COL, "S_sep_kept", "share", "preserved_sep", "coverage", "n_bins_sep","n_mols_kept","n_mols_total"]
    if matched is not None and not matched.empty:
        matched_cols = ["S_sep_cond", "S_sep_kept", "share", "preserved_sep", "coverage", "n_bins", "n_mols_kept", "n_mols_total"]
        sep = matched[SCORE_KEYS + matched_cols].rename( columns={"S_sep_cond": SEP_COL, "n_bins": "n_bins_sep"})
    else:
        sep = pd.DataFrame(columns=SCORE_KEYS + sep_cols)

    summary = sep.merge(sensitivity_summary(all_cond_df, mols_per_bin), on=SCORE_KEYS, how="outer")
    report_sep_check(summary)
    summary = summary[SCORE_KEYS + sep_cols + [SENS_COL, "n_bins_sens", "mean_terms_sens"]]
    return summary.sort_values(SCORE_KEYS).reset_index(drop=True)


def score_slice(df_subset_pairs, df_across_samesubset, graph_types, comparison, curvature,aggregation):
    """
    Unconditioned S_sep and S_sens of one slice. Constructions missing a
    required score are dropped, since they are not a point on the
    separation-sensitivity plane.

    :param df_subset_pairs: across-subset W1 table
    :param df_across_samesubset: across-construction W1 table
    :return: DataFrame with one row per graph that has the required scores
    """
    sep = compute_separation_score(df_subset_pairs, graph_types, comparison, curvature, aggregation)
    sens = compute_sensitivity_score(df_across_samesubset, graph_types, comparison, curvature, aggregation)
    if SKIP_SENSITIVITY:
        required = [SEP_COL]
    elif comparison in SENS_ONLY_COMPARISONS:
        required = [SENS_COL]
    else:
        required = [SEP_COL, SENS_COL]

    scores = pd.DataFrame({
        SEP_COL: sep[SEP_COL],
        SENS_COL: sens[SENS_COL],
        "n_terms_sep": sep["n_terms"],
        "n_terms_sens": sens["n_terms"],
    }).dropna(subset=required)
    if scores.empty:
        print(f"[warn] empty slice: {comparison} | {curvature} | {aggregation} (no construction has {required})")
    scores = scores.reset_index().rename(columns={"index": "Graph"})
    scores["Comparison"] = comparison
    scores["Curvature"] = curvature
    scores["Aggregation"] = aggregation
    return scores


def slice_inputs(across_subsets_nodes, across_subsets_edges, across_graphs_nodes,
                 across_graphs_edges):
    """
    (table kind, scored comparisons, across-subset table, across-graph table) per kind
    """
    return [
        ("nodes", NODE_SCORED, across_subsets_nodes, across_graphs_nodes),
        ("edges", EDGE_SCORED, across_subsets_edges, across_graphs_edges),
    ]


def run_unconditioned_scores(across_subsets_nodes, across_subsets_edges, across_graphs_nodes, across_graphs_edges, graph_types, save_dir):
    """
    Unconditioned scores of every slice, one CSV per slice plus
    scores_all_slices.csv.

    :return: all slices in one DataFrame
    """
    all_rows = []
    for table_kind, comparisons, df_subsets, df_graphs in slice_inputs(
            across_subsets_nodes, across_subsets_edges, across_graphs_nodes, across_graphs_edges):
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = score_slice(df_subsets, df_graphs, graph_types, comparison, curvature, aggregation)
            print(f"\n{table_kind.upper()}: {comparison} {curvature} {aggregation}")
            print(df_slice.sort_values(SEP_COL, ascending=False).head(10).to_string(index=False))
            file_name = f"scores_{table_kind}_{comparison}_{curvature}_{aggregation}.csv"
            save_csv(df_slice, save_dir, file_name.replace("/", "_"))
            all_rows.append(df_slice)

    all_scores_df = pd.concat(all_rows, ignore_index=True)
    save_csv(all_scores_df, save_dir, "scores_all_slices.csv")
    return all_scores_df


def score_conditioned_slices(across_subsets_nodes, across_subsets_edges, across_graphs_nodes, across_graphs_edges, graph_types, cond_var, bins, save_dir):
    """
    Per-bin scores of every slice, one CSV per slice; returns the list of slice tables
    """
    all_rows = []
    for table_kind, comparisons, df_subsets, df_graphs in slice_inputs(across_subsets_nodes, across_subsets_edges, across_graphs_nodes, across_graphs_edges):
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = score_conditioned_slice(df_subsets, df_graphs, graph_types, comparison, curvature, aggregation, cond_var, bins)
            print(f"\n{table_kind.upper()} (conditioned on {cond_var}): {comparison} {curvature} {aggregation}")
            if df_slice.empty:
                print("[warn] empty conditioned slice")
                continue
            print(df_slice.to_string(index=False))
            file_name = f"scores_conditioned_{cond_var}_{table_kind}_{comparison}_{curvature}_{aggregation}.csv"
            save_csv(df_slice, save_dir, file_name.replace("/", "_"))
            all_rows.append(df_slice)
    return all_rows


def run_conditioned_scores(data_dict, graph_types, cond_var, bins, save_dir):
    """
    Conditioned pass for one conditioning variable: bin the distributions,
    compute the W1 tables inside each bin and the matched denominators,
    score every slice per bin, and write the molecule-weighted summary, the
    CSVs and the figures.

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :param cond_var: "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges for cond_var
    :return: output of weighted_summary, or None if no slice was scored
    """
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = cond_bin_labels(cond_var, bins)
    pass_start = time.time()
    print(f"\n conditioned on {cond_var}, {len(bin_labels)} bins: {bin_labels} ")

    # Coverage counts every molecule, including those outside all bins.
    n_total_mols = int(data_dict[graph_types[0]]["nodes"]["mol_id"].nunique())

    data_binned, mols_per_bin = collect_binned_distributions(data_dict, graph_types, cond_var, bins)
    save_csv(mols_per_bin, save_dir, f"mols_per_bin_{cond_var}.csv")

    (across_graphs_nodes, across_graphs_edges,
     across_subsets_nodes, across_subsets_edges) = compute_binned_w1_tables(data_binned, graph_types, cond_var, bins, across_graphs=not SKIP_SENSITIVITY)

    # Matched denominators read the binned arrays, so they run before those are freed.
    matched_start = time.time()
    matched = matched_denominators(data_binned, across_subsets_nodes, across_subsets_edges, graph_types, mols_per_bin, n_total_mols, cond_var, bins)
    print(f"  matched denominators: {len(matched)} (construction, slice) rows in {(time.time() - matched_start) / 60:.1f} min", flush=True)
    del data_binned

    prefix = f"w1_binned_{cond_var}"
    save_csv(across_graphs_nodes, save_dir, f"{prefix}_across_graphs_nodes.csv")
    save_csv(across_graphs_edges, save_dir, f"{prefix}_across_graphs_edges.csv")
    save_csv(across_subsets_nodes, save_dir, f"{prefix}_across_subsets_nodes.csv")
    save_csv(across_subsets_edges, save_dir, f"{prefix}_across_subsets_edges.csv")

    if not matched.empty:
        matched = matched.sort_values(SLICE_KEYS + ["Graph"])
        save_csv(matched, save_dir, f"scores_conditioned_{cond_var}_matched.csv")
        print(f"\nMATCHED (conditioned on {cond_var}; denominator on the same molecules):")
        print(matched.drop(columns=["bins"]).to_string(index=False))

    all_rows = score_conditioned_slices(across_subsets_nodes, across_subsets_edges, across_graphs_nodes, across_graphs_edges, graph_types, cond_var, bins, save_dir)
    if not all_rows:
        print("[warn] no conditioned scores produced")
        return None

    all_cond_df = order_bins(pd.concat(all_rows, ignore_index=True), bin_labels)
    all_cond_df = all_cond_df.sort_values(SLICE_KEYS + ["Graph", "Bin"])
    save_csv(all_cond_df, save_dir, f"scores_conditioned_{cond_var}_all_slices.csv")

    weighted = weighted_summary(all_cond_df, mols_per_bin, matched)
    save_csv(weighted, save_dir, f"scores_conditioned_{cond_var}_weighted.csv")
    print(f"\nSUMMARY (molecule-weighted over {cond_var} bins; S_sep with matched denominator and share):")
    print(weighted.to_string(index=False))

    score_cols = (SEP_COL,) if SKIP_SENSITIVITY else (SEP_COL, SENS_COL)
    plot_scores_vs_bin(all_cond_df, cond_var, bin_labels, save_dir, STYLING, score_cols=score_cols)

    print(f"{cond_var} finished in {(time.time() - pass_start) / 60:.1f} min ", flush=True)
    return weighted
