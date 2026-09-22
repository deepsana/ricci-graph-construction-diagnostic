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


def compute_separation_score(df_subset_pairs, graph_types, comparison, curvature, aggregation,  reduce="mean", cond_bin=None):
    """
    Separation score (S_sep) for each construction:
    the W1 distance between curvature distributions of two compared subsets within the same construction.

    Higher values indicate better population separation. 
    Each comparison contains a single subset pair (n_terms is 0 or 1) unless multiple pairs are pooled.

    :param df_subset_pairs: across-subsets W1 table
    :param graph_types: constructions to report
    :param comparison: pair label, a key of SUBSET_PAIRS or SEP_POOL
    :param curvature: "orc" or "frc"
    :param aggregation: "union" or "per_mol_mean"
    :param reduce: "mean" or "max"
    :param cond_bin: bin label to restrict the rows to, or None
    :return: DataFrame indexed by upper-cased graph with columns SEP_COL and n_terms
    """
    graph_names = upper_names(graph_types)

    if comparison in SEP_POOL:
        comp_mask = df_subset_pairs["Comparison"].isin(SEP_POOL[comparison])
    else:
        comp_mask = df_subset_pairs["Comparison"] == comparison
    mask = (comp_mask & (df_subset_pairs["Curvature"] == curvature) & (df_subset_pairs["Aggregation"] == aggregation))
    sub = df_subset_pairs[mask]
    if cond_bin is not None:
        sub = sub[sub["Bin"] == cond_bin]
    sub = apply_value_guard(sub, MIN_VALUES_PER_DIST)
    sub = sub[~sub["Graph"].isin(upper_names(DUPLICATE_GRAPHS))]
    sub = sub.dropna(subset=["W_Distance"])

    if sub.empty:
        return pd.DataFrame({SEP_COL: np.nan, "n_terms": 0}, index=graph_names)

    grouped = sub.groupby("Graph")["W_Distance"]
    if reduce == "mean":
        scores = grouped.mean()
    elif reduce == "max":
        scores = grouped.max()
    else:
        raise ValueError("reduce must be 'mean' or 'max'")

    result = pd.DataFrame({SEP_COL: scores, "n_terms": grouped.size()})
    result = result.reindex(graph_names)
    result["n_terms"] = result["n_terms"].fillna(0).astype(int)
    return result


def compute_sensitivity_score(df_across_samesubset, graph_types, comparison, curvature, aggregation, subsets=None, reduce="mean", cond_bin=None):
    """
    THIS IS OLD CODE --- not used for the paper
    Construction sensitivity S_sens of every construction: how much the
    curvature distribution of one subset changes against the other
    constructions, averaged over those constructions and over the subsets.
    Lower means the construction is more interchangeable with the rest.

    For curvatures in EXCLUDE_FC_FROM_SENS_FOR, FC is left out of everyone
    else's average but still gets its own score against all constructions.

    :param df_across_samesubset: across-construction W1 table
    :param graph_types: constructions to report
    :param comparison: pair label, a key of SUBSET_PAIRS
    :param curvature: "orc" or "frc"
    :param aggregation: "union" or "per_mol_mean"
    :param subsets: subsets to average over; defaults to the two subsets of comparison
    :param reduce: "mean" or "max"
    :param cond_bin: bin label to restrict the rows to, or None
    :return: DataFrame indexed by upper-cased graph with columns SENS_COL and n_terms
    """
    if subsets is None:
        subsets = comparison_subsets(comparison)
    graph_names = upper_names(graph_types)
    if SKIP_SENSITIVITY or df_across_samesubset is None or df_across_samesubset.empty:
        return pd.DataFrame({SENS_COL: np.nan, "n_terms": 0}, index=graph_names)

    mask = ((df_across_samesubset["Curvature"] == curvature) & (df_across_samesubset["Aggregation"] == aggregation) & df_across_samesubset["Subset"].isin(subsets))
    sub = df_across_samesubset[mask]
    if cond_bin is not None:
        sub = sub[sub["Bin"] == cond_bin]
    sub = apply_value_guard(sub, MIN_VALUES_PER_DIST)
    duplicates = upper_names(DUPLICATE_GRAPHS)
    sub = sub[~sub["Graph_A"].isin(duplicates) & ~sub["Graph_B"].isin(duplicates)]
    sub = sub.dropna(subset=["W_Distance"])

    if sub.empty:
        return pd.DataFrame({SENS_COL: np.nan, "n_terms": 0}, index=graph_names)

    fc_name = FC_GRAPH.upper()
    if curvature in EXCLUDE_FC_FROM_SENS_FOR:
        involves_fc = (sub["Graph_A"] == fc_name) | (sub["Graph_B"] == fc_name)
        pair_rows = sub[~involves_fc]
        fc_rows = sub[involves_fc]
    else:
        pair_rows = sub
        fc_rows = sub.iloc[0:0]

    side_a = pair_rows[["Graph_A", "Subset", "W_Distance"]].rename(columns={"Graph_A": "Graph"})
    side_b = pair_rows[["Graph_B", "Subset", "W_Distance"]].rename(columns={"Graph_B": "Graph"})
    long_table = pd.concat([side_a, side_b], ignore_index=True)

    if not fc_rows.empty:
        fc_table = fc_rows[["Subset", "W_Distance"]].copy()
        fc_table["Graph"] = fc_name
        fc_table = fc_table[["Graph", "Subset", "W_Distance"]]
        long_table = pd.concat([long_table, fc_table], ignore_index=True)

    grouped = long_table.groupby("Graph")["W_Distance"]
    if reduce == "mean":
        scores = grouped.mean()
    elif reduce == "max":
        scores = grouped.max()
    else:
        raise ValueError("reduce must be 'mean' or 'max'")

    result = pd.DataFrame({SENS_COL: scores, "n_terms": grouped.size()})
    result = result.reindex(graph_names)
    result["n_terms"] = result["n_terms"].fillna(0).astype(int)
    return result


def matched_denominators(data_binned, subsets_nodes, subsets_edges, graph_types, mols_per_bin, n_total_mols, cond_var, bins):
    """
    Compare, for every (graph, comparison, slice), the conditioned
    separation with the unconditioned separation on the same molecules:

        S_sep_cond = sum_b w_b * W1(A_b, B_b) / sum_b w_b
        S_sep_kept = W1(concat_b A_b, concat_b B_b)
        share = 1 - S_sep_cond / S_sep_kept

    Here
    - b runs over the bins in which the comparison has a W1 ro
    - A_b and B_b are the two subset distributions in bin b and w_b is the number of
    molecules in bin b.
    Since the bins partition the molecules, the kept distributions are the concatenated per-bin arrays.

    :param data_binned: first output of collect_binned_distributions
    :param subsets_nodes: binned across-subset node table
    :param subsets_edges: binned across-subset edge table
    :param graph_types: constructions (keys of data_binned)
    :param mols_per_bin: second output of collect_binned_distributions
    :param n_total_mols: molecules in the sample, used for the coverage
    :param cond_var: conditioning variable name
    :param bins: bins for cond_var
    :return: DataFrame with one row per (Graph, Comparison, Curvature, Aggregation)
    """
    tables = []
    for table in (subsets_nodes, subsets_edges):
        if table is not None and not table.empty:
            tables.append(table)
    if not tables:
        return pd.DataFrame()

    bin_labels = cond_bin_labels(cond_var, bins)
    first_graph = mols_per_bin["Graph"].iloc[0]
    first_rows = mols_per_bin[mols_per_bin["Graph"] == first_graph]
    mols_in_bin = {}
    for label, n_mols in zip(first_rows["Bin"], first_rows["n_mols"]):
        mols_in_bin[str(label)] = int(n_mols)
    graph_by_upper = {}
    for graph_type in graph_types:
        graph_by_upper[graph_type.upper()] = graph_type

    pairs = pd.concat(tables, ignore_index=True)
    pairs = apply_value_guard(pairs, MIN_VALUES_PER_DIST)
    pairs = pairs[~pairs["Graph"].isin(upper_names(DUPLICATE_GRAPHS))]
    pairs = pairs.dropna(subset=["W_Distance"])

    rows = []
    for group_key, group in pairs.groupby(SCORE_KEYS, sort=False):
        graph_upper, comparison, curvature, aggregation = group_key
        graph_type = graph_by_upper.get(graph_upper)
        if graph_type is None or comparison not in SUBSET_PAIRS:
            continue
        subset_a, subset_b = SUBSET_PAIRS[comparison]
        key_a = dist_key(subset_a, curvature, aggregation)
        key_b = dist_key(subset_b, curvature, aggregation)
        w1_by_bin = group.set_index(group["Bin"].astype(str))["W_Distance"]

        used_bins = []
        bin_weights = []
        bin_w1 = []
        for label in bin_labels:
            if label in w1_by_bin.index and mols_in_bin.get(label, 0) > 0:
                used_bins.append(label)
                bin_weights.append(mols_in_bin[label])
                bin_w1.append(float(w1_by_bin[label]))
        if not used_bins:
            continue

        weights = np.array(bin_weights, dtype=float)
        w1_values = np.array(bin_w1, dtype=float)
        s_cond = float((w1_values * weights).sum() / weights.sum())

        arrays_a = []
        arrays_b = []
        for label in used_bins:
            arrays_a.append(data_binned[graph_type][label][key_a])
            arrays_b.append(data_binned[graph_type][label][key_b])
        values_a = np.concatenate(arrays_a)
        values_b = np.concatenate(arrays_b)
        s_kept = float(wasserstein_distance(values_a, values_b))

        share = np.nan
        if s_kept > 0:
            share = 1.0 - s_cond / s_kept
        n_kept = int(weights.sum())
        coverage = np.nan
        if n_total_mols:
            coverage = n_kept / n_total_mols

        rows.append({
            "Graph": graph_upper,
            "Comparison": comparison,
            "Curvature": curvature,
            "Aggregation": aggregation,
            "CondVar": cond_var,
            "S_sep_cond": s_cond,
            "S_sep_kept": s_kept,
            "share": share,
            "n_bins": len(used_bins),
            "bins": ";".join(used_bins),
            "n_A": int(values_a.size),
            "n_B": int(values_b.size),
            "n_mols_kept": n_kept,
            "n_mols_total": int(n_total_mols),
            "coverage": coverage,
        })
    per_pair = pd.DataFrame(rows)
    if per_pair.empty:
        return per_pair
    per_pair["n_pairs"] = 1

    # all_nodes / all_edges pool their pairs as the jets pool flavor pairs:
    # the mean over pairs of S_sep_cond, the mean over pairs of S_sep_kept,
    # and the share of those two means. Coverage counts the molecules in the
    # union of the bins any member pair used.
    pooled_rows = []
    for pooled_name, members in SEP_POOL.items():
        member_rows = per_pair[per_pair["Comparison"].isin(members)]
        for group_key, group in member_rows.groupby(["Graph", "Curvature", "Aggregation"], sort=False):
            graph_upper, curvature, aggregation = group_key
            s_cond = float(group["S_sep_cond"].mean())
            s_kept = float(group["S_sep_kept"].mean())
            share = np.nan
            if s_kept > 0:
                share = 1.0 - s_cond / s_kept
            used = set()
            for bins_text in group["bins"]:
                used.update(bins_text.split(";"))
            used_bins = []
            n_kept = 0
            for label in bin_labels:
                if label in used:
                    used_bins.append(label)
                    n_kept += mols_in_bin[label]
            coverage = np.nan
            if n_total_mols:
                coverage = n_kept / n_total_mols
            pooled_rows.append({
                "Graph": graph_upper,
                "Comparison": pooled_name,
                "Curvature": curvature,
                "Aggregation": aggregation,
                "CondVar": cond_var,
                "S_sep_cond": s_cond,
                "S_sep_kept": s_kept,
                "share": share,
                "n_bins": len(used_bins),
                "bins": ";".join(used_bins),
                "n_A": int(group["n_A"].sum()),
                "n_B": int(group["n_B"].sum()),
                "n_mols_kept": n_kept,
                "n_mols_total": int(n_total_mols),
                "coverage": coverage,
                "n_pairs": len(group),
            })
    if not pooled_rows:
        return per_pair
    return pd.concat([per_pair, pd.DataFrame(pooled_rows)], ignore_index=True)


def score_conditioned_slice(df_subset_pairs_binned, df_across_samesubset_binned, graph_types, comparison, curvature, aggregation, cond_var, bins):
    """
    Score one (comparison, curvature, aggregation) slice separately in
    every bin. A row is kept when at least one of the two scores exists,
    since radius_bond, for example, loses S_sep on nonbond edges in most
    bins while S_sens survives.

    :param df_subset_pairs_binned: binned across-subset W1 table
    :param df_across_samesubset_binned: binned across-construction table
    :param graph_types: constructions to score
    :param comparison: pair label, a key of SUBSET_PAIRS
    :param curvature: "orc" or "frc"
    :param aggregation: "union" or "per_mol_mean"
    :param cond_var: conditioning variable name
    :param bins: bins for cond_var
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
    """
    Molecule-weighted mean of a column over the rows where it exists and
    the weight is positive.

    :param rows: frame with col and a weight column "w"
    :param col: column to average
    :return: the weighted mean, or nan if no row qualifies
    """
    valid = rows[col].notna() & (rows["w"] > 0)
    if not valid.any():
        return np.nan
    weights = rows.loc[valid, "w"]
    return float((rows.loc[valid, col] * weights).sum() / weights.sum())


def weighted_summary(all_cond_df, mols_per_bin, matched):
    """
    Conditioned scores per (graph, comparison, curvature, aggregation).
    S_sens is the molecule-weighted mean over bins of the per-bin score.
    S_sep, S_sep_kept, the share and the coverage come from
    matched_denominators; the weighted S_sep is also recomputed from the
    per-bin table as a check, and the two should agree exactly.

    :param all_cond_df: concatenated per-bin scores
    :param mols_per_bin: second output of collect_binned_distributions
    :param matched: output of matched_denominators
    :return: one row per (Graph, Comparison, Curvature, Aggregation)
    """
    first_graph = mols_per_bin["Graph"].iloc[0]
    first_rows = mols_per_bin[mols_per_bin["Graph"] == first_graph]
    used_rows = first_rows[first_rows["used"].astype(bool)]
    bin_weights = used_rows.groupby("Bin")["n_mols"].sum()

    scores = all_cond_df.copy()
    scores["w"] = scores["Bin"].astype(str).map(bin_weights)
    scores["w"] = scores["w"].fillna(0.0).astype(float)

    sens_rows = []
    for group_key, group in scores.groupby(SCORE_KEYS):
        mean_terms_sens = np.nan
        if "n_terms_sens" in group.columns:
            mean_terms_sens = float(group["n_terms_sens"].mean())
        valid_sens = group[SENS_COL].notna() & (group["w"] > 0)
        sens_rows.append({
            "Graph": group_key[0],
            "Comparison": group_key[1],
            "Curvature": group_key[2],
            "Aggregation": group_key[3],
            "S_sep_check": weighted_mean(group, SEP_COL),
            SENS_COL: weighted_mean(group, SENS_COL),
            "n_bins_sens": int(valid_sens.sum()),
            "mean_terms_sens": mean_terms_sens,
        })
    sens_cols = ["S_sep_check", SENS_COL, "n_bins_sens", "mean_terms_sens"]
    sens = pd.DataFrame(sens_rows, columns=SCORE_KEYS + sens_cols)

    sep_cols = [SEP_COL, "S_sep_kept", "share", "coverage", "n_bins_sep", "n_mols_kept", "n_mols_total"]
    if matched is not None and not matched.empty:
        matched_cols = ["S_sep_cond", "S_sep_kept", "share", "coverage", "n_bins", "n_mols_kept", "n_mols_total"]
        sep = matched[SCORE_KEYS + matched_cols]
        sep = sep.rename(columns={"S_sep_cond": SEP_COL, "n_bins": "n_bins_sep"})
    else:
        sep = pd.DataFrame(columns=SCORE_KEYS + sep_cols)

    summary = sep.merge(sens, on=SCORE_KEYS, how="outer")
    # Pooled comparisons take S_sep from matched_denominators (mean over pairs
    # of S_sep_cond, as for the jets). S_sep_check pools per bin instead, which
    # differs when the pairs cover different bins, so it is only compared on
    # the single-pair rows.
    pooled = summary["Comparison"].isin(list(SEP_POOL))
    both = summary[SEP_COL].notna() & summary["S_sep_check"].notna() & ~pooled
    if both.any():
        differences = (summary.loc[both, SEP_COL] - summary.loc[both, "S_sep_check"]).abs()
        print(f"  [check] weighted S_sep from the per-bin table vs matched_denominators: max |diff| = {float(differences.max()):.3e} over {int(both.sum())} rows",flush=True)

    final_cols = SCORE_KEYS + sep_cols + [SENS_COL, "n_bins_sens", "mean_terms_sens"]
    summary = summary[final_cols]
    return summary.sort_values(SCORE_KEYS).reset_index(drop=True)


def score_slice(df_subset_pairs, df_across_samesubset, graph_types, comparison, curvature,aggregation):
    """
    Unconditioned S_sep and S_sens of one slice. Constructions missing a
    required score are dropped, since they are not a point on the
    separation-sensitivity plane.

    :param df_subset_pairs: across-subset W1 table
    :param df_across_samesubset: across-construction W1 table
    :param graph_types: constructions to score
    :param comparison: pair label, a key of SUBSET_PAIRS
    :param curvature: "orc" or "frc"
    :param aggregation: "union" or "per_mol_mean"
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
    })
    scores = scores.dropna(subset=required)
    if scores.empty:
        print(f"[warn] empty slice: {comparison} | {curvature} | {aggregation} (no construction has {required})")
    scores = scores.reset_index().rename(columns={"index": "Graph"})
    scores["Comparison"] = comparison
    scores["Curvature"] = curvature
    scores["Aggregation"] = aggregation
    return scores


def run_conditioned_scores(data_dict, graph_types, cond_var, bins, save_dir, min_mols=MIN_MOLS_PER_BIN, make_plots=True):
    """
    Conditioned pipeline for one conditioning variable: bin the
    distributions, compute the W1 tables inside each bin, compute the
    matched denominators, score every slice per bin, and write the
    molecule-weighted summary, CSVs and plots.

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :param graph_types: constructions to score
    :param cond_var: "n", "n_H", "n_heavy", "h_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges for cond_var
    :param save_dir: output directory
    :param min_mols: bins with fewer molecules than this are left empty
    :param make_plots: whether to draw the per-bin figures
    :return: output of weighted_summary, or None if no slice was scored
    """
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = cond_bin_labels(cond_var, bins)
    pass_start = time.time()
    print(f"\n conditioned on {cond_var}, {len(bin_labels)} bins: {bin_labels} ")

    # The coverage counts every molecule, including those outside all bins
    first_nodes = data_dict[graph_types[0]]["nodes"]
    n_total_mols = int(first_nodes["mol_id"].nunique())

    data_binned, mols_per_bin = collect_binned_distributions(data_dict, graph_types, cond_var, bins, min_mols=min_mols)
    save_csv(mols_per_bin, save_dir, f"mols_per_bin_{cond_var}.csv")

    w1_tables = compute_binned_w1_tables(data_binned, graph_types, cond_var, bins, across_graphs=not SKIP_SENSITIVITY)
    across_graphs_nodes, across_graphs_edges, across_subsets_nodes, across_subsets_edges = w1_tables

    # The matched denominators are read from data_binned, so this has to run before it is freed
    matched_start = time.time()
    matched = matched_denominators(data_binned, across_subsets_nodes, across_subsets_edges, graph_types, mols_per_bin, n_total_mols, cond_var, bins)
    print(f"  matched denominators: {len(matched)} (construction, slice) rows in "
          f"{(time.time() - matched_start) / 60:.1f} min", flush=True)
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

    slice_inputs = [
        ("nodes", NODE_SCORED, across_subsets_nodes, across_graphs_nodes),
        ("edges", EDGE_SCORED, across_subsets_edges, across_graphs_edges),
    ]
    all_rows = []
    for table_kind, comparisons, df_subsets, df_graphs in slice_inputs:
        for comparison, curvature, aggregation in product(comparisons, CURVATURES, AGGREGATIONS):
            df_slice = score_conditioned_slice(df_subsets, df_graphs, graph_types, comparison, curvature, aggregation, cond_var, bins)
            print(f"\n{table_kind.upper()} (conditioned on {cond_var}): {comparison} {curvature} {aggregation}")
            if df_slice.empty:
                print("[warn] empty conditioned slice")
                continue
            print(df_slice.to_string(index=False))
            file_name = (f"scores_conditioned_{cond_var}_{table_kind}_{comparison}_{curvature}_{aggregation}.csv")
            save_csv(df_slice, save_dir, file_name.replace("/", "_"))
            all_rows.append(df_slice)

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

    if make_plots:
        score_cols = (SEP_COL,)
        if not SKIP_SENSITIVITY:
            score_cols = (SEP_COL, SENS_COL)
        plot_scores_vs_bin(all_cond_df, cond_var, bin_labels, save_dir, STYLING, score_cols=score_cols)

    print(f"{cond_var} finished in {(time.time() - pass_start) / 60:.1f} min ",flush=True)
    return weighted

