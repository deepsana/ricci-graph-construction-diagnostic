import os
import numpy as np

SEP_COL = "Separability (higher better)"
SENS_COL = "Sensitivity (lower better)"
SCORE_KEYS = ["Graph", "Comparison", "Curvature", "Aggregation"]
SLICE_KEYS = ["Comparison", "Curvature", "Aggregation"]


SEP_COL = "Separability (higher better)"
SENS_COL = "Sensitivity (lower better)"
# A score row is identified by these columns; a slice drops the construction.
SCORE_KEYS = ["Graph", "Comparison", "Curvature", "Aggregation"]
SLICE_KEYS = ["Comparison", "Curvature", "Aggregation"]


def upper_names(names):
    """
    Construction names as they appear in the CSVs (upper case)
    """
    return [name.upper() for name in names]


def safe_divide(numerator, denominator):
    """
    Elementwise numerator / denominator, NaN where the denominator is not positive
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(len(numerator), np.nan)
    positive = denominator > 0
    result[positive] = numerator[positive] / denominator[positive]
    return result


def preserved_sep(s_sep_cond, s_sep_kept):
    """
    Fraction of the separation that survives conditioning:
    S_sep_cond / S_sep_kept. NaN where S_sep_kept is not positive.
    The share removed by conditioning is 1 - preserved_sep.
    Works on scalars and arrays alike.
    """
    s_sep_cond = np.asarray(s_sep_cond, dtype=float)
    s_sep_kept = np.asarray(s_sep_kept, dtype=float)
    return s_sep_cond / np.where(s_sep_kept > 0, s_sep_kept, np.nan)


def apply_value_guard(table, min_values):
    """
    Drop W1 rows where either side has fewer than min_values values.
    A min_values of 0, or a table without n_A / n_B, leaves the table as is.
    """
    has_counts = "n_A" in table.columns and "n_B" in table.columns
    if min_values > 0 and has_counts:
        table = table[(table["n_A"] >= min_values) & (table["n_B"] >= min_values)]
    return table


def save_csv(df, save_dir, file_name):
    df.to_csv(os.path.join(save_dir, file_name), index=False)


def share_table(all_scores_df, weighted, cond_var, unit, extra_cols):
    """
    Share of the separation removed by conditioning on cond_var:

        preserved_sep = S_sep(conditioned) / S_sep(unconditioned, same point clouds)
        share         = 1 - preserved_sep

    Both terms come from the same jets or molecules, so no coverage cutoff is
    needed. The same two quantities against the full-sample score
    (*_all_<unit>_denom) are kept as a cross-check: they agree with the
    matched ones when coverage is high.

    :param all_scores_df: unconditioned scores
    :param weighted: weighted summary of the conditioned pass
    :param cond_var: conditioning variable
    :param unit: "jets" or "mols", the word used in the count columns
    :param extra_cols: columns of weighted to carry along, e.g. ["n_pairs"]
    :return: one row per SCORE_KEYS
    """
    all_col = f"S_sep_all_{unit}"
    reference = all_scores_df[SCORE_KEYS + [SEP_COL]].rename(columns={SEP_COL: all_col})
    merged = weighted.merge(reference, on=SCORE_KEYS, how="left")
    preserved_all = preserved_sep(merged[SEP_COL], merged[all_col])
    merged[f"share_all_{unit}_denom"] = 1.0 - preserved_all
    merged[f"preserved_sep_all_{unit}_denom"] = preserved_all
    merged["CondVar"] = cond_var
    cols = SCORE_KEYS + ["CondVar", SEP_COL, "S_sep_kept", "share", "preserved_sep", "coverage"]
    cols += list(extra_cols)
    cols += [f"n_{unit}_kept", f"n_{unit}_total", all_col, f"share_all_{unit}_denom",f"preserved_sep_all_{unit}_denom"]
    return merged[cols]

