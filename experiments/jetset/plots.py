
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.jetset.config import AGG_LINESTYLES, COMPARISON_LABELS, COMPARISON_ORDER, CURVATURES, DEFAULT_KMAX, KMAX_MARKERS, P_COLOR_IDX,SWEEP_AXES, SWEEP_GRAPH_TYPES
from experiments.jetset.constructions import parse_graph
from plotting.common import CURVATURE_LABELS, ordered_comparisons, save_figure
from utils.tables import SEP_COL


# For Laman and Unique-k, p = 0 and p = +1 give the same graph.
RIGID_FAMILIES = ["laman", "unique_k"]
SWEEP_PARAMS = ("k", "r", "q", "kmax", "p")


def p_color(p_value):
    """tab10 color of a p exponent; black for NaN (families without p)."""
    if np.isnan(p_value):
        return "black"
    return plt.get_cmap("tab10")(P_COLOR_IDX.get(int(p_value), 5))


def p_legend_label(p_value, merged_rigid):
    """
    :param merged_rigid: True if p = 0 and p = +1 were drawn as one series for Laman/Unique-k
    """
    if p_value == 1 and merged_rigid:
        return r"$p=+1$ ($=p{=}0$ for Laman/Unique)"
    if p_value != 0:
        return f"$p={p_value:+d}$"
    return "$p=0$"


def draw_sweep_lines(ax, family_rows, aggregation, linestyle, x_param, y_col):
    """
    Draw one aggregation on a sweep panel.
    On the q axis, kT variants with a non default kmax are drawn as single markers instead of joining the line.

    :param family_rows: rows of the families on this sweep axis
    :param x_param: swept parameter on the x axis
    :return: True if a line was drawn
    """
    rows = family_rows[family_rows["Aggregation"] == aggregation]
    if rows.empty:
        return False

    if x_param == "q":
        off_line = rows[rows["kmax"] != DEFAULT_KMAX]
        rows = rows[rows["kmax"] == DEFAULT_KMAX]
    else:
        off_line = rows.iloc[0:0]

    drawn = False
    if rows["p"].notna().any():
        for p_value, p_rows in rows.groupby("p"):
            p_rows = p_rows.sort_values(x_param)
            ax.plot(p_rows[x_param].to_numpy(), p_rows[y_col].to_numpy(), ls=linestyle,
                    marker="o", ms=4, lw=1.2, color=p_color(p_value))
            drawn = True
    elif not rows.empty:
        rows = rows.sort_values(x_param)
        ax.plot(rows[x_param].to_numpy(), rows[y_col].to_numpy(), ls=linestyle, marker="o",
                ms=4, lw=1.2, color="black")
        drawn = True

    for _, off_row in off_line.iterrows():
        color = p_color(off_row["p"])
        face = "none" if aggregation == "per_jet_mean" else color
        marker = KMAX_MARKERS.get(int(off_row["kmax"]), "x")
        ax.scatter(off_row[x_param], off_row[y_col], marker=marker, s=45, facecolors=face,
                   edgecolors=color, zorder=3)
    return drawn


def with_sweep_params(df):
    """
    Add the family and the parsed parameters of every construction as
    columns. p = 0 of a rigid family is relabeled p = +1 so both draw as one
    series.
    """
    parsed = {graph: parse_graph(graph) for graph in df["Graph"].unique()}
    df["family"] = df["Graph"].map({graph: info["family"] for graph, info in parsed.items()})
    for param in SWEEP_PARAMS:
        df[param] = df["Graph"].map({graph: info.get(param, np.nan)
                                     for graph, info in parsed.items()})
    rigid = df["family"].isin(RIGID_FAMILIES) & (df["p"] == 0)
    df.loc[rigid, "p"] = 1
    return df


def sweep_legend_handles(comp_rows):
    """
    Legend: p colors and kmax markers present, plus the aggregation styles
    """
    tab10 = plt.get_cmap("tab10")
    merged_rigid = (comp_rows["family"].isin(RIGID_FAMILIES) & (comp_rows["p"] == 1)).any()
    handles = []
    for p_value in (-1, 0, 1):
        if (comp_rows["p"] == p_value).any():
            handles.append(Line2D([0], [0], color=tab10(P_COLOR_IDX[p_value]), lw=1.5,
                                  label=p_legend_label(p_value, merged_rigid)))
    for aggregation, linestyle in AGG_LINESTYLES.items():
        handles.append(Line2D([0], [0], color="black", lw=1.5, ls=linestyle,
                              label=aggregation.replace("_", "-")))
    for kmax, marker in KMAX_MARKERS.items():
        if (comp_rows["kmax"] == kmax).any():
            handles.append(Line2D([0], [0], marker=marker, color="w", markeredgecolor="black",
                                  markerfacecolor="gray", markersize=8,
                                  label=f"$k_{{\\max}}={kmax}$"))
    return handles



def plot_parameter_sweep(scores_df, curvature, sep_col, out_stem, save_dir, title):
    """
    One figure per comparison, one panel per sweep axis (SWEEP_AXES), colored
    by p; solid lines are union, dashed the per-jet mean.

    :param scores_df: score table
    :param out_stem: file name stem; the comparison is appended
    :param title: figure title prefix
    """
    df = scores_df[scores_df["Curvature"] == curvature].dropna(subset=[sep_col]).copy()
    if df.empty:
        print(f"[warn] plot_parameter_sweep: nothing to plot for {out_stem}")
        return
    df = with_sweep_params(df)
    curvature_label = CURVATURE_LABELS.get(curvature, curvature)

    for comparison in ordered_comparisons(df["Comparison"].unique(), COMPARISON_ORDER):
        comp_rows = df[df["Comparison"] == comparison]
        n_axes = len(SWEEP_AXES)
        fig, axes = plt.subplots(1, n_axes, figsize=(4.2 * n_axes, 3.9), squeeze=False)
        any_drawn = False

        for ax, (families, x_param, x_label) in zip(axes[0], SWEEP_AXES):
            family_rows = comp_rows[comp_rows["family"].isin(families)]
            for aggregation, linestyle in AGG_LINESTYLES.items():
                if draw_sweep_lines(ax, family_rows, aggregation, linestyle, x_param, sep_col):
                    any_drawn = True
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(sep_col, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        if not any_drawn:
            plt.close(fig)
            continue

        handles = sweep_legend_handles(comp_rows)
        legend = fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,
                            frameon=False, bbox_to_anchor=(0.5, 0.11))
        comparison_label = COMPARISON_LABELS.get(comparison, comparison)
        fig.suptitle(f"{title} | {comparison_label} | {curvature_label}", fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0.13, 1, 0.93])
        save_figure(fig, save_dir, f"{out_stem}_{comparison}", (legend,), 300)


def plot_sweep_set(scores, stem, title_suffix, save_dir):
    """
    Parameter sweep figures of S_sep for every curvature, nothing is drawn without sweep constructions

    :param scores: score table
    :param stem: file name stem
    :param title_suffix: text appended to the figure titles
    :param save_dir: output directory
    :return: None
    """
    if len(SWEEP_GRAPH_TYPES) == 0:
        return
    for curvature in CURVATURES:
        plot_parameter_sweep(scores, curvature, SEP_COL, f"sweep_{stem}_{curvature}", save_dir, title=f"Parameter sweep{title_suffix}")

