import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plotting.common import aggregation_handles,graph_line_handles, hide_unused_axes, ordered_comparisons,present_bins,save_figure, slice_title, tick_rotation, CURVATURE_LABELS
from utils.tables import SEP_COL, SLICE_KEYS


def plot_lines_through_finite(ax, df, labels, col, styling):
    """
    One line per construction through the bins where it has a finite value.
    Bins without a value are skipped, so the line runs straight through them.

    :param ax: matplotlib axis
    :param df: rows of one slice with Graph, Bin, and col
    :param labels: x-axis bin labels
    :param col: value column
    :param styling: GraphStyling of the experiment
    :return: list of constructions drawn
    """
    x = np.arange(len(labels))
    table = df.copy()
    table["Bin"] = table["Bin"].astype(str)
    graphs = styling.sorted(table["Graph"].unique())
    styles = styling.style_of(graphs)
    drawn = []
    for graph in graphs:
        graph_rows = table[table["Graph"] == graph]
        y = graph_rows.set_index("Bin")[col].reindex(labels).to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            continue
        style = styles[graph]
        ax.plot(x[finite], y[finite], color=style["color"], marker=style["marker"], ls=style["ls"], ms=3.5, lw=1.1, label=styling.label_of(graph))
        drawn.append(graph)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=tick_rotation(labels), fontsize=7)
    return drawn


def finish_line_figure(fig, graphs, file_stem, save_dir, styling, ncol=4):
    """
    Add a shared construction legend below a line figure and save it.

    :param fig: the figure
    :param graphs: constructions drawn in the figure
    :param file_stem: file name without extension
    :param save_dir: output directory
    :param styling: GraphStyling of the experiment
    :param ncol: legend columns
    :return: None
    """
    fig.tight_layout()
    extra_artists = ()
    if graphs:
        legend = fig.legend(handles=graph_line_handles(graphs, styling), loc="upper center", ncol=ncol, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
        extra_artists = (legend,)
    save_figure(fig, save_dir, file_stem, extra_artists, 200)


def plot_scores_vs_bin(all_cond_df, cond_var, bin_labels, save_dir, styling, score_cols=(SEP_COL,), comparison_labels=None, legend_ncol=4):
    """
    Scores across bins, one line per construction, one panel per score
    column and one figure per (comparison, curvature, aggregation). Only the
    bins where a score exists are shown.

    :param all_cond_df: per-bin scores
    :param cond_var: conditioning variable, used for the x label and file name
    :param bin_labels: bin labels in plotting order
    :param save_dir: output directory
    :param styling: GraphStyling of the experiment
    :param score_cols: score columns, one panel each
    :param comparison_labels: {comparison: display text} for the titles
    :param legend_ncol: legend columns
    :return: None
    """
    if comparison_labels is None:
        comparison_labels = {}
    for slice_key, slice_rows in all_cond_df.groupby(SLICE_KEYS):
        comparison, curvature, aggregation = slice_key
        if not present_bins(slice_rows, bin_labels, score_cols):
            continue

        fig, axes = plt.subplots(1, len(score_cols), figsize=(6.5 * len(score_cols), 4.4), squeeze=False)
        drawn = set()
        for col, ax in zip(score_cols, axes[0]):
            labels = present_bins(slice_rows, bin_labels, (col,))
            if not labels:
                ax.set_title("no score in any bin", fontsize=8)
                continue
            drawn.update(plot_lines_through_finite(ax, slice_rows, labels, col, styling))
            ax.set_xlabel(cond_var)
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)
        axes[0][0].set_title(slice_title(comparison, curvature, aggregation, comparison_labels), fontsize=9, loc="left")

        file_stem = f"plot_conditioned_{cond_var}_{comparison}_{curvature}_{aggregation}"
        finish_line_figure(fig, styling.sorted(drawn), file_stem.replace("/", "_"), save_dir, styling, ncol=legend_ncol)


def pairs_present(df, pair_labels):
    """
    Pair labels that appear in df, in the order of pair_labels.

    :param df: table with a FlavorPair column
    :param pair_labels: pair labels in the desired order
    :return: list of pair labels
    """
    available = set(df["FlavorPair"].unique())
    pairs = []
    for pair in pair_labels:
        if pair in available:
            pairs.append(pair)
    return pairs


def plot_pair_scores_vs_bin(pair_bins_df, cond_var, bin_labels, pair_labels, save_dir, styling, pair_style, comparison_labels):
    """
    S_sep(g, pair) across bins: one panel per population pair, one line per
    construction, one figure per (comparison, curvature, aggregation).

    :param pair_bins_df: per-pair, per-bin scores
    :param cond_var: conditioning variable
    :param bin_labels: bin labels in plotting order
    :param pair_labels: pairs, in panel order
    :param save_dir: output directory
    :param styling: GraphStyling of the experiment
    :param pair_style: function pair label -> (color, display text)
    :param comparison_labels: {comparison: display text} for the titles
    :return: None
    """
    if pair_bins_df is None or pair_bins_df.empty:
        return

    for group_key, group in pair_bins_df.groupby(SLICE_KEYS):
        comparison, curvature, aggregation = group_key
        labels = present_bins(group, bin_labels, [SEP_COL])
        pairs = pairs_present(group, pair_labels)
        if not labels or not pairs:
            continue

        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2), squeeze=False)
        drawn = set()
        for ax, pair in zip(axes[0], pairs):
            pair_rows = group[group["FlavorPair"] == pair]
            drawn.update(plot_lines_through_finite(ax, pair_rows, labels, SEP_COL, styling))
            ax.set_xlabel(cond_var)
            ax.set_ylabel(SEP_COL, fontsize=8)
            ax.set_title(pair_style(pair)[1], fontsize=10)
            ax.grid(alpha=0.3)
        fig.suptitle(slice_title(comparison, curvature, aggregation, comparison_labels), fontsize=9, x=0.01, ha="left")

        file_stem = f"plot_conditioned_{cond_var}_per_pair_{comparison}_{curvature}_{aggregation}"
        finish_line_figure(fig, styling.sorted(drawn), file_stem.replace("/", "_"), save_dir, styling)


def plot_pair_scores_vs_bin_by_graph(pair_bins_df, cond_var, bin_labels, pair_labels, save_dir, styling, pair_style, comparison_labels, ncols=4):
    """
    S_sep(g, pair) across bins: one panel per construction, one line per
    population pair, one figure per (comparison, curvature, aggregation). The
    y axis is shared across panels.

    :param pair_bins_df: per-pair, per-bin scores
    :param cond_var: conditioning variable
    :param bin_labels: bin labels in plotting order
    :param pair_labels: pairs, in legend order
    :param save_dir: output directory
    :param styling: GraphStyling of the experiment
    :param pair_style: function pair label -> (color, display text)
    :param comparison_labels: {comparison: display text} for the titles
    :param ncols: panels per row
    :return: None
    """
    if pair_bins_df is None or pair_bins_df.empty:
        return

    for group_key, group in pair_bins_df.groupby(SLICE_KEYS):
        comparison, curvature, aggregation = group_key
        labels = present_bins(group, bin_labels, [SEP_COL])
        pairs = pairs_present(group, pair_labels)
        if not labels or not pairs:
            continue

        table = group.copy()
        table["Bin"] = table["Bin"].astype(str)
        graphs = styling.sorted(table["Graph"].unique())
        n_graphs = len(graphs)
        nrows = int(np.ceil(n_graphs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False, sharey=True)
        x = np.arange(len(labels))

        for index, graph in enumerate(graphs):
            row = index // ncols
            col = index % ncols
            ax = axes[row][col]
            graph_rows = table[table["Graph"] == graph]
            for pair in pairs:
                color, text = pair_style(pair)
                pair_rows = graph_rows[graph_rows["FlavorPair"] == pair]
                y = pair_rows.set_index("Bin")[SEP_COL].reindex(labels).to_numpy(dtype=float)
                finite = np.isfinite(y)
                if not finite.any():
                    continue
                ax.plot(x[finite], y[finite], color=color, marker="o", ms=3.5, lw=1.1, label=text)
            ax.set_title(styling.label_of(graph), fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=tick_rotation(labels), fontsize=6)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(SEP_COL, fontsize=7)
            if row == nrows - 1:
                ax.set_xlabel(cond_var, fontsize=8)
        hide_unused_axes(axes, n_graphs, ncols)

        handles = []
        for pair in pairs:
            color, text = pair_style(pair)
            handles.append(Line2D([0], [0], color=color, marker="o", lw=1.2, ms=4, label=text))
        fig.suptitle(slice_title(comparison, curvature, aggregation, comparison_labels), fontsize=10, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        legend = fig.legend(handles=handles, loc="upper center", ncol=len(pairs), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

        file_stem = (f"plot_conditioned_{cond_var}_pairs_by_graph_{comparison}_{curvature}_{aggregation}")
        save_figure(fig, save_dir, file_stem.replace("/", "_"), (legend,), 200)


def plot_share(share_df, cond_var, save_dir, styling, comparison_order, comparison_labels, aggregations, mean_label, unit, ncols=3):
    """
    Share per construction: one figure per curvature and one panel per
    comparison, constructions along x in legend order. Filled markers are
    union, open markers the per-cloud mean, and the coverage range is written
    into each panel title.

    :param share_df: output of share_table
    :param cond_var: conditioning variable, used in the file name and y label
    :param save_dir: output directory
    :param styling: GraphStyling of the experiment
    :param comparison_order: comparisons in panel order
    :param comparison_labels: {comparison: display text} for the panel titles
    :param aggregations: aggregation names, "union" and the per-cloud mean
    :param mean_label: legend text of the mean aggregation
    :param unit: "jets" or "molecules", for the figure title
    :param ncols: panels per row
    :return: None
    """
    for curvature in sorted(share_df["Curvature"].unique()):
        curv_rows = share_df[share_df["Curvature"] == curvature].dropna(subset=["share"])
        if curv_rows.empty:
            continue

        graphs = styling.sorted(curv_rows["Graph"].unique())
        styles = styling.style_of(graphs)
        comparisons = ordered_comparisons(curv_rows["Comparison"].unique(), comparison_order)
        n_panels = len(comparisons)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows), squeeze=False)
        x = np.arange(len(graphs))
        tick_labels = []
        for graph in graphs:
            tick_labels.append(styling.label_of(graph))

        for index, comparison in enumerate(comparisons):
            ax = axes[index // ncols][index % ncols]
            comp_rows = curv_rows[curv_rows["Comparison"] == comparison]
            for aggregation in aggregations:
                agg_rows = comp_rows[comp_rows["Aggregation"] == aggregation].set_index("Graph")
                for position, graph in enumerate(graphs):
                    if graph not in agg_rows.index:
                        continue
                    share = float(agg_rows.loc[graph, "share"])
                    if not np.isfinite(share):
                        continue
                    style = styles[graph]
                    face = "white"
                    if aggregation == "union":
                        face = style["color"]
                    ax.scatter(position, share, marker=style["marker"], s=55, facecolors=face, edgecolors=style["color"], linewidths=1.2, zorder=2)
            ax.axhline(0.0, color="gray", lw=0.8, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)

            coverage = comp_rows["coverage"].dropna()
            coverage_text = ""
            if not coverage.empty:
                coverage_text = f"  (coverage {coverage.min():.2f} to {coverage.max():.2f})"
            comparison_label = comparison_labels.get(comparison, comparison)
            ax.set_title(f"{comparison_label}{coverage_text}", fontsize=9)
            if index % ncols == 0:
                ax.set_ylabel(f"share removed by conditioning on {cond_var}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        hide_unused_axes(axes, n_panels, ncols)

        legend = fig.legend(handles=aggregation_handles(9, mean_label), loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.02))
        curvature_label = CURVATURE_LABELS.get(curvature, curvature)
        fig.suptitle(f"Share of separation attributable to {cond_var} ({curvature_label}); denominator on the same {unit}", fontsize=11, y=0.995)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_figure(fig, save_dir, f"share_{cond_var}_{curvature}", (legend,), 300)

