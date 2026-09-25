
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p", "d", "8", "H"]
LINESTYLES = ["-", "--", ":", "-."]

CURVATURE_LABELS = {"frc": "Forman-Ricci Curvature", "orc": "Ollivier-Ricci Curvature"}


class GraphStyling:
    """
    How one experiment draws its constructions.

    :param style_of: function names -> {name: {"color", "marker", "ls"}}
    :param label_of: function name -> legend or tick text
    :param sort_key: function name -> sort key that fixes the legend order
    """

    def __init__(self, style_of, label_of, sort_key):
        self.style_of = style_of
        self.label_of = label_of
        self.sort_key = sort_key

    def sorted(self, graphs):
        """Construction names in legend order."""
        return sorted(graphs, key=self.sort_key)


def save_figure(fig, save_dir, file_stem, extra_artists=(), dpi=300):
    """
    Write a figure as pdf and png and close it.

    :param fig: the figure
    :param save_dir: output directory
    :param file_stem: file name without extension
    :param extra_artists: artists outside the axes, such as legends
    :param dpi: resolution of the png
    :return: None
    """
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(save_dir, f"{file_stem}.{extension}"), bbox_inches="tight", bbox_extra_artists=extra_artists, dpi=dpi)
    plt.close(fig)


def hide_unused_axes(axes, n_used, ncols):
    """
    Turn off the panels of a grid that hold no plot.

    :param axes: 2D array of axes from plt.subplots(squeeze=False)
    :param n_used: number of panels in use, filled row by row
    :param ncols: panels per row
    :return: None
    """
    for index in range(n_used, len(axes) * ncols):
        axes[index // ncols][index % ncols].axis("off")


def tick_rotation(labels):
    """
    :param labels: x tick labels
    :return: 90 for long label lists, 0 otherwise
    """
    if len(labels) > 8:
        return 90
    return 0


def present_bins(df, bin_labels, value_cols):
    """
    Bins, in order, where at least one row has a value in one of value_cols.

    :param df: table with a Bin column
    :param bin_labels: labels in plotting order
    :param value_cols: columns to check
    :return: list of labels
    """
    bin_strings = df["Bin"].astype(str)
    counts = df[list(value_cols)].groupby(bin_strings).count().sum(axis=1)
    labels = []
    for label in bin_labels:
        if counts.get(label, 0) > 0:
            labels.append(label)
    return labels


def ordered_comparisons(present, order):
    """
    Comparisons in the given order first, then any others sorted.

    :param present: comparison names that occur in a table
    :param order: preferred order
    :return: list of comparison names
    """
    present = set(present)
    ordered = []
    for comparison in order:
        if comparison in present:
            ordered.append(comparison)
    for comparison in sorted(present):
        if comparison not in order:
            ordered.append(comparison)
    return ordered


def slice_title(comparison, curvature, aggregation, comparison_labels):
    """
    :param comparison: structural subset or comparison name
    :param curvature: "orc" or "frc"
    :param aggregation: aggregation name
    :param comparison_labels: {comparison: display text}
    :return: title text for one slice
    """
    comparison_label = comparison_labels.get(comparison, comparison)
    curvature_label = CURVATURE_LABELS.get(curvature, curvature)
    return f"{comparison_label} | {curvature_label} | {aggregation}"


def aggregation_handles(marker_size, mean_label):
    """
    Legend handles for the two aggregations: filled for union, open for the
    per-cloud mean.

    :param marker_size: marker size in points
    :param mean_label: legend text of the mean aggregation
    :return: list of Line2D
    """
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=marker_size, label="union"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="black", markersize=marker_size, label=mean_label),
    ]


def graph_line_handles(graphs, styling):
    """
    Legend handles for line plots, one per construction in legend order.

    :param graphs: construction names
    :param styling: GraphStyling of the experiment
    :return: list of Line2D
    """
    styles = styling.style_of(graphs)
    handles = []
    for graph in styling.sorted(graphs):
        style = styles[graph]
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"], ls=style["ls"], lw=1.2, ms=4, label=styling.label_of(graph)))
    return handles

