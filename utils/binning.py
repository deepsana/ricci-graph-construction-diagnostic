import pandas as pd

def bin_label(low, high):
    """
    label for an inclusive bin, e.g. (3, 5) -> "3-5" and (0, 0) -> "0"

    :param low: lower edge
    :param high: upper edge
    :return: bin label
    """
    if low == high:
        return f"{low:g}"
    return f"{low:g}-{high:g}"


def bin_labels_of(bins):
    """
    Labels of every bin, in bin order

    :param bins: list of inclusive (low, high) ranges
    :return: list of labels
    """
    labels = []
    for low, high in bins:
        labels.append(bin_label(low, high))
    return labels


def bin_series(values, bins):
    """
    Assign every value to the first bin that contains it

    :param values: numeric Series indexed by point-cloud id
    :param bins: list of inclusive (low, high) ranges, checked in order
    :return: object Series of bin labels, None where no bin matches
    """
    labels = pd.Series([None] * len(values), index=values.index, dtype=object)
    for low, high in bins:
        in_bin = labels.isna() & (values >= low) & (values <= high)
        labels[in_bin] = bin_label(low, high)
    return labels


def joint_bin_labels(first_bins, second_bins, joint_label):
    """
    Every label of a two-variable grid, first variable major.

    :param first_bins: bins of the first variable
    :param second_bins: bins of the second variable
    :param joint_label: function (first label, second label) -> cell label
    :return: list of labels
    """
    labels = []
    for first_label in bin_labels_of(first_bins):
        for second_label in bin_labels_of(second_bins):
            labels.append(joint_label(first_label, second_label))
    return labels


def joint_bin_series(first, first_bins, second, second_bins, joint_label):
    """
    Cell of the two-variable grid for every point cloud. A cloud gets a
    label only if it falls in a bin of both variables.

    :param first: Series of the first variable, indexed by point cloud id
    :param first_bins: its bins
    :param second: Series of the second variable, same index
    :param second_bins: its bins
    :param joint_label: function (first label, second label) -> cell label
    :return: object Series of labels, None outside the grid
    """
    first_labels = bin_series(first, first_bins)
    second_labels = bin_series(second, second_bins)
    in_both = first_labels.notna() & second_labels.notna()
    labels = pd.Series([None] * len(first), index=first.index, dtype=object)
    joint = []
    for first_label, second_label in zip(first_labels[in_both], second_labels[in_both]):
        joint.append(joint_label(first_label, second_label))
    if joint:
        labels[in_both] = joint
    return labels


def order_bins(df, bin_labels):
    """
    Make Bin an ordered categorical so that sorting follows bin order instead of string order

    :param df: table with a Bin column
    :param bin_labels: labels in the desired order
    :return: copy of df
    """
    df = df.copy()
    df["Bin"] = pd.Categorical(df["Bin"], categories=bin_labels, ordered=True)
    return df

