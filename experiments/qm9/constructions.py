import matplotlib.pyplot as plt

from experiments.qm9.config import ALL_GRAPH_TYPES, GRAPH_TYPES
from plotting.common import LINESTYLES, MARKERS, GraphStyling
from utils.duplicates import  duplicate_pairs_from_table,duplicate_pairs_from_values, register_duplicate_pairs
from utils.tables import upper_names

DUPLICATE_GRAPHS = []

TAB20 = plt.get_cmap("tab20")
COLOR_ORDER = list(range(0, 20, 2)) + list(range(1, 20, 2))
GRAPH_STYLES = {}

STYLING = GraphStyling(graph_style, graph_label, graph_sort_key)

def style_for_index(index):
    """
    Plot style of the construction at a given position.

    :param index: position of the construction in the style registry
    :return: dict with "color", "marker" and "ls"
    """
    return {
        "color": TAB20(COLOR_ORDER[index % 20]),
        "marker": MARKERS[index % len(MARKERS)],
        "ls": LINESTYLES[(index // 20) % len(LINESTYLES)],
    }


def graph_style(graphs):
    """
    Fixed style of each construction. The registry is filled on first use
    from every known construction in sorted order, so a construction looks
    the same in every figure.

    :param graphs: construction names, any case
    :return: {name as given: {"color", "marker", "ls"}}
    """
    if not GRAPH_STYLES:
        known = set(upper_names(list(ALL_GRAPH_TYPES) + list(GRAPH_TYPES)))
        for index, name in enumerate(sorted(known)):
            GRAPH_STYLES[name] = style_for_index(index)
    for graph in graphs:
        if graph.upper() not in GRAPH_STYLES:
            GRAPH_STYLES[graph.upper()] = style_for_index(len(GRAPH_STYLES))
    styles = {}
    for graph in graphs:
        styles[graph] = GRAPH_STYLES[graph.upper()]
    return styles


def graph_label(name):
    #QM9 constructions are labeled by their name
    return name


def graph_sort_key(name):
    #QM9 constructions are ordered by name
    return name


def pick_drop(name_a, name_b):
    """
    Choose which of two identical constructions to leave out of scoring.
    No QM9 construction is a known alias of another, so the name that
    sorts first is kept.

    :param name_a: first construction name
    :param name_b: second construction name
    :return: the lower-cased name to drop
    """
    lower_a = name_a.lower()
    lower_b = name_b.lower()
    if lower_a < lower_b:
        return lower_b
    return lower_a


def register_duplicates_from_table(df_across):
    """
    Duplicates from the across-construction W1 table (W1 = 0 on every row)
    """
    register_duplicate_pairs(DUPLICATE_GRAPHS, duplicate_pairs_from_table(df_across), "W1 = 0 on every row", pick_drop)


def register_duplicates_from_values(data_dict, graph_types):
    """
    Duplicate check without the across-construction W1. Two constructions
    are identical when every atom has the same ORC and FRC in both. The
    subset masks come from truth columns shared by all constructions, so
    this implies W1 = 0 on every node subset, curvature and aggregation.

    :param data_dict: {graph_type: {"nodes": df, ...}}
    :param graph_types: constructions to check
    :return: None
    """
    pairs = duplicate_pairs_from_values(data_dict, graph_types, "mol_id")
    register_duplicate_pairs(DUPLICATE_GRAPHS, pairs, "same ORC and FRC on every atom", pick_drop)

