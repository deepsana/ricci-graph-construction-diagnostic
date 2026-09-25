import re
import time

import matplotlib.pyplot as plt

from experiments.jetset.config import COND_GRAPH_TYPES, DEFAULT_KMAX,DEFAULT_QUANTILE,DEFAULT_RADIUS, FAMILY_COLOR_IDX, FAMILY_ORDER, GRAPH_TYPES, KT_PATTERN,P_VALUES, PAIR_COLOR_IDX, PAIR_TEXT
from plotting.common import LINESTYLES, MARKERS, GraphStyling
from src.graphs import ALL_GRAPH_TYPES
from utils.duplicates import duplicate_pairs_from_values, register_duplicate_pairs

# Constructions identical to another one (for Laman and Unique-k, p = 0 and
# p = +1 give the same graph). Filled by register_duplicates and skipped when scoring. Lower-cased names
DUPLICATE_GRAPHS = []

# Style per upper-cased construction name, filled on first use so that a construction looks the same in every figure.
GRAPH_STYLE = {}


def parse_graph(name):
    """
    Family and parameters of a construction, parsed from its name.
    Laman gets k=2 so it can be drawn at k=2 on the Unique-k 

    :param name: construction name, e.g. "knn_3" or "unique_4_p_pos1"
    :return: dict with "family" and, where relevant, k, r, q, kmax, and p;
        {"family": "other"} for unknown names
    """
    name = name.lower()
    if name == "fully_connected":
        return {"family": "fully_connected"}

    match = re.fullmatch(r"knn_(\d+)", name)
    if match:
        return {"family": "knn", "k": int(match.group(1))}

    match = re.fullmatch(r"radius_based(?:_r(\d)p(\d+))?", name)
    if match:
        radius = DEFAULT_RADIUS
        if match.group(1):
            radius = float(f"{match.group(1)}.{match.group(2)}")
        return {"family": "radius", "r": radius}

    match = re.fullmatch(KT_PATTERN, name)
    if match:
        quantile = DEFAULT_QUANTILE
        if match.group(1):
            quantile = float(f"{match.group(1)}.{match.group(2)}")
        kmax = DEFAULT_KMAX
        if match.group(3):
            kmax = int(match.group(3))
        return {"family": "kt_threshold","q": quantile, "kmax": kmax,"p": P_VALUES[match.group(4)]}

    match = re.fullmatch(r"laman_p_(neg1|pos1|zero)", name)
    if match:
        return {"family": "laman", "k": 2, "p": P_VALUES[match.group(1)]}

    match = re.fullmatch(r"unique_(\d+)_p_(neg1|pos1|zero)", name)
    if match:
        return {"family": "unique_k", "k": int(match.group(1)), "p": P_VALUES[match.group(2)]}

    return {"family": "other"}


def graph_sort_key(name):
    """
    Sort key that orders constructions by family and then by parameters.

    :param name: construction name
    :return: tuple
    """
    parsed = parse_graph(name)
    return (FAMILY_ORDER.index(parsed["family"]), parsed.get("k", 0), parsed.get("r", 0.0),
            parsed.get("q", 0.0), parsed.get("kmax", 0), parsed.get("p", 0))


def p_text(name, parsed):
    """
    LaTeX fragment for the p exponent. The surviving p = +1 member of a
    dropped p = 0 twin is labeled p in {0, +1}.

    :param name: construction name
    :param parsed: output of parse_graph for this name
    :return: label fragment
    """
    p_value = parsed["p"]
    if p_value == 1:
        twin = re.sub(r"_p_pos1$", "_p_zero", name.lower())
        if twin in DUPLICATE_GRAPHS:
            return r"p\in\{0,+1\}"
    if p_value != 0:
        return f"p={p_value:+d}"
    return "p=0"


def graph_label(name):
    """
    Readable label for a construction, e.g. "knn_3" -> "$k$NN ($k=3$)".

    :param name: construction name
    :return: label; unknown names are returned unchanged
    """
    parsed = parse_graph(name)
    family = parsed["family"]
    if family == "fully_connected":
        return "Fully connected"
    if family == "knn":
        k = parsed["k"]
        return f"$k$NN ($k={k}$)"
    if family == "radius":
        radius = parsed["r"]
        return f"Radius ($r={radius:g}$)"
    if family == "kt_threshold":
        parts = [f"q={parsed['q']:g}"]
        if parsed["kmax"] != DEFAULT_KMAX:
            parts.append(f"k_{{\\max}}={parsed['kmax']}")
        parts.append(p_text(name, parsed))
        joined = ", ".join(parts)
        return f"$k_T$ thr. (${joined}$)"
    if family == "laman":
        return f"Laman (${p_text(name, parsed)}$)"
    if family == "unique_k":
        return f"Unique-{parsed['k']} (${p_text(name, parsed)}$)"
    return name


def register_graph_styles(names):
    """
    Give every new construction a color (by family), a marker, and a line
    style (by position within its family). Existing entries never change.

    :param names: construction names, any case
    :return: None
    """
    tab10 = plt.get_cmap("tab10")
    family_counts = {}
    for style in GRAPH_STYLE.values():
        family = style["family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    unique_names = set()
    for name in names:
        unique_names.add(name.upper())

    for name in sorted(unique_names, key=graph_sort_key):
        if name in GRAPH_STYLE:
            continue
        family = parse_graph(name)["family"]
        position = family_counts.get(family, 0)
        GRAPH_STYLE[name] = {
            "family": family,
            "color": tab10(FAMILY_COLOR_IDX.get(family, 5)),
            "marker": MARKERS[position % len(MARKERS)],
            "ls": LINESTYLES[position % len(LINESTYLES)],
        }
        family_counts[family] = position + 1


def graph_style(graphs):
    """
    Fixed style of each construction.

    :param graphs: construction names, any case
    :return: {name as given: {"family", "color", "marker", "ls"}}
    """
    if not GRAPH_STYLE:
        register_graph_styles(list(ALL_GRAPH_TYPES) + list(GRAPH_TYPES) + list(COND_GRAPH_TYPES))
    register_graph_styles(graphs)
    styles = {}
    for graph in graphs:
        styles[graph] = GRAPH_STYLE[graph.upper()]
    return styles


STYLING = GraphStyling(graph_style, graph_label, graph_sort_key)


def pair_style(pair):
    """
    Fixed color and display text of a flavor pair.

    :param pair: pair label, e.g. "b_vs_light"
    :return: tuple (color, text)
    """
    tab10 = plt.get_cmap("tab10")
    return tab10(PAIR_COLOR_IDX.get(pair, 7)), PAIR_TEXT.get(pair, pair)


def pick_drop(graph_a, graph_b):
    """
    Which of two identical constructions to leave out of the scores: a
    unique_2 variant first (it duplicates Laman), then a p = 0 variant (so the
    p = +1 one keeps its label), otherwise the second name.

    :param graph_a: first construction name
    :param graph_b: second construction name
    :return: lower-cased name to drop
    """
    name_a = graph_a.lower()
    name_b = graph_b.lower()

    if name_a.startswith("unique_2") and not name_b.startswith("unique_2"):
        return name_a
    if name_b.startswith("unique_2") and not name_a.startswith("unique_2"):
        return name_b
    if name_a.endswith("_zero") and not name_b.endswith("_zero"):
        return name_a
    if name_b.endswith("_zero") and not name_a.endswith("_zero"):
        return name_b
    return name_b


def register_duplicates(data_dict, graph_types):
    """
    Add one member of every pair of constructions with identical ORC and FRC
    on every node to DUPLICATE_GRAPHS.

    :param data_dict: loaded split
    :param graph_types: constructions to check
    :return: None
    """
    start = time.time()
    pairs = duplicate_pairs_from_values(data_dict, graph_types, "jet_id")
    register_duplicate_pairs(DUPLICATE_GRAPHS, pairs, "same curvature on every node", pick_drop)
    print(f"[dup] checked {len(graph_types)} constructions in {time.time() - start:.1f} s", flush=True)


def duplicate_free(graph_types):
    """
    :param graph_types: construction names
    :return: the names not listed in DUPLICATE_GRAPHS
    """
    kept = []
    for graph_type in graph_types:
        if graph_type.lower() not in DUPLICATE_GRAPHS:
            kept.append(graph_type)
    return kept

