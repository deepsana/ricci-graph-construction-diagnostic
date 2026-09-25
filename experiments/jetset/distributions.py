import gc
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.jetset.config import AGGREGATIONS, ALL_UNION, CURVATURES, DIST_KEYS, EDGE_SPECS, EDGE_TAGS, JOINT_COND, MIN_JETS_PER_BIN,MIN_VALUES_PER_SIDE, NODE_SPECS, PROGRESS_EVERY, SPEC_KEY, WITHIN_COMPARISONS
from utils.binning import bin_labels_of, bin_series, joint_bin_labels, joint_bin_series
from utils.memory import rss_gb
from utils.wassertein1_parallel import _endpoint_vertices, classify_edges

EMPTY_ARRAY = np.empty(0, dtype=np.float32)
MIN_VALUES_PER_W1 = max(1, MIN_VALUES_PER_SIDE)
NODE_TAGS = {0: "pv", 1: "sv"}


def flavor_by_jet(nodes):
    """
    Flavor code of every jet, as a Series indexed by jet_id.
    """
    jets = pd.DataFrame({"jet_id": nodes["jet_id"].to_numpy(),"flavor": nodes["flavor"].to_numpy() }).drop_duplicates("jet_id")
    return jets.set_index("jet_id")["flavor"]


def jets_per_flavor(nodes, flavors):
    """{flavor: number of jets of that flavor in the sample}."""
    counts = flavor_by_jet(nodes).value_counts()
    return {int(flavor): int(counts.get(flavor, 0)) for flavor in flavors}


def joint_label(mult_label, sv_label):
    """Cell label of the joint (n, n_sv) grid, e.g. "n=6-9|sv=1-2"."""
    return f"n={mult_label}|sv={sv_label}"


def cond_bin_labels(cond_var, bins):
    """
    Every bin label of a conditioning variable, in plotting order.

    :param cond_var: "none", "n", "n_sv", "n_pv", "sv_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges; for JOINT_COND the pair (MULT_BINS, SV_BINS)
    :return: list of labels, multiplicity-major for the joint grid
    """
    if cond_var != JOINT_COND:
        return bin_labels_of(bins)
    mult_bins, sv_bins = bins
    return joint_bin_labels(mult_bins, sv_bins, joint_label)


def per_jet_counts(nodes):
    """
    Track counts per jet. truth_vertex_idx is 0 for the primary vertex (PV),
    > 0 for a displaced vertex (SV) and < 0 for an unmatched track, which
    counts toward n but toward neither n_sv nor n_pv.

    :return: tuple (n, n_sv, n_pv, sv_frac) of Series indexed by jet_id
    """
    vertex_idx = nodes["truth_vertex_idx"].to_numpy()
    flags = pd.DataFrame({
        "jet_id": nodes["jet_id"].to_numpy(),
        "is_sv": vertex_idx > 0,
        "is_pv": vertex_idx == 0,
    })
    grouped = flags.groupby("jet_id", sort=True)
    n = grouped["is_sv"].size()
    n_sv = grouped["is_sv"].sum()
    n_pv = grouped["is_pv"].sum()
    return n, n_sv, n_pv, n_sv / n


def per_jet_bin(nodes, cond_var, bins):
    """
    Bin label of every jet for one conditioning variable. For JOINT_COND a jet
    gets a label only if it falls in a bin of both variables.

    :param cond_var: "none", "n", "n_sv", "n_pv", "sv_frac" or JOINT_COND
    :return: object Series indexed by jet_id, None for jets outside every bin
    """
    n, n_sv, n_pv, sv_frac = per_jet_counts(nodes)

    if cond_var == JOINT_COND:
        mult_bins, sv_bins = bins
        return joint_bin_series(n, mult_bins, n_sv, sv_bins, joint_label)

    # "none" bins on n with a single catch-all bin.
    counts = {"n": n, "n_sv": n_sv, "n_pv": n_pv, "sv_frac": sv_frac, "none": n}
    if cond_var not in counts:
        raise ValueError(f"cond_var must be one of {list(counts)} or '{JOINT_COND}', got '{cond_var}'")
    return bin_series(counts[cond_var], bins)


def fill_cell(cell, tag, rows, means_only=False, keep_ids=False):
    """
    Store the distributions of one (subset, cell) in place: every ORC and FRC
    value, and one mean per jet of each. With keep_ids, the jet id of every
    value is stored next to it (<tag>_ids, <tag>_ids_mean) so the bootstrap
    can resample by jet.

    :param cell: data_binned[graph][flavor][bin]
    :param tag: subset tag, "pv", "sv", "pvpv", "svsv", "pvsv", "all" or "alle"
    :param rows: rows of this subset and cell with jet_id, orc and frc columns
    :param means_only: store only the per-jet means
    """
    if not means_only:
        cell[f"{tag}_orc"] = rows["orc"].to_numpy(dtype=np.float32, copy=True)
        cell[f"{tag}_frc"] = rows["frc"].to_numpy(dtype=np.float32, copy=True)

    jet_means = rows.groupby("jet_id", sort=True)[["orc", "frc"]].mean()
    cell[f"{tag}_orc_mean"] = jet_means["orc"].to_numpy(dtype=np.float32, copy=True)
    cell[f"{tag}_frc_mean"] = jet_means["frc"].to_numpy(dtype=np.float32, copy=True)

    if keep_ids:
        jets = rows["jet_id"].to_numpy(copy=True)
        if means_only:
            jets = np.empty(0, dtype=jets.dtype)
        cell[f"{tag}_ids"] = jets
        cell[f"{tag}_ids_mean"] = jet_means.index.to_numpy()


def empty_binned(graph_types, flavors, bin_labels):
    """
    data_binned with an empty array under every key of every cell
    """
    return {graph_type:
                {flavor:
                     {label: dict.fromkeys(DIST_KEYS, EMPTY_ARRAY)  for label in bin_labels} for flavor in flavors}
            for graph_type in graph_types}


def count_cell_jets(nodes, jet_bin, flavors, bin_labels, cond_var, min_jets):
    """
    Jets per (flavor, bin) cell, printed per flavor with thin cells marked.

    :param jet_bin: output of per_jet_bin
    :return: {(flavor, bin label): number of jets}
    """
    jet_flavor = flavor_by_jet(nodes).reindex(jet_bin.index)
    outside = jet_bin.isna().to_numpy()
    cell_jets = {}
    for flavor in flavors:
        is_flavor = (jet_flavor == flavor).to_numpy()
        parts = []
        for label in bin_labels:
            n_jets = int(((jet_bin == label).to_numpy() & is_flavor).sum())
            cell_jets[(flavor, label)] = n_jets
            skip = "" if n_jets >= min_jets else "(skip)"
            parts.append(f"{label}: {n_jets}{skip}")
        n_outside = int((outside & is_flavor).sum())
        print(f"[flavor {flavor}] jets per {cond_var}-bin  {',  '.join(parts)},  outside bins: {n_outside}", flush=True)
    return cell_jets


def fill_cells(graph_cells, frame, class_col, class_tags, pooled_tag, bin_labels, cell_ok, keep_ids):
    """
    Fill the cells of one construction from a node or edge frame: first the
    pooled subset of every row (all tracks or all edges), then one subset
    per class. Rows with a negative class (unmatched) enter only the pool.

    :param graph_cells: data_binned[graph_type]
    :param frame: rows with jet_id, flavor, bin_code, class_col, orc and frc,
        restricted to binned jets of the scored flavors
    :param class_tags: {class code: subset tag}
    :param pooled_tag: tag of the pooled subset, "all" or "alle"
    :param cell_ok: {(flavor, bin label): whether the cell has enough jets}
    """
    for (flavor, bin_code), rows in frame.groupby(["flavor", "bin_code"], sort=False):
        label = bin_labels[int(bin_code)]
        if cell_ok[(int(flavor), label)]:
            fill_cell(graph_cells[int(flavor)][label], pooled_tag, rows, means_only=not ALL_UNION, keep_ids=keep_ids)

    matched = frame[frame[class_col] >= 0]
    for (flavor, bin_code, row_class), rows in matched.groupby(["flavor", "bin_code", class_col], sort=False):
        label = bin_labels[int(bin_code)]
        if cell_ok[(int(flavor), label)]:
            fill_cell(graph_cells[int(flavor)][label], class_tags[int(row_class)], rows, keep_ids=keep_ids)


def binned_frame(table, class_col, row_class, orc_col, frc_col, jet_bin_code, flavors):
    """
    Compact frame of the rows of a node or edge table that fall in a bin and
    belong to a scored flavor.

    :param row_class: class code of every row (node: PV/SV, edge: PV-PV/SV-SV/PV-SV)
    :param jet_bin_code: bin index of every jet, -1 outside every bin
    """
    frame = pd.DataFrame({
        "jet_id": table["jet_id"].to_numpy(),
        "flavor": table["flavor"].to_numpy(),
        "bin_code": table["jet_id"].map(jet_bin_code).to_numpy(),
        class_col: row_class,
        "orc": table[orc_col].to_numpy(dtype=np.float32),
        "frc": table[frc_col].to_numpy(dtype=np.float32),
    })
    keep = (frame["bin_code"] >= 0) & frame["flavor"].isin(flavors)
    return frame[keep]


def node_classes(nodes):
    """0 for a PV track, 1 for an SV track, -1 for an unmatched track."""
    vertex_idx = nodes["truth_vertex_idx"].to_numpy()
    node_class = np.full(len(vertex_idx), -1, dtype=np.int8)
    node_class[vertex_idx == 0] = 0
    node_class[vertex_idx > 0] = 1
    return node_class


def collect_binned_distributions(data_dict, graph_types, flavors, cond_var, bins,  min_jets=MIN_JETS_PER_BIN, keep_ids=False):
    """
    Split every curvature distribution by flavor and by bin of the
    conditioning variable.

    All constructions share the same jets, so the jet-to-bin map and the cell
    counts come from the first construction. An edge's class comes from the
    truth_vertex_idx of its two endpoints, with the same rule as the nodes, so
    unmatched tracks never enter an edge subset. Cells with fewer than
    min_jets jets are left empty.

    :param data_dict: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    :param flavors: integer flavor codes to keep
    :param cond_var: "none", "n", "n_sv", "n_pv", "sv_frac" or JOINT_COND
    :param bins: inclusive (low, high) ranges; for JOINT_COND the pair (MULT_BINS, SV_BINS)
    :param min_jets: minimum number of jets for a (flavor, bin) cell to be used
    :param keep_ids: store the jet id of every value next to it (see fill_cell)
    :return: tuple (data_binned, jets_per_bin DataFrame)
    """
    bin_labels = cond_bin_labels(cond_var, bins)
    label_to_code = {label: code for code, label in enumerate(bin_labels)}
    data_binned = empty_binned(graph_types, flavors, bin_labels)

    jets_per_bin_rows = []
    jet_bin_code = None
    reference_n_nodes = None
    cell_jets = {}

    for graph_type in graph_types:
        if graph_type not in data_dict:
            print(f"WARNING: '{graph_type}' not loaded; skipping.", flush=True)
            continue
        nodes = data_dict[graph_type]["nodes"]
        edges = data_dict[graph_type]["edges"]
        if nodes.empty:
            print(f"WARNING: '{graph_type}' has an empty node table; skipping.", flush=True)
            continue

        if jet_bin_code is None:
            jet_bin = per_jet_bin(nodes, cond_var, bins)
            jet_bin_code = jet_bin.map(label_to_code).fillna(-1).astype(np.int32)
            reference_n_nodes = len(nodes)
            cell_jets = count_cell_jets(nodes, jet_bin, flavors, bin_labels, cond_var, min_jets)
        elif len(nodes) != reference_n_nodes:
            print(f"WARNING: '{graph_type}' has {len(nodes)} node rows against {reference_n_nodes} for the first construction; the constructions "
                  f"do not share the point cloud, so the jet-to-bin map of the first construction may not apply to it.", flush=True)
        cell_ok = {cell: n_jets >= min_jets for cell, n_jets in cell_jets.items()}

        for flavor in flavors:
            for label in bin_labels:
                jets_per_bin_rows.append({
                    "Graph": graph_type.upper(),
                    "Flavor": f"Flavor {flavor}",
                    "CondVar": cond_var,
                    "Bin": label,
                    "n_jets": cell_jets[(flavor, label)],
                    "used": cell_ok[(flavor, label)],
                })

        node_frame = binned_frame(nodes, "cls", node_classes(nodes), "orc_curvature","frc_curvature", jet_bin_code, flavors)
        fill_cells(data_binned[graph_type], node_frame, "cls", NODE_TAGS, "all", bin_labels,
                   cell_ok, keep_ids)
        del node_frame
        gc.collect()

        if edges.empty:
            continue

        src_vertex, dst_vertex = _endpoint_vertices(nodes, edges)
        edge_type = classify_edges(src_vertex, dst_vertex)
        n_unmatched = int((edge_type < 0).sum())
        del src_vertex, dst_vertex

        edge_frame = binned_frame(edges, "etype", edge_type, "orc_edge_curvature", "frc_edge_curvature", jet_bin_code, flavors)
        fill_cells(data_binned[graph_type], edge_frame, "etype", EDGE_TAGS, "alle", bin_labels,
                   cell_ok, keep_ids)
        del edge_frame
        gc.collect()

        print(f"[{graph_type:34s}] binned; {n_unmatched} of {len(edges)} edges dropped (unmatched endpoint); RSS {rss_gb():.1f} GB", flush=True)

    return data_binned, pd.DataFrame(jets_per_bin_rows)


def w1_row(base, comparison, curvature, aggregation, values_a, values_b):
    """
    One W1 table row for a pair of distributions.

    :param base: fixed fields copied into the row
    :return: dict with the base fields, value counts, means, spreads and W_Distance
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    row = dict(base)
    row["Comparison"] = comparison
    row["Curvature"] = curvature
    row["Aggregation"] = aggregation
    row["n_A"] = len(values_a)
    row["n_B"] = len(values_b)
    row["mean_A"] = float(values_a.mean())
    row["std_A"] = float(values_a.std())
    row["mean_B"] = float(values_b.mean())
    row["std_B"] = float(values_b.std())
    row["W_Distance"] = wasserstein_distance(values_a, values_b)
    return row


def spec_rows(base, specs, dist_a, dist_b):
    """W1 rows of every spec whose two sides have at least MIN_VALUES_PER_W1 values."""
    rows = []
    for comparison, key, curvature, aggregation in specs:
        if len(dist_a[key]) >= MIN_VALUES_PER_W1 and len(dist_b[key]) >= MIN_VALUES_PER_W1:
            rows.append(w1_row(base, comparison, curvature, aggregation, dist_a[key], dist_b[key]))
    return rows


def within_rows(base, comparison, dist, min_values):
    """
    W1 rows of one within-flavor subset: PV against SV nodes of one flavor,
    one row per curvature and aggregation. Sides with fewer than min_values
    values are skipped.

    :param base: fixed fields copied into every row, with Flavor_A == Flavor_B
    :param comparison: a key of WITHIN_COMPARISONS
    :param dist: distribution dict of that flavor
    """
    rows = []
    for curvature in CURVATURES:
        for aggregation in AGGREGATIONS:
            values_a = dist[SPEC_KEY[("PV_nodes", curvature, aggregation)]]
            values_b = dist[SPEC_KEY[("SV_nodes", curvature, aggregation)]]
            if len(values_a) >= min_values and len(values_b) >= min_values:
                rows.append(w1_row(base, comparison, curvature, aggregation, values_a, values_b))
    return rows


def run_tasks(tasks, description):
    """
    Evaluate W1 tasks one after another, printing progress and an ETA.

    :param tasks: list of (base, dist_a, dist_b)
    :return: tuple (node rows, edge rows) over all tasks
    """
    node_rows = []
    edge_rows = []
    start = time.time()
    total = len(tasks)
    for done, (base, dist_a, dist_b) in enumerate(tasks, start=1):
        node_rows.extend(spec_rows(base, NODE_SPECS, dist_a, dist_b))
        edge_rows.extend(spec_rows(base, EDGE_SPECS, dist_a, dist_b))
        if done % PROGRESS_EVERY == 0 or done == total:
            elapsed = time.time() - start
            remaining = elapsed / done * (total - done)
            print(f"{description} {done}/{total}  ({elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left)", flush=True)
    return node_rows, edge_rows


def row_base(graph_type, flavor_a, flavor_b, cond_var, label):
    return {
        "Graph": graph_type.upper(),
        "Flavor_A": f"Flavor {flavor_a}",
        "Flavor_B": f"Flavor {flavor_b}",
        "CondVar": cond_var,
        "Bin": label,
    }


def compute_binned_w1_tables(data_binned, graph_types, flavors, cond_var, bins):
    """
    Binned across-flavor W1 tables of every construction, with the
    within-flavor PV-vs-SV rows added to the node table. The tasks reference
    the arrays in data_binned, so nothing is copied.

    :param data_binned: first output of collect_binned_distributions
    :param flavors: integer flavor codes; every unordered pair is compared
    :return: tuple (across-flavor nodes, across-flavor edges)
    """
    bin_labels = cond_bin_labels(cond_var, bins)

    flavor_tasks = []
    for label in bin_labels:
        for graph_type in graph_types:
            for flavor_a, flavor_b in combinations(flavors, 2):
                flavor_tasks.append((row_base(graph_type, flavor_a,
                                              flavor_b,
                                              cond_var,
                                              label),
                                     data_binned[graph_type][flavor_a][label],
                                     data_binned[graph_type][flavor_b][label]))

    print(f"  W1: {len(flavor_tasks)} flavor-pair tasks (serial)", flush=True)
    node_rows, edge_rows = run_tasks(flavor_tasks, "flavor pairs")
    del flavor_tasks

    for label in bin_labels:
        for graph_type in graph_types:
            for comparison, flavor in WITHIN_COMPARISONS.items():
                node_rows.extend(within_rows(row_base(graph_type,
                                                      flavor,
                                                      flavor,
                                                      cond_var,
                                                      label),
                                             comparison,
                                             data_binned[graph_type][flavor][label],
                                             MIN_VALUES_PER_W1))

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)
