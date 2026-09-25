from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed


NODE_SPECS = [
    ('PV_nodes', 'pv_orc', 'orc', 'union'),
    ('PV_nodes', 'pv_orc_mean', 'orc', 'per_jet_mean'),
    ('PV_nodes', 'pv_frc', 'frc', 'union'),
    ('PV_nodes', 'pv_frc_mean', 'frc', 'per_jet_mean'),
    ('SV_nodes', 'sv_orc', 'orc', 'union'),
    ('SV_nodes', 'sv_orc_mean', 'orc', 'per_jet_mean'),
    ('SV_nodes', 'sv_frc', 'frc', 'union'),
    ('SV_nodes', 'sv_frc_mean', 'frc', 'per_jet_mean'),
]
EDGE_SPECS = [
    ('PVPV_edges', 'pvpv_orc', 'orc', 'union'),
    ('PVPV_edges', 'pvpv_orc_mean', 'orc', 'per_jet_mean'),
    ('PVPV_edges', 'pvpv_frc', 'frc', 'union'),
    ('PVPV_edges', 'pvpv_frc_mean', 'frc', 'per_jet_mean'),
    ('SVSV_edges', 'svsv_orc', 'orc', 'union'),
    ('SVSV_edges', 'svsv_orc_mean', 'orc', 'per_jet_mean'),
    ('SVSV_edges', 'svsv_frc', 'frc', 'union'),
    ('SVSV_edges', 'svsv_frc_mean', 'frc', 'per_jet_mean'),
    ('PVSV_edges', 'pvsv_orc', 'orc', 'union'),
    ('PVSV_edges', 'pvsv_orc_mean', 'orc', 'per_jet_mean'),
    ('PVSV_edges', 'pvsv_frc', 'frc', 'union'),
    ('PVSV_edges', 'pvsv_frc_mean', 'frc', 'per_jet_mean'),
]

_ALL_KEYS = ['pv_orc', 'sv_orc', 'pv_frc', 'sv_frc',
            'pv_orc_mean', 'sv_orc_mean', 'pv_frc_mean', 'sv_frc_mean',
            'pvpv_orc', 'svsv_orc', 'pvsv_orc', 'pvpv_frc', 'svsv_frc', 'pvsv_frc',
            'pvpv_orc_mean', 'svsv_orc_mean', 'pvsv_orc_mean',
            'pvpv_frc_mean', 'svsv_frc_mean', 'pvsv_frc_mean']

# joblib settings for every pool in this module. max_nbytes="1M": any numpy
# array of at least 1 MB in a task's arguments is written once to shared memory
# (/dev/shm, or JOBLIB_TEMP_FOLDER) and every worker maps it read-only, instead
# of each task shipping a pickled copy. Python lists and DataFrames are NOT
# memmapped, which is why everything handed to a worker below is an ndarray.
# pre_dispatch="n_jobs": queue n_jobs tasks at a time, not 2*n_jobs.
_POOL_KW = dict(pre_dispatch="n_jobs", max_nbytes="1M", mmap_mode="r")


def _per_jet_mean(jet_id, orc, frc):
    '''
    One mean per jet for both curvature columns.

    :param jet_id: jet id per value
    :param orc: ORC value per entry
    :param frc: FRC value per entry
    :return: tuple (orc_means, frc_means) as float32 arrays, in jet_id order
    '''
    if len(jet_id) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    df = pd.DataFrame({"jet_id": jet_id, "orc": orc, "frc": frc})
    m = df.groupby("jet_id", sort=True)[["orc", "frc"]].mean()
    orc_f32 = np.ascontiguousarray(m["orc"].to_numpy(), dtype=np.float32)
    frc_f32 = np.ascontiguousarray(m["frc"].to_numpy(), dtype=np.float32)
    return orc_f32, frc_f32


def _endpoint_vertices(nodes, edges):
    '''
    Gets truth_vertex_idx for edge src/dst nodes without copying the edge table,
    replacing the old (jet_id, node_id) left-merge. Missing nodes map to -1
    (matching former NaN "not PV" behavior).
    Falls back to the merge if keys aren't unique or indices are negative

    :param nodes: node table with jet_id, node_id, truth_vertex_idx
    :param edges: edge table with jet_id, src, dst
    :return: tuple (src_vertex, dst_vertex), int arrays aligned with edges
    '''
    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    src = edges["src"].to_numpy().astype(np.int64)
    dst = edges["dst"].to_numpy().astype(np.int64)
    ok = len(node_id) > 0 and node_id.min() >= 0 and src.min() >= 0 and dst.min() >= 0
    if ok:
        mult = int(max(node_id.max(), src.max(), dst.max())) + 1
        key = nodes["jet_id"].to_numpy().astype(np.int64) * mult + node_id
        idx = pd.Index(key)
        ok = idx.is_unique
    if not ok:
        print("[warn] _endpoint_vertices: (jet_id, node_id) not unique or negative; using merge")
        lookup = nodes[["jet_id", "node_id", "truth_vertex_idx"]]
        e = edges[["jet_id", "src", "dst"]].merge(
            lookup.rename(columns={"node_id": "src", "truth_vertex_idx": "src_vertex"}),
            on=["jet_id", "src"], how="left",
        ).merge(
            lookup.rename(columns={"node_id": "dst", "truth_vertex_idx": "dst_vertex"}),
            on=["jet_id", "dst"], how="left",
        )
        sv = e["src_vertex"].fillna(-1).to_numpy().astype(np.int64)
        dv = e["dst_vertex"].fillna(-1).to_numpy().astype(np.int64)
        return sv, dv

    # sentinel -1 appended at the end, so a miss (-1 from get_indexer) maps to it
    tvi = np.append(nodes["truth_vertex_idx"].to_numpy().astype(np.int64), -1)
    miss = len(tvi) - 1
    ejid = edges["jet_id"].to_numpy().astype(np.int64)
    out = []
    for endpoint in (src, dst):
        pos = idx.get_indexer(ejid * mult + endpoint)
        pos[pos < 0] = miss
        out.append(tvi[pos])
    return out[0], out[1]

def classify_edges(src_vertex, dst_vertex):
    """
    Edge type from the truth_vertex_idx of the two endpoints, using the same
    rule as the nodes: 0 = PV-PV, 1 = SV-SV, 2 = PV-SV, and -1 when either
    endpoint is unmatched (truth_vertex_idx < 0).

    :param src_vertex: truth_vertex_idx of the source endpoint of every edge
    :param dst_vertex: truth_vertex_idx of the destination endpoint of every edge
    :return: int8 array of edge types
    """
    src_vertex = np.asarray(src_vertex)
    dst_vertex = np.asarray(dst_vertex)
    matched = (src_vertex >= 0) & (dst_vertex >= 0)
    src_pv = src_vertex == 0
    dst_pv = dst_vertex == 0
    edge_type = np.full(len(src_vertex), -1, dtype=np.int8)
    edge_type[matched & src_pv & dst_pv] = 0
    edge_type[matched & ~src_pv & ~dst_pv] = 1
    edge_type[matched & (src_pv != dst_pv)] = 2
    return edge_type

def _build_distributions(data_dict, graph_types, flavors):
    '''
    Per-(graph, flavor) curvature distributions, used by every function in
    this module and returned to the caller as `data`.

    Works on 1-D column views and boolean masks only: no filtered copies of
    the node/edge tables and no merged edge table are created. Each cell is a
    fresh float32 array (a boolean-indexed copy of one column), so the total
    is one float32 per curvature value and nothing keeps the frames alive.

    :param data_dict: dict of graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    :param graph_types: list of graph constructions to build distributions for
    :param flavors: list of flavor codes to keep
    :return: nested dict data[graph_type][flavor][key] -> float32 ndarray
    '''
    data = {}
    for graph_type in graph_types:
        data[graph_type] = {}
        for fl in flavors:
            data[graph_type][fl] = {}
            for k in _ALL_KEYS:
                data[graph_type][fl][k] = np.empty(0, dtype=np.float32)

    for graph_type in graph_types:
        if graph_type not in data_dict:
            print(f"WARNING: '{graph_type}' not in loaded; skipping.")
            continue
        nodes = data_dict[graph_type]["nodes"]
        edges = data_dict[graph_type]["edges"]
        if nodes.empty:
            continue

        # nodes: column views, no frame copies
        fl = nodes["flavor"].to_numpy()
        tvi = nodes["truth_vertex_idx"].to_numpy()
        jid = nodes["jet_id"].to_numpy()
        orc = nodes["orc_curvature"].to_numpy()
        frc = nodes["frc_curvature"].to_numpy()
        is_pv = tvi == 0
        is_sv = tvi > 0  # unmatched (< 0) are in neither

        for flavor in flavors:
            mf = fl == flavor
            d = data[graph_type][flavor]
            for tag, mask in (("pv", is_pv & mf), ("sv", is_sv & mf)):
                if not mask.any():
                    continue

                o = np.ascontiguousarray(orc[mask], dtype=np.float32)
                f = np.ascontiguousarray(frc[mask], dtype=np.float32)
                d[f"{tag}_orc"] = o
                d[f"{tag}_frc"] = f
                d[f"{tag}_orc_mean"], d[f"{tag}_frc_mean"] = _per_jet_mean(jid[mask], o, f)

        if edges.empty:
            continue

        # edges: endpoint vertex types by lookup, then the same three classes
        # as before: pvpv, svsv (both endpoints not PV, incl. unmatched), pvsv
        src_v, dst_v = _endpoint_vertices(nodes, edges)
        s_pv = src_v == 0
        d_pv = dst_v == 0
        etype = {"pvpv": s_pv & d_pv, "svsv": ~s_pv & ~d_pv, "pvsv": s_pv ^ d_pv}

        efl = edges["flavor"].to_numpy()
        ejid = edges["jet_id"].to_numpy()
        eorc = edges["orc_edge_curvature"].to_numpy()
        efrc = edges["frc_edge_curvature"].to_numpy()

        for flavor in flavors:
            mf = efl == flavor
            d = data[graph_type][flavor]
            for tag, mt in etype.items():
                mask = mt & mf
                if not mask.any():
                    continue

                o = np.ascontiguousarray(eorc[mask], dtype=np.float32)
                f = np.ascontiguousarray(efrc[mask], dtype=np.float32)
                d[f"{tag}_orc"] = o
                d[f"{tag}_frc"] = f
                d[f"{tag}_orc_mean"], d[f"{tag}_frc_mean"] = _per_jet_mean(ejid[mask], o, f)

    return data


def _rows_for_pair(base, a, b):
    '''
    Computes every NODE_SPECS/EDGE_SPECS row for one pair of distribution
    dicts. Doing all the specs together in one call instead of one call per
    wasserstein_distance keeps the number of tasks handed to workers down.

    The arrays arrive as read-only float32 memmaps; wasserstein_distance makes
    its own float64 working copy, so each worker holds only the pair it is on.

    :param base: dict of fixed fields to stick onto every row (graph names,
        flavor names, etc.)
    :param a: distribution dict for side A
    :param b: distribution dict for side B
    :return: tuple (node_rows, edge_rows), both lists of dicts
    '''
    node_rows = []
    edge_rows = []

    for label, key, kind, agg in NODE_SPECS:
        av = a[key]
        bv = b[key]
        if len(av) > 0 and len(bv) > 0:
            row = dict(base)
            row['Comparison'] = label
            row['Curvature'] = kind
            row['Aggregation'] = agg
            row['n_A'] = len(av)
            row['n_B'] = len(bv)
            row['W_Distance'] = wasserstein_distance(av, bv)
            node_rows.append(row)

    for label, key, kind, agg in EDGE_SPECS:
        av = a[key]
        bv = b[key]
        if len(av) > 0 and len(bv) > 0:
            row = dict(base)
            row['Comparison'] = label
            row['Curvature'] = kind
            row['Aggregation'] = agg
            row['n_A'] = len(av)
            row['n_B'] = len(bv)
            row['W_Distance'] = wasserstein_distance(av, bv)
            edge_rows.append(row)

    return node_rows, edge_rows


def analyze_across_construction_from_hdf5_parallel(data_dict, graph_types, flavors,
                                                    n_jobs=-1, verbose=5):
    '''
    Parallel version of analyze_across_construction_from_hdf5.

    :param data_dict: dict of graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    :param graph_types: list of graph constructions to compare
    :param flavors: list of flavor codes to compare
    :param n_jobs: workers to use, -1 means use all cores
    :param verbose: joblib verbosity level
    :return: tuple (df_nodes, df_edges, df_flavor_nodes, df_flavor_edges, data)
    '''
    data = _build_distributions(data_dict, graph_types, flavors)

    # across constructions, same flavor
    graph_tasks = []
    for graphtype_a, graphtype_b in combinations(graph_types, 2):
        if graphtype_a not in data or graphtype_b not in data:
            continue
        for flavor in flavors:
            base = {
                'Graph_A': graphtype_a.upper(),
                'Graph_B': graphtype_b.upper(),
                'Flavor': f'Flavor {flavor}',
            }
            graph_tasks.append(delayed(_rows_for_pair)(base, data[graphtype_a][flavor], data[graphtype_b][flavor]))

    graph_results = Parallel(n_jobs=n_jobs, verbose=verbose, **_POOL_KW)(graph_tasks)

    node_rows = []
    edge_rows = []
    for nr, er in graph_results:
        node_rows.extend(nr)
        edge_rows.extend(er)

    # across flavors, same construction
    flavor_tasks = []
    for graph_type in graph_types:
        if graph_type not in data:
            continue
        for flavor_a, flavor_b in combinations(flavors, 2):
            base = {
                'Graph': graph_type.upper(),
                'Flavor_A': f'Flavor {flavor_a}',
                'Flavor_B': f'Flavor {flavor_b}',
            }
            flavor_tasks.append(delayed(_rows_for_pair)(base, data[graph_type][flavor_a], data[graph_type][flavor_b]))

    flavor_results = Parallel(n_jobs=n_jobs, verbose=verbose, **_POOL_KW)(flavor_tasks)

    flavor_node_rows = []
    flavor_edge_rows = []
    for nr, er in flavor_results:
        flavor_node_rows.extend(nr)
        flavor_edge_rows.extend(er)

    return (pd.DataFrame(node_rows), pd.DataFrame(edge_rows),
            pd.DataFrame(flavor_node_rows), pd.DataFrame(flavor_edge_rows), data)


def compute_cross_flavor_graph_w1_parallel(data, graph_types, flavors, n_jobs=-1, verbose=5):
    '''
    Parallel version of compute_cross_flavor_graph_w1. This one's the biggest
    of the bunch since it's O(flavor_pairs * graph_types^2).

    :param data: distribution dict already built by analyze_across_construction_from_hdf5
        (or its parallel version)
    :param graph_types: list of graph constructions to compare
    :param flavors: list of flavor codes to compare
    :param n_jobs: workers to use, -1 means use all cores
    :param verbose: joblib verbosity level
    :return: tuple (df_nodes, df_edges)
    '''
    tasks = []
    for flavor_a, flavor_b in combinations(flavors, 2):
        for graph_a in graph_types:
            for graph_b in graph_types:
                base = {
                    'Graph_A': graph_a.upper(),
                    'Graph_B': graph_b.upper(),
                    'Flavor_A': f'Flavor {flavor_a}',
                    'Flavor_B': f'Flavor {flavor_b}',
                }
                tasks.append(delayed(_rows_for_pair)(base, data[graph_a][flavor_a], data[graph_b][flavor_b]))

    results = Parallel(n_jobs=n_jobs, verbose=verbose, **_POOL_KW)(tasks)

    node_rows = []
    edge_rows = []
    for nr, er in results:
        node_rows.extend(nr)
        edge_rows.extend(er)

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def _pv_sv_node_row(graph_type, flavor, pv_orc, sv_orc, pv_frc, sv_frc):
    '''
    Builds the PV-vs-SV row pair for one graph type and flavor.

    :param graph_type: name of the graph construction
    :param flavor: flavor code
    :param pv_orc: ORC values of the PV nodes of this (graph, flavor)
    :param sv_orc: ORC values of the SV nodes
    :param pv_frc: FRC values of the PV nodes
    :param sv_frc: FRC values of the SV nodes
    :return: tuple (row_orc, row_frc), or (None, None) if either side is empty
    '''
    num_pv_nodes = len(pv_orc)
    num_sv_nodes = len(sv_orc)

    if num_pv_nodes == 0 or num_sv_nodes == 0:
        return None, None

    base_row = {
        "Graph": graph_type.upper(),
        "Flavor": f"Flavor {flavor}",
        "num_pv_nodes": num_pv_nodes,
        "num_sv_nodes": num_sv_nodes,
    }

    row_orc = dict(base_row)
    row_orc["W_Distance_orc"] = wasserstein_distance(pv_orc, sv_orc)

    row_frc = dict(base_row)
    row_frc["W_Distance_frc"] = wasserstein_distance(pv_frc, sv_frc)

    return row_orc, row_frc


def analyze_flavor_separation_from_hdf5_parallel(data_dict, flavors, n_jobs=-1, verbose=0, data=None):
    '''
    Parallel version of analyze_flavor_separation_from_hdf5: W1 between the
    PV-node and SV-node curvature distributions of the same flavor, per
    construction.

    Works from the `data` dict of _build_distributions. Pass the one returned
    by analyze_across_construction_from_hdf5_parallel to avoid building it
    twice; otherwise it is built here from data_dict.

    :param data_dict: dict of graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    :param flavors: list of flavor codes to check
    :param n_jobs: workers to use, -1 means use all cores
    :param verbose: joblib verbosity level
    :param data: optional prebuilt distributions (see above)
    :return: tuple (df_orc, df_frc)
    '''
    if data is None:
        data = _build_distributions(data_dict, list(data_dict), flavors)

    tasks = []
    for graph_type in data:
        for flavor in flavors:
            d = data[graph_type][flavor]
            tasks.append(delayed(_pv_sv_node_row)(
                graph_type, flavor, d["pv_orc"], d["sv_orc"], d["pv_frc"], d["sv_frc"]))

    results = Parallel(n_jobs=n_jobs, verbose=verbose, **_POOL_KW)(tasks)

    stats_orc = []
    stats_frc = []
    for row_orc, row_frc in results:
        if row_orc is not None:
            stats_orc.append(row_orc)
        if row_frc is not None:
            stats_frc.append(row_frc)

    return pd.DataFrame(stats_orc), pd.DataFrame(stats_frc)


def _edge_type_rows(graph_type, flavor, pvpv_orc, svsv_orc, pvsv_orc, pvpv_frc, svsv_frc, pvsv_frc):
    '''
    Builds the three pairwise edge-type comparison rows (PVPV vs SVSV, PVPV
    vs PVSV, SVSV vs PVSV) for one graph type and flavor.

    :param graph_type: name of the graph construction
    :param flavor: flavor code
    :param pvpv_orc: ORC values of the PV-PV edges of this (graph, flavor)
    :param svsv_orc: ORC values of the SV-SV edges
    :param pvsv_orc: ORC values of the PV-SV edges
    :param pvpv_frc: FRC values of the PV-PV edges
    :param svsv_frc: FRC values of the SV-SV edges
    :param pvsv_frc: FRC values of the PV-SV edges
    :return: tuple (orc_rows, frc_rows), both lists of dicts
    '''
    base_row = {
        "Graph": graph_type.upper(),
        "Flavor": f"Flavor {flavor}",
        "num_pvpv_edges": len(pvpv_orc),
        "num_svsv_edges": len(svsv_orc),
        "num_pvsv_edges": len(pvsv_orc),
    }

    orc_rows = []
    frc_rows = []
    pairs = [("PVPV_vs_SVSV", pvpv_orc, svsv_orc, pvpv_frc, svsv_frc),
             ("PVPV_vs_PVSV", pvpv_orc, pvsv_orc, pvpv_frc, pvsv_frc),
             ("SVSV_vs_PVSV", svsv_orc, pvsv_orc, svsv_frc, pvsv_frc)]

    for comp, xo, yo, xf, yf in pairs:
        if len(xo) > 0 and len(yo) > 0:
            row_orc = dict(base_row)
            row_orc["comparison"] = comp
            row_orc["W_Distance_orc"] = wasserstein_distance(xo, yo)
            orc_rows.append(row_orc)

            row_frc = dict(base_row)
            row_frc["comparison"] = comp
            row_frc["W_Distance_frc"] = wasserstein_distance(xf, yf)
            frc_rows.append(row_frc)

    return orc_rows, frc_rows


def analyze_flavor_separation_edge_from_hdf5_parallel(data_dict, flavors, n_jobs=-1, verbose=0, data=None):
    '''
    Parallel version of analyze_flavor_separation_edge_from_hdf5: W1 between
    the curvature distributions of the three edge types, within a flavor, per
    construction.

    Works from the `data` dict of _build_distributions, like
    analyze_flavor_separation_from_hdf5_parallel. Constructions with no edges
    have empty edge distributions and produce no rows, as before.

    :param data_dict: dict of graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    :param flavors: list of flavor codes to check
    :param n_jobs: workers to use, -1 means use all cores
    :param verbose: joblib verbosity level
    :param data: optional prebuilt distributions (see above)
    :return: tuple (df_orc, df_frc)
    '''
    if data is None:
        data = _build_distributions(data_dict, list(data_dict), flavors)

    tasks = []
    for graph_type in data:
        for flavor in flavors:
            d = data[graph_type][flavor]
            if len(d["pvpv_orc"]) == 0 and len(d["svsv_orc"]) == 0 and len(d["pvsv_orc"]) == 0:
                continue
            tasks.append(delayed(_edge_type_rows)(
                graph_type, flavor,
                d["pvpv_orc"], d["svsv_orc"], d["pvsv_orc"],
                d["pvpv_frc"], d["svsv_frc"], d["pvsv_frc"]))

    results = Parallel(n_jobs=n_jobs, verbose=verbose, **_POOL_KW)(tasks)

    stats_orc = []
    stats_frc = []
    for orc_rows, frc_rows in results:
        stats_orc.extend(orc_rows)
        stats_frc.extend(frc_rows)

    return pd.DataFrame(stats_orc), pd.DataFrame(stats_frc)