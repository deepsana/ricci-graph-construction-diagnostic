import os
import numpy as np
import networkx as nx
import yaml
from itertools import combinations

def construct_knn_graph_edges(jet, k, coord_type):
    '''
    Builds an undirected edge list connecting each track to its k nearest
    neighbors in angular space.

    :param jet: jet dict with track angular coordinates
    :param k: number of neighbors to connect each track to
    :param coord_type: which coordinate space to measure distance in, only "angular" is handled
    :return: array of edges, shape (num_edges, 2)
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]

    num_nodes = len(deta)
    dr_squared = 0.0

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)
    k_max = min(k, num_nodes - 1)

    if coord_type == 'angular':
        dr_squared = (deta[:, np.newaxis] - deta[np.newaxis, :]) ** 2 + (dphi[:, np.newaxis] - dphi[np.newaxis, :]) ** 2

    np.fill_diagonal(dr_squared, np.inf)

    edge_index = set()
    for i in range(num_nodes):
        neighbors = np.argsort(dr_squared[i])[:k_max]
        for neighbor in neighbors:
            if neighbor == i:
                continue
            edge_index.add(tuple(sorted((i, neighbor))))

    return np.array(list(edge_index))


def construct_radius_based_graph_edges(jet, radius):
    '''
    Builds an undirected edge list where tracks are connected if their
    distance is less than radius.

    :param jet: jet dict with track angular coordinates
    :param radius: distance cutoff for connecting two tracks
    :return: list of edge tuples
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]
    pt = jet['tracks']['pt']

    num_nodes = len(deta)

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)

    dr_squared = (deta[:, np.newaxis] - deta[np.newaxis, :]) ** 2 + (dphi[:, np.newaxis] - dphi[np.newaxis, :]) ** 2
    np.fill_diagonal(dr_squared, np.inf)

    rows, cols = np.where(dr_squared < radius ** 2)

    edge_index = set()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        edge_index.add(tuple(sorted((int(i), int(j)))))

    return list(edge_index)


def construct_kt_threshold_graph_edges(jet, quantile, min_neighbors, max_neighbors, p):
    '''
    Use generalized kt distance metric to build undirected edges.
    d_ij^2 = min(pT_i^2p, pT_j^2p) * (deta^2 + dphi^2)

    :param jet: jet dict with track angular coordinates and pt
    :param quantile: quantile of pairwise distances used as the connection threshold
    :param min_neighbors: minimum neighbors to guarantee per track
    :param max_neighbors: maximum neighbors allowed per track
    :param p: exponent applied to pt in the distance metric
    :return: list of edge tuples
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]
    pt = jet['tracks']['pt']

    num_nodes = len(deta)

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)

    dr_squared = (deta[:, np.newaxis] - deta[np.newaxis, :]) ** 2 + (dphi[:, np.newaxis] - dphi[np.newaxis, :]) ** 2
    np.fill_diagonal(dr_squared, np.inf)

    pt_pow = pt ** (2 * p)
    pt_min_part = np.minimum(pt_pow[:, np.newaxis], pt_pow[np.newaxis, :])
    d_ij = pt_min_part * dr_squared

    finite_distances = d_ij[np.isfinite(d_ij)]
    if len(finite_distances) == 0:
        return np.array([]).reshape(0, 2)

    d_cut = np.quantile(finite_distances, quantile)

    edge_index = set()
    for i in range(num_nodes):
        neighbors = np.where(d_ij[i] < d_cut)[0]

        if len(neighbors) < min_neighbors:
            k_min = min(min_neighbors, num_nodes - 1)
            neighbors = np.argsort(dr_squared[i])[:k_min]

        if len(neighbors) > max_neighbors:
            neighbor_distances = d_ij[i][neighbors]
            neighbors = neighbors[np.argsort(neighbor_distances)[:max_neighbors]]

        for neighbor in neighbors:
            edge_index.add(tuple(sorted((i, neighbor))))

    return list(edge_index)


def construct_laman_kt_threshold_graph(jet, p):
    '''
    Building Laman graph using generalized kt distance metric to build
    undirected edges. d_ij^2 = min(pT_i^2p, pT_j^2p) * (deta^2 + dphi^2)

    :param jet: jet dict with track angular coordinates and pt
    :param p: exponent applied to pt in the distance metric
    :return: list of edge tuples
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]
    pt = jet['tracks']['pt']

    num_nodes = len(deta)

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)

    sort_idx = np.argsort(-pt)
    deta_sorted = deta[sort_idx]
    dphi_sorted = dphi[sort_idx]
    pt_sorted = pt[sort_idx]

    dr_sorted_squared = (deta_sorted[:, np.newaxis] - deta_sorted[np.newaxis, :])**2 + (dphi_sorted[:, np.newaxis] - dphi_sorted[np.newaxis, :])**2
    pt_pow = pt_sorted ** (2 * p)
    pt_min_part = np.minimum(pt_pow[:, np.newaxis], pt_pow[np.newaxis, :])
    d_ij = pt_min_part * dr_sorted_squared

    np.fill_diagonal(d_ij, np.inf)

    edges_sorted = set()

    if num_nodes == 2:
        edges_sorted.add((0, 1))
    else:
        edges_sorted.add((0, 1))
        edges_sorted.add((0, 2))
        edges_sorted.add((1, 2))

        added = [0, 1, 2]
        for i in range(3, num_nodes):
            dists_to_added = [(d_ij[i, j], j) for j in added]
            dists_to_added.sort()
            for _, j in dists_to_added[:2]:
                edges_sorted.add(tuple(sorted((i, j))))
            added.append(i)

    edge_index = set()
    for i, j in edges_sorted:
        orig_i = sort_idx[i]
        orig_j = sort_idx[j]
        edge_index.add(tuple(sorted((int(orig_i), int(orig_j)))))

    return list(edge_index)


def construct_unique_k_graph(jet, k, p):
    '''
    Builds a graph by seeding a clique of the hardest particles, then
    attaching each remaining track to its k nearest existing neighbors.

    :param jet: jet dict with track angular coordinates and pt
    :param k: how many existing neighbors to attach each new track to
    :param p: exponent applied to pt in the distance metric, 0 disables the pt weighting
    :return: list of edge tuples
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]
    pt = jet['tracks']['pt']

    num_nodes = len(deta)

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)

    sort_idx = np.argsort(-pt)
    deta_sorted = deta[sort_idx]
    dphi_sorted = dphi[sort_idx]
    pt_sorted = pt[sort_idx]

    dr_sorted_squared = (deta_sorted[:, np.newaxis] - deta_sorted[np.newaxis, :])** 2 + (dphi_sorted[:, np.newaxis] - dphi_sorted[np.newaxis, :]) ** 2
    if p == 0:
        d_ij = dr_sorted_squared
    else:
        pt_pow = pt_sorted ** (2 * p)
        pt_min_part = np.minimum(pt_pow[:, np.newaxis], pt_pow[np.newaxis, :])
        d_ij = pt_min_part * dr_sorted_squared

    np.fill_diagonal(d_ij, np.inf)

    edges_sorted = set()

    if num_nodes < 2:
        return []

    seed_size = min(k + 1, num_nodes)
    for i, j in combinations(range(seed_size), 2):
        edges_sorted.add((i, j))

    added = list(range(seed_size))

    for i in range(seed_size, num_nodes):
        dists_to_added = [(d_ij[i, j], j) for j in added]
        dists_to_added.sort()

        n_attach = min(k, len(added))
        for _, j in dists_to_added[:n_attach]:
            edges_sorted.add(tuple(sorted((i, j))))
        added.append(i)

    edge_index = set()
    for i, j in edges_sorted:
        orig_i = sort_idx[i]
        orig_j = sort_idx[j]
        edge_index.add(tuple(sorted((int(orig_i), int(orig_j)))))

    return list(edge_index)


def construct_fully_connected_edges(jet):
    '''
    Connects every track to every other track.

    :param jet: jet dict with track angular coordinates and pt
    :return: set of edge tuples (or an empty (0, 2) array if there is at most one track)
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]
    pt = jet['tracks']['pt']

    num_nodes = len(deta)

    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)

    edge_index = set()
    for i, j in combinations(range(num_nodes), 2):
        edge_index.add(tuple(sorted((int(i), int(j)))))

    return edge_index


FAMILIES = {
    'knn': construct_knn_graph_edges,
    'radius': construct_radius_based_graph_edges,
    'kt_threshold': construct_kt_threshold_graph_edges,
    'laman': construct_laman_kt_threshold_graph,
    'unique_k': construct_unique_k_graph,
    'fully_connected': construct_fully_connected_edges,
}

YAML_PATH = "configs/jetset_graph_configs.yaml"


def load_graph_configs(path=YAML_PATH):
    '''
    Read the yaml and return (configs, default_names, sweep_names).
    Crashes on missing or broken yaml

    :param path: path to the graph config yaml file
    :return: tuple (configs, default_names, sweep_names)
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find the graph config yaml at: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise RuntimeError(f"{path} is empty or broken")

    defaults = cfg.get("defaults", [])
    sweep = cfg.get("sweep", [])
    if len(defaults) == 0 and len(sweep) == 0:
        raise RuntimeError(f"{path} has no 'defaults' and no 'sweep' entries")

    configs = {}
    default_names = []
    sweep_names = []

    sections = (("defaults", defaults, default_names), ("sweep", sweep, sweep_names))
    for section, entries, name_list in sections:
        for entry in entries:
            name = entry["name"]
            family = entry["family"]
            params = entry.get("params", {}) or {}

            if family not in FAMILIES:
                raise ValueError(f"[{section}/{name}] unknown family '{family}'. Known: {sorted(FAMILIES)}")
            if name in configs:
                raise ValueError(f"the name '{name}' appears twice in the yaml")

            configs[name] = (FAMILIES[family], params)
            name_list.append(name)

    print(f"Loaded {len(default_names)} default + {len(sweep_names)} sweep graph types from {path}")
    return configs, default_names, sweep_names


GRAPH_CONFIGS, DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES = load_graph_configs()
ALL_GRAPH_TYPES = DEFAULT_GRAPH_TYPES + SWEEP_GRAPH_TYPES


def construct_graphs(jet, graph_type='knn_3'):
    '''
    Build one jet's graph by name, then attach all the physics info (track
    features, truth labels, jet-level values) to its nodes.

    :param jet: jet dict holding tracks, truth particles, and substructure info
    :param graph_type: name of the graph construction to use, must be a key in GRAPH_CONFIGS
    :return: networkx Graph with node and graph-level attributes filled in
    '''
    deta = jet['tracks']['angular_coord'][0]
    dphi = jet['tracks']['angular_coord'][1]

    d0 = jet['tracks']['impact_parameter_coord'][0]
    z0SinTheta = jet['tracks']['impact_parameter_coord'][1]

    lifetimeSignedD0 = jet['tracks']['lifetime_coord'][0]
    lifetimeSignedZ0SinTheta = jet['tracks']['lifetime_coord'][1]

    num_nodes = len(deta)

    if graph_type not in GRAPH_CONFIGS:
        raise ValueError(f"Unknown graph_type '{graph_type}'")

    build_function, params = GRAPH_CONFIGS[graph_type]
    edge_index = build_function(jet, **params)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index)
    G.remove_edges_from(nx.selfloop_edges(G))

    G.graph['label'] = jet['jet']['label']
    G.graph['truth_Lxy'] = jet['truth_particles']['Lxy']
    G.graph['truth_decayVertexZ'] = jet['truth_particles']['decayVertexZ']
    G.graph['truth_decayVertexDPhi'] = jet['truth_particles']['decayVertexDPhi']
    G.graph['frac_from_b'] = jet['physical_substructure']['frac_from_b']
    G.graph['frac_from_c'] = jet['physical_substructure']['frac_from_c']
    G.graph['num_sv_tracks'] = jet['physical_substructure']['num_sv_tracks']
    G.graph['num_tracks'] = jet['physical_substructure']['num_tracks']

    G.graph['max_Lxy'] = jet['global_truth']['max_Lxy']
    G.graph['max_d0_sig'] = jet['global_truth']['max_d0_sig']

    for i in range(num_nodes):
        G.nodes[i]['pos'] = (deta[i], dphi[i])
        G.nodes[i]['pos_impact_parameter'] = (d0[i], z0SinTheta[i])
        G.nodes[i]['pos_lifetime_coord'] = (lifetimeSignedD0[i], lifetimeSignedZ0SinTheta[i])
        G.nodes[i]['track_pt_frac'] = jet['tracks']['ptfrac'][i]
        G.nodes[i]['track_pt'] = jet['tracks']['pt'][i]
        G.nodes[i]['truth_vertex_idx'] = jet['tracks']['track_vertex_id'][i]
        G.nodes[i]['truth_origin_label'] = jet['tracks']['origin_label'][i]
        G.nodes[i]['dR'] = jet['tracks']['dR'][i]

    return G
