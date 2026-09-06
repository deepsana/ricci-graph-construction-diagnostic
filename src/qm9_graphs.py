import os
import numpy as np
import networkx as nx
import yaml
from itertools import combinations

from .qm9_true_bonds import get_true_labels, attach_true_labels

ATOM_MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}
ATOM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

TARGET_NAMES = ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H_enthalpy", "G", "Cv"]

DROP_HYDROGENS = False


def fix_number(text):
    '''
    Some QM9 files write numbers in a weird style like '1.6*^-6'.
    Python's float() can't handle '*^', so swap it for 'e' first.

    :param text: the raw number string from the file
    :return: the parsed float
    '''
    return float(text.replace("*^", "e"))


def read_qm9_xyz(filepath):
    '''
    Read one molecule's .xyz file and return everything in a plain dictionary.

    :param filepath: path to the .xyz file
    :return: the mol dict, or None if the file couldn't be parsed
    '''
    try:
        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        num_atoms = int(lines[0])

        parts = lines[1].split()
        mol_id = int(parts[1])
        targets = {}
        for position, name in enumerate(TARGET_NAMES):
            targets[name] = fix_number(parts[2 + position])

        elements = []
        coords = []
        mulliken = []
        for line in lines[2:2 + num_atoms]:
            parts = line.split()
            elements.append(parts[0])
            x = fix_number(parts[1])
            y = fix_number(parts[2])
            z = fix_number(parts[3])
            coords.append([x, y, z])
            mulliken.append(fix_number(parts[4]))

        smiles_parts = lines[2 + num_atoms + 1].split()
        smiles = smiles_parts[0]
        smiles_relaxed = smiles_parts[1] if len(smiles_parts) > 1 else smiles_parts[0]

        coords = np.array(coords)
        mulliken = np.array(mulliken)

        mass_list = []
        z_list = []
        for e in elements:
            mass_list.append(ATOM_MASS[e])
            z_list.append(ATOM_Z[e])
        mass = np.array(mass_list)
        Z = np.array(z_list, dtype=np.int32)

        if DROP_HYDROGENS:
            keep = Z > 1
            coords = coords[keep]
            mulliken = mulliken[keep]
            mass = mass[keep]
            Z = Z[keep]
            kept_elements = []
            for e in elements:
                if e != 'H':
                    kept_elements.append(e)
            elements = kept_elements
            num_atoms = len(elements)

        mol = {
            'mol_id': mol_id,
            'num_atoms': num_atoms,
            'smiles': smiles,
            'smiles_relaxed': smiles_relaxed,
            'atoms': {
                'coords': coords,
                'element': elements,
                'Z': Z,
                'mass': mass,
                'mulliken': mulliken,
            },
            'targets': targets,
        }

    except Exception:
        return None

    get_true_labels(mol)

    return mol


def pairwise_sq_distances(coords):
    '''
    A table of squared distances between every pair of atoms.

    :param coords: (N, 3) array of atom positions
    :return: (N, N) array of squared distances, diagonal set to infinity
    '''
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    return dist_sq


def construct_knn_graph_edges(mol, k):
    '''
    Connect every atom to its k nearest atoms.

    :param mol: mol dict with atom coordinates
    :param k: number of neighbors to connect each atom to
    :return: list of edge tuples
    '''
    coords = mol['atoms']['coords']
    num_nodes = len(coords)

    if num_nodes <= 1:
        return []
    k_max = min(k, num_nodes - 1)

    dist_sq = pairwise_sq_distances(coords)

    edges = set()
    for i in range(num_nodes):
        neighbors = np.argsort(dist_sq[i])[:k_max]
        for j in neighbors:
            edges.add(tuple(sorted((int(i), int(j)))))
    return list(edges)


def construct_radius_based_graph_edges(mol, radius):
    '''
    Connect every pair of atoms that are closer than radius.

    :param mol: mol dict with atom coordinates
    :param radius: distance cutoff, in Angstrom
    :return: list of edge tuples
    '''
    coords = mol['atoms']['coords']
    num_nodes = len(coords)

    if num_nodes <= 1:
        return []

    dist_sq = pairwise_sq_distances(coords)
    rows, cols = np.where(dist_sq < radius ** 2)

    edges = set()
    for i, j in zip(rows, cols):
        edges.add(tuple(sorted((int(i), int(j)))))
    return list(edges)


def mass_weighted_distances(coords, mass, p):
    '''
    The generalized kt distance.

    :param coords: (N, 3) array of atom positions
    :param mass: (N,) array of atom masses
    :param p: exponent applied to mass
    :return: (N, N) array of weighted squared distances
    '''
    dist_sq = pairwise_sq_distances(coords)
    if p == 0:
        return dist_sq
    m_pow = mass ** (2 * p)
    m_min = np.minimum(m_pow[:, np.newaxis], m_pow[np.newaxis, :])
    return m_min * dist_sq


def construct_kt_threshold_graph_edges(mol, quantile, min_neighbors, max_neighbors, p):
    '''
    Use generalized kt distance metric to build undirected edges.

    :param mol: mol dict with atom coordinates and mass
    :param quantile: quantile of pairwise distances used as the connection threshold
    :param min_neighbors: minimum neighbors to guarantee per atom
    :param max_neighbors: maximum neighbors allowed per atom
    :param p: exponent applied to mass in the distance metric
    :return: list of edge tuples
    '''
    coords = mol['atoms']['coords']
    mass = mol['atoms']['mass']
    num_nodes = len(coords)

    if num_nodes <= 1:
        return []

    dist_sq = pairwise_sq_distances(coords)
    d_ij = mass_weighted_distances(coords, mass, p)

    finite = d_ij[np.isfinite(d_ij)]
    if len(finite) == 0:
        return []
    d_cut = np.quantile(finite, quantile)

    edges = set()
    for i in range(num_nodes):
        neighbors = np.where(d_ij[i] < d_cut)[0]

        if len(neighbors) < min_neighbors:
            k_min = min(min_neighbors, num_nodes - 1)
            neighbors = np.argsort(dist_sq[i])[:k_min]

        if len(neighbors) > max_neighbors:
            neighbor_dists = d_ij[i][neighbors]
            neighbors = neighbors[np.argsort(neighbor_dists)[:max_neighbors]]

        for j in neighbors:
            edges.add(tuple(sorted((int(i), int(j)))))
    return list(edges)


def construct_laman_graph_edges(mol, p):
    '''
    Building Laman graph using generalized kt distance metric.

    :param mol: mol dict with atom coordinates and mass
    :param p: exponent applied to mass in the distance metric
    :return: list of edge tuples
    '''
    coords = mol['atoms']['coords']
    mass = mol['atoms']['mass']
    num_nodes = len(coords)

    if num_nodes <= 1:
        return []
    if num_nodes == 2:
        return [(0, 1)]

    sort_idx = np.argsort(-mass)
    coords_sorted = coords[sort_idx]
    mass_sorted = mass[sort_idx]

    d_ij = mass_weighted_distances(coords_sorted, mass_sorted, p)

    edges_sorted = set()
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

    edges = set()
    for i, j in edges_sorted:
        edges.add(tuple(sorted((int(sort_idx[i]), int(sort_idx[j])))))
    return list(edges)


def construct_unique_k_graph_edges(mol, k, p):
    '''
    Builds a graph by seeding a clique of the hardest particles, then
    attaching each remaining atom to its k nearest existing neighbors.

    :param mol: mol dict with atom coordinates and mass
    :param k: how many existing atoms to connect each new atom to
    :param p: exponent applied to mass in the distance metric
    :return: list of edge tuples
    '''
    coords = mol['atoms']['coords']
    mass = mol['atoms']['mass']
    num_nodes = len(coords)

    if num_nodes <= 1:
        return []

    sort_idx = np.argsort(-mass)
    coords_sorted = coords[sort_idx]
    mass_sorted = mass[sort_idx]

    d_ij = mass_weighted_distances(coords_sorted, mass_sorted, p)

    edges_sorted = set()
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

    edges = set()
    for i, j in edges_sorted:
        edges.add(tuple(sorted((int(sort_idx[i]), int(sort_idx[j])))))
    return list(edges)


def construct_fully_connected_edges(mol):
    '''
    Connect every pair of atoms.

    :param mol: mol dict with num_atoms
    :return: list of edge tuples
    '''
    num_nodes = mol['num_atoms']
    if num_nodes <= 1:
        return []

    edges = []
    for i, j in combinations(range(num_nodes), 2):
        edges.append((i, j))
    return edges


FAMILIES = {
    'knn': construct_knn_graph_edges,
    'radius': construct_radius_based_graph_edges,
    'kt_threshold': construct_kt_threshold_graph_edges,
    'laman': construct_laman_graph_edges,
    'unique_k': construct_unique_k_graph_edges,
    'fully_connected': construct_fully_connected_edges,
}

YAML_PATH = os.environ.get("QM9_GRAPH_CONFIG", "configs/qm9_graph_configs.yaml")


def load_graph_configs(path=None):
    '''
    Read the yaml and return (configs, default_names, sweep_names).

    :param path: path to the yaml file, defaults to YAML_PATH if not given
    :return: tuple (configs, default_names, sweep_names)
    '''
    if path is None:
        path = YAML_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the QM9 graph config yaml at: {path}\n"
            "If running on a remote machine, make sure the configs/ "
            "folder was copied over too (or set QM9_GRAPH_CONFIG).")

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

    for entry in defaults:
        name = entry["name"]
        family = entry["family"]
        params = entry.get("params", {})

        if family not in FAMILIES:
            raise ValueError(f"unknown family '{family}' for variant '{name}'. Known: {sorted(FAMILIES)}")
        if name in configs:
            raise ValueError(f"the name '{name}' appears twice in the yaml")

        configs[name] = (FAMILIES[family], params)
        default_names.append(name)

    for entry in sweep:
        name = entry["name"]
        family = entry["family"]
        params = entry.get("params", {})

        if family not in FAMILIES:
            raise ValueError(f"unknown family '{family}' for variant '{name}'. Known: {sorted(FAMILIES)}")
        if name in configs:
            raise ValueError(f"the name '{name}' appears twice in the yaml")

        configs[name] = (FAMILIES[family], params)
        sweep_names.append(name)

    print(f"Loaded {len(default_names)} default + {len(sweep_names)} sweep QM9 graph types from {path}")
    return configs, default_names, sweep_names


GRAPH_CONFIGS, DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES = load_graph_configs()
ALL_GRAPH_TYPES = DEFAULT_GRAPH_TYPES + SWEEP_GRAPH_TYPES


def construct_graphs(mol, graph_type='knn_3'):
    '''
    Build one molecule's graph by name, then attach the chemistry information to it.

    :param mol: mol dict holding atoms, targets, and true-bond info
    :param graph_type: name of the graph construction to use, must be a key in GRAPH_CONFIGS
    :return: networkx Graph with node and graph-level attributes filled in
    '''
    num_nodes = mol['num_atoms']

    if graph_type not in GRAPH_CONFIGS:
        raise ValueError(f"Unknown graph_type '{graph_type}'. Valid names come from qm9_graph_configs.yaml")

    build_function, params = GRAPH_CONFIGS[graph_type]
    edge_index = build_function(mol, **params)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index)
    G.remove_edges_from(nx.selfloop_edges(G))

    G.graph['mol_id'] = mol['mol_id']
    G.graph['num_atoms'] = num_nodes
    G.graph['smiles'] = mol['smiles']
    for name in TARGET_NAMES:
        G.graph[name] = mol['targets'][name]

    coords = mol['atoms']['coords']
    for i in range(num_nodes):
        G.nodes[i]['pos'] = (coords[i, 0], coords[i, 1], coords[i, 2])
        G.nodes[i]['element'] = mol['atoms']['element'][i]
        G.nodes[i]['Z'] = int(mol['atoms']['Z'][i])
        G.nodes[i]['mass'] = float(mol['atoms']['mass'][i])
        G.nodes[i]['mulliken'] = float(mol['atoms']['mulliken'][i])

    attach_true_labels(G, get_true_labels(mol))

    return G