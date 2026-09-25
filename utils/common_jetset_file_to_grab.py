import glob
import h5py
import numpy as np

from src.graphs import DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES
from utils.graph_data_to_arrays import load_all_files_parallel

# The default and sweep runs were saved separately and dropped different
# jets, so evaluations are restricted to jets present in both shard
# families. Jet ids are row indices of the source JetSet split ("jet_<idx>").
SHARD_ROOT = "/media/mule/scratch/JetSet/07-21"
FAMILIES = {"default": list(DEFAULT_GRAPH_TYPES),
            "sweep": list(SWEEP_GRAPH_TYPES),
            }
# One graph type per family, used only to read jet ids from the group names without loading any node data.
ID_PROBE = {"default": "knn_3", "sweep": "knn_4"}
INTERSECT_FAMILIES = ["default", "sweep"]
# The parallel loader adds a per-file multiple of this offset to every jet_id
LOADER_FILE_OFFSET = 10000000


def shard_files(split, family):
    """
    List the shard paths of one split of one shard family.

    :param split: "train" or "test"
    :param family: "default" or "sweep"
    :return: sorted list of paths
    """
    pattern = f"{SHARD_ROOT}/temp_{split}/{family}_{split}_*.h5"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"FATAL: no {family} shards found for {split} under {SHARD_ROOT}")
    return files


def jet_ids_in_shards(files, graph_type):
    """
    Read the jet ids in a list of shards from the group names under one
    graph type. No node data is loaded.

    :param files: shard paths
    :param graph_type: graph type group to read the names from
    :return: set of jet ids (raw source index)
    """
    jet_ids = set()
    for path in files:
        with h5py.File(path, "r") as shard:
            if graph_type not in shard:
                raise SystemExit(f"FATAL: {graph_type} not in {path}; fix ID_PROBE")
            for group_name in shard[graph_type].keys():
                jet_ids.add(int(group_name.split("_")[1]))
    return jet_ids


def common_jet_ids(split, families):
    """
    Find the jets present in every shard family, i.e. the jets that
    survived the zero-edge rule in every saving run.

    :param split: "train" or "test"
    :param families: shard families to intersect
    :return: set of jet ids
    """
    id_sets = []
    for family in families:
        files = shard_files(split, family)
        id_sets.append(jet_ids_in_shards(files, ID_PROBE[family]))

    common = set(id_sets[0])
    for family_ids in id_sets[1:]:
        common = common & family_ids

    counts = []
    for family, family_ids in zip(families, id_sets):
        counts.append(f"{family}={len(family_ids)}")
    counts_text = ", ".join(counts)
    print(f"{split}: {counts_text}, common={len(common)}")
    return common


def subsample_ids(ids, n_max, seed):
    """
    Draw a reproducible subsample of jet ids. It is applied once per split
    so every construction sees exactly the same jets.

    :param ids: jet ids to draw from
    :param n_max: number of ids to keep, or None to keep all of them
    :param seed: seed of the draw
    :return: set of jet ids
    """
    if n_max is None or len(ids) <= n_max:
        return set(ids)
    sorted_ids = np.array(sorted(ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    picked = rng.choice(sorted_ids, size=n_max, replace=False)
    print(f"  subsampled {len(ids)} to {n_max} jets (seed {seed})")

    subsample = set()
    for jet_id in picked:
        subsample.add(int(jet_id))
    return subsample


def load_split(split, graph_types, common_ids, n_workers):
    """
    Load the graph types of one split. Each type is read from the shard
    family that contains it, the loader offset is removed from jet_id, and
    nodes and edges are restricted to the common jet ids so every
    construction sees exactly the same jets.

    :param split: "train" or "test"
    :param graph_types: constructions to load
    :param common_ids: set of jet ids to keep (raw source index)
    :param n_workers: worker processes for load_all_files_parallel
    :return: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    """
    keep_ids = np.array(sorted(common_ids), dtype=np.int64)
    loaded = {}
    for family, family_types in FAMILIES.items():
        wanted = []
        for graph_type in graph_types:
            if graph_type in family_types:
                wanted.append(graph_type)
        if not wanted:
            continue

        files = shard_files(split, family)
        family_data = load_all_files_parallel(files, graph_types=wanted, n_workers=n_workers)
        for graph_type in wanted:
            tables = family_data[graph_type]
            nodes = tables["nodes"]
            edges = tables["edges"]
            # Back to the raw index, so the ids match the group names
            nodes["jet_id"] = nodes["jet_id"] % LOADER_FILE_OFFSET
            edges["jet_id"] = edges["jet_id"] % LOADER_FILE_OFFSET
            kept_nodes = nodes[nodes["jet_id"].isin(keep_ids)]
            kept_edges = edges[edges["jet_id"].isin(keep_ids)]
            tables["nodes"] = kept_nodes.reset_index(drop=True)
            tables["edges"] = kept_edges.reset_index(drop=True)
            loaded[graph_type] = tables

    missing = []
    for graph_type in graph_types:
        if graph_type not in loaded:
            missing.append(graph_type)
    if missing:
        raise SystemExit(f"FATAL: graph types not in any shard family: {sorted(missing)}")

    first_type = graph_types[0]
    first_nodes = loaded[first_type]["nodes"]

    # Catches any id mismatch, including a stale cache of common ids
    n_kept = first_nodes["jet_id"].nunique()
    if n_kept != len(keep_ids):
        message = (f"{split}: {n_kept} of {len(keep_ids)} common jets found in {first_type}")
        if n_kept < 0.5 * len(keep_ids):
            raise SystemExit(f"FATAL: {message}. jet_id does not match the common ids; stale common_*_jet_ids.npy or changed id rule.")
        print(f"WARNING: {message}")

    #  jet has one flavor, so more than one means ids collided
    flavors_per_jet = first_nodes.groupby("jet_id")["flavor"].nunique()
    if flavors_per_jet.max() > 1:
        raise SystemExit(f"FATAL: {split}: a jet_id maps to more than one flavor; different jets are being merged under one id.")

    reference_ids = set(first_nodes["jet_id"].unique())
    for graph_type in loaded:
        graph_ids = set(loaded[graph_type]["nodes"]["jet_id"].unique())
        if graph_ids != reference_ids:
            raise SystemExit(f"FATAL: {graph_type} has {len(graph_ids)} jets after filtering, expected {len(reference_ids)}.")

    tracks_per_jet = first_nodes.groupby("jet_id").size()
    print(f"{split}: {len(reference_ids)} jets, identical across {len(loaded)} graph types; tracks per jet median {int(tracks_per_jet.median())}, max {int(tracks_per_jet.max())}")
    return loaded
