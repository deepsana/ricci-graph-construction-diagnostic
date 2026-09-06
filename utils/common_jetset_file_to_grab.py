import glob
import h5py
import numpy as np
from utils.graph_data_to_arrays import load_all_files_parallel

from src.graphs import DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES, ALL_GRAPH_TYPES

# Restrict evaluations to the intersection of jets present in both shard families
# to maintain consistent same-jets enforcement across separate default and sweep runs.
# Jet IDs correspond to source JetSet split row indices ("jet_<idx>").
SHARD_ROOT = "/media/mule/scratch/JetSet/07-21"
FAMILIES = {"default": list(DEFAULT_GRAPH_TYPES),
            "sweep":   list(SWEEP_GRAPH_TYPES)}

# Representative graph type per family used solely to read group-name jet IDs without loading node data.
ID_PROBE = {"default": "knn_3", "sweep": "knn_4"}

# Intersect across both families to ensure uniform jet selection regardless of run configuration.
INTERSECT_FAMILIES = ["default", "sweep"]

# File position offset added by parallel loader
LOADER_FILE_OFFSET = 10000000

def shard_files(split, family):
    '''
    Sorted shard paths for one split of one shard family.

    :param split: "train" or "test"
    :param family: "default" or "sweep"
    :return: list of paths
    '''
    files = sorted(glob.glob(f"{SHARD_ROOT}/temp_{split}/{family}_{split}_*.h5"))
    if len(files) == 0:
        raise SystemExit(f"FATAL: no {family} shards found for {split} under {SHARD_ROOT}")
    return files


def jet_ids_in_shards(files, gtype):
    '''
    Jet ids present in a list of shards, read from the group names under one
    graph type. Cheap: opens the files but loads no node data.

    :param files: shard paths
    :param gtype: graph type group to read the names from
    :return: set of int jet ids (raw source index)
    '''
    ids = set()
    for f in files:
        with h5py.File(f, "r") as h:
            if gtype not in h:
                raise SystemExit(f"FATAL: {gtype} not in {f}; fix ID_PROBE")
            for k in h[gtype].keys():
                ids.add(int(k.split("_")[1]))
    return ids


def common_jet_ids(split, families):
    '''
    Jets present in every shard family, i.e. jets that survived the same-jets rule
    in every saving run.

    :param split: "train" or "test"
    :param families: shard families to intersect
    :return: set of int jet ids
    '''
    sets = []
    for fam in families:
        sets.append(jet_ids_in_shards(shard_files(split, fam), ID_PROBE[fam]))

    common = set(sets[0])
    for s in sets[1:]:
        common = common & s

    pieces = []
    for fam, s in zip(families, sets):
        pieces.append(f"{fam}={len(s)}")
    counts = ", ".join(pieces)

    print(f"{split}: {counts}, common={len(common)}")
    return common


def subsample_ids(ids, n_max, seed):
    '''
    Deterministic subsample of a jet id set. Applied once per split so every
    construction sees exactly the same jets.

    :param ids: jet ids to draw from
    :param n_max: how many to keep; None (or a set already this small) keeps everything
    :param seed: seeds the draw, so the same call always returns the same jets
    :return: set of int jet ids
    '''
    if n_max is None or len(ids) <= n_max:
        return set(ids)
    arr = np.array(sorted(ids), dtype=np.int64)
    pick = np.random.default_rng(seed).choice(arr, size=n_max, replace=False)
    print(f"  subsampled {len(ids)} -> {n_max} jets (seed {seed})")

    out = set()
    for i in pick:
        out.add(int(i))
    return out


def load_split(split, graph_types, common, n_workers):
    '''
    Load graph_types for one split. Each type is read from the shard family that
    contains it, the loader's file offset is stripped from jet_id, then nodes and
    edges are restricted to the common jet ids so every construction sees exactly
    the same jets.

    :param split: "train" or "test"
    :param graph_types: constructions to load
    :param common: set of jet ids to keep (raw source index)
    :param n_workers: workers for load_all_files_parallel
    :return: dict graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    '''
    keep = np.array(sorted(common), dtype=np.int64)
    out = {}
    for family, family_types in FAMILIES.items():
        wanted = []
        for g in graph_types:
            if g in family_types:
                wanted.append(g)
        if len(wanted) == 0:
            continue

        d = load_all_files_parallel(shard_files(split, family), graph_types=wanted, n_workers=n_workers)
        for g in wanted:
            nodes = d[g]["nodes"]
            edges = d[g]["edges"]
            #back to the raw index so ids match the group names and `common`
            nodes["jet_id"] = nodes["jet_id"] % LOADER_FILE_OFFSET
            edges["jet_id"] = edges["jet_id"] % LOADER_FILE_OFFSET
            d[g]["nodes"] = nodes[nodes["jet_id"].isin(keep)].reset_index(drop=True)
            d[g]["edges"] = edges[edges["jet_id"].isin(keep)].reset_index(drop=True)
            out[g] = d[g]

    missing = set(graph_types) - set(out)
    if len(missing) > 0:
        raise SystemExit(f"FATAL: graph types not in any shard family: {sorted(missing)}")

    nodes0 = out[graph_types[0]]["nodes"]

    #  did the filter keep the jets we asked for? (catches any id mismatch, including a stale cache)
    n_kept = nodes0["jet_id"].nunique()
    if n_kept != len(keep):
        msg = f"{split}: {n_kept} of {len(keep)} common jets found in {graph_types[0]}"
        if n_kept < 0.5 * len(keep):
            raise SystemExit("FATAL: " + msg + ". jet_id does not match the ids in `common`; stale common_*_jet_ids.npy or changed id rule.")
        print("WARNING: " + msg)

    #  a physical jet has one flavor; more than one means ids collided
    if nodes0.groupby("jet_id")["flavor"].nunique().max() > 1:
        raise SystemExit(f"FATAL: {split}: a jet_id maps to more than one flavor;different jets are being merged under one id.")

    # every construction must now carry the identical jet set
    ref = set(nodes0["jet_id"].unique())
    for g in out:
        ids = set(out[g]["nodes"]["jet_id"].unique())
        if ids != ref:
            raise SystemExit(f"FATAL: {g} has {len(ids)} jets after filtering, expected {len(ref)}.")

    n_tracks = nodes0.groupby("jet_id").size()
    print(f"{split}: {len(ref)} jets, identical across {len(out)} graph types; tracks per jet median {int(n_tracks.median())}, max {int(n_tracks.max())}")
    return out