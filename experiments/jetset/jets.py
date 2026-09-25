
import glob
import hashlib
import os
import h5py
import numpy as np

from experiments.jetset.config import CACHE_ROOT, ID_PROBE, INTERSECT_FAMILIES, SHARD_ROOT


def shard_files(split, family):
    """
    Sorted shard paths of one split of one shard family.

    :param split: "train" or "test"
    :param family: "default" or "sweep"
    """
    pattern = f"{SHARD_ROOT}/temp_{split}/{family}_{split}_*.h5"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"FATAL: no {family} shards found for {split} under {SHARD_ROOT}")
    return files


def jet_ids_in_shards(files, graph_type):
    """
    Jet ids of one graph type, read from the group names ("jet_<idx>") without
    loading any node data.

    :return: set of raw source indices
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
    """Jets present in every shard family: those that survived the zero-edge filter of every run."""
    id_sets = [jet_ids_in_shards(shard_files(split, family), ID_PROBE[family])
               for family in families]
    common = set.intersection(*id_sets)
    counts_text = ", ".join(f"{family}={len(family_ids)}" for family, family_ids in zip(families, id_sets))
    print(f"{split}: {counts_text}, common={len(common)}")
    return common


def subsample_ids(ids, n_max, seed):
    """
    Reproducible subsample of jet ids. Sorting before the draw makes the
    result independent of set iteration order, so every construction of a
    split gets the same jets.

    :param n_max: number of ids to keep, or None to keep all of them
    :return: set of int jet ids
    """
    if n_max is None or len(ids) <= n_max:
        return set(ids)
    sorted_ids = np.array(sorted(ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    picked = rng.choice(sorted_ids, size=n_max, replace=False)
    print(f"  subsampled {len(ids)} to {n_max} jets (seed {seed})")
    return {int(jet_id) for jet_id in picked}


def kept_ids_path(split, max_jets, seed):
    """
    Path of the frozen subsample of one split. Size and seed are part of the
    name, so runs with different settings never overwrite each other.
    """
    size = "all" if max_jets is None else str(max_jets)
    return os.path.join(CACHE_ROOT, f"kept_{split}_jet_ids_{size}_seed{seed}.npy")


def ids_fingerprint(ids):
    """
    Order-independent fingerprint of a set of jet ids, for the log.

    :return: tuple (count, short sha1 of the sorted ids)
    """
    arr = np.array(sorted(int(jet_id) for jet_id in ids), dtype=np.int64)
    return len(arr), hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def load_common_ids(split):
    """Jet ids common to all shard families, cached under CACHE_ROOT after the first scan."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    path = os.path.join(CACHE_ROOT, f"common_{split}_jet_ids.npy")
    if os.path.exists(path):
        arr = np.load(path)
    else:
        arr = np.array(sorted(common_jet_ids(split, INTERSECT_FAMILIES)), dtype=np.int64)
        np.save(path, arr)
    arr = np.sort(arr.astype(np.int64))
    print(f"{split}: {len(arr)} common jet ids ({path})", flush=True)
    return set(arr.tolist())


def get_kept_ids(split, max_jets, seed):
    """
    The jets this run uses. The first call draws the subsample and freezes it
    to disk; later calls, including the bootstrap, read it back.

    :param max_jets: subsample size, or None for all common jets
    :return: set of jet ids
    """
    path = kept_ids_path(split, max_jets, seed)
    if os.path.exists(path):
        kept = set(np.load(path).astype(np.int64).tolist())
        source = "frozen file"
    else:
        kept = subsample_ids(load_common_ids(split), max_jets, seed)
        np.save(path, np.array(sorted(kept), dtype=np.int64))
        source = "new subsample, frozen"
    count, digest = ids_fingerprint(kept)
    print(f"[jets] {split}: {count} kept jets, fingerprint {digest} ({source}: {path})", flush=True)
    return kept


def check_loaded_jets(data_dict, kept_ids, split):
    """
    Stop the run unless every loaded construction holds exactly kept_ids.

    :param data_dict: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    :param kept_ids: output of get_kept_ids
    :raises RuntimeError: on any mismatch
    """
    kept = {int(jet_id) for jet_id in kept_ids}
    problems = []
    for graph_type, tables in data_dict.items():
        loaded = set(np.unique(tables["nodes"]["jet_id"].to_numpy()).astype(np.int64).tolist())
        if loaded != kept:
            problems.append(f"{graph_type}: {len(kept - loaded)} kept jets missing, {len(loaded - kept)} unexpected jets")
    if problems:
        raise RuntimeError(f"[jets] {split}: loaded jets differ from kept list:\n  + \n  ".join(problems))
    count, digest = ids_fingerprint(kept)
    print(f"[jets] {split}: all {len(data_dict)} constructions hold the same {count} jets, fingerprint {digest}", flush=True)
