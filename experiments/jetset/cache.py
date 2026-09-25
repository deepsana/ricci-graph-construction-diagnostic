import os
import numpy as np
import pandas as pd

IDS_FILE = "jet_ids.npy"


IDS_FILE = "jet_ids.npy"
TABLE_NAMES = ("nodes", "edges")


def table_dir(cache_dir, graph_type, table_name):
    return os.path.join(cache_dir, f"{graph_type}_{table_name}")


def read_cached_split(cache_dir, graph_types, kept_ids, node_cols, edge_cols):
    """
    Cached node and edge frames, if the cache holds exactly these jets, constructions and columns.

    :param kept_ids: jet ids the cache must hold
    :return: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}, or None when anything is missing or stale
    """
    ids_path = os.path.join(cache_dir, IDS_FILE)
    if not os.path.exists(ids_path):
        print(f"[cache] nothing cached in {cache_dir}; reading the shards", flush=True)
        return None

    wanted_ids = np.array(sorted(kept_ids), dtype=np.int64)
    if not np.array_equal(np.load(ids_path), wanted_ids):
        print(f"[cache] {cache_dir} was built from other jets; reading the shards", flush=True)
        return None

    data_dict = {}
    for graph_type in graph_types:
        data_dict[graph_type] = {}
        for table_name, columns in zip(TABLE_NAMES, (node_cols, edge_cols)):
            folder = table_dir(cache_dir, graph_type, table_name)
            if not os.path.isdir(folder):
                print(f"[cache] {graph_type} is not cached; reading the shards", flush=True)
                return None
            # An empty folder stands for an empty table.
            if not os.listdir(folder):
                data_dict[graph_type][table_name] = pd.DataFrame()
                continue

            loaded = {}
            for col in columns:
                path = os.path.join(folder, f"{col}.npy")
                if not os.path.exists(path):
                    print(f"[cache] {graph_type}/{table_name} has no column {col}; "
                          f"reading the shards", flush=True)
                    return None
                loaded[col] = np.load(path)
            data_dict[graph_type][table_name] = pd.DataFrame(loaded)

    print(f"[cache] loaded {len(data_dict)} constructions from {cache_dir}", flush=True)
    return data_dict


def write_cached_split(cache_dir, data_dict, kept_ids):
    """
    Save every node and edge frame as one .npy per column.
    The jet ids are removed first and written last, so an interrupted write never looks complete to read_cached_split.
    """
    os.makedirs(cache_dir, exist_ok=True)
    ids_path = os.path.join(cache_dir, IDS_FILE)
    if os.path.exists(ids_path):
        os.remove(ids_path)

    for graph_type, tables in data_dict.items():
        for table_name in TABLE_NAMES:
            folder = table_dir(cache_dir, graph_type, table_name)
            os.makedirs(folder, exist_ok=True)
            for old_file in os.listdir(folder):
                if old_file.endswith(".npy"):
                    os.remove(os.path.join(folder, old_file))

            df = tables.get(table_name)
            if df is None:
                continue
            for col in df.columns:
                np.save(os.path.join(folder, f"{col}.npy"), df[col].to_numpy())

    np.save(ids_path, np.array(sorted(kept_ids), dtype=np.int64))
    print(f"[cache] wrote {len(data_dict)} constructions to {cache_dir}", flush=True)


