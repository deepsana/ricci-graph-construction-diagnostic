import gc
import numpy as np
import pandas as pd

from experiments.jetset.cache import read_cached_split, write_cached_split
from experiments.jetset.config import CACHE_DIR, FAMILIES, GRAPH_TYPES, KEEP_EDGE_COLS, KEEP_NODE_COLS, LOADER_FILE_OFFSET, N_WORKERS, SMALL_INT_COLS
from experiments.jetset.jets import check_loaded_jets, shard_files
from utils.graph_data_to_arrays import load_all_files_parallel
from utils.memory import frame_gb, rss_gb


TABLE_COLUMNS = (("nodes", KEEP_NODE_COLS), ("edges", KEEP_EDGE_COLS))


def load_split(split, graph_types, common_ids, n_workers, node_cols, edge_cols):
    """
    Read the requested constructions of one split from their shard families,
    keeping only common_ids.

    :param split: "train" or "test"
    :param common_ids: jet ids to keep
    :param node_cols: node columns to read
    :param edge_cols: edge columns to read
    :return: loaded split
    """
    keep_ids = np.array(sorted(common_ids), dtype=np.int64)
    loaded = {}
    for family, family_types in FAMILIES.items():
        wanted = [graph_type for graph_type in graph_types if graph_type in family_types]
        if not wanted:
            continue

        family_data = load_all_files_parallel(
            shard_files(split, family), graph_types=wanted, n_workers=n_workers,
            keep_ids=keep_ids, node_cols=node_cols, edge_cols=edge_cols)
        for graph_type in wanted:
            tables = family_data[graph_type]
            for table_name in ("nodes", "edges"):
                table = tables[table_name]
                # Undo the loader's per-file offset to get the raw source index back.
                table["jet_id"] = table["jet_id"] % LOADER_FILE_OFFSET
                # The workers already skip other jets; this is a cheap safety net.
                tables[table_name] = table[table["jet_id"].isin(keep_ids)].reset_index(drop=True)
            loaded[graph_type] = tables

    validate_split(loaded, graph_types, keep_ids, split)
    return loaded


def validate_split(loaded, graph_types, keep_ids, split):
    """
    Stop the run if a construction is missing or the constructions disagree on their jets
    """
    missing = [graph_type for graph_type in graph_types if graph_type not in loaded]
    if missing:
        raise SystemExit(f"FATAL: graph types not in any shard family: {sorted(missing)}")

    first_type = graph_types[0]
    first_nodes = loaded[first_type]["nodes"]
    if first_nodes.empty:
        raise SystemExit(f"FATAL: {split}: no nodes loaded for {first_type}")

    n_kept = first_nodes["jet_id"].nunique()
    if n_kept != len(keep_ids):
        message = f"{split}: {n_kept}/{len(keep_ids)} common jets found in {first_type}"
        # Losing a few jets is survivable; losing half means the id rule changed.
        if n_kept < 0.5 * len(keep_ids):
            raise SystemExit(f"FATAL: {message}. Stale common IDs cache or changed ID rule.")
        print(f"WARNING: {message}")

    # Two source jets mapped to one id would show up as a jet with several flavors.
    if first_nodes.groupby("jet_id")["flavor"].nunique().max() > 1:
        raise SystemExit(f"FATAL: {split}: Jet ID collision (multiple flavors mapped to a single ID).")

    reference_ids = set(first_nodes["jet_id"].unique())
    for graph_type, tables in loaded.items():
        graph_ids = set(tables["nodes"]["jet_id"].unique())
        if graph_ids != reference_ids:
            raise SystemExit(f"FATAL: {graph_type} has {len(graph_ids)} jets, expected {len(reference_ids)}.")

    tracks = first_nodes.groupby("jet_id").size()
    print(f"{split}: {len(reference_ids)} jets across {len(loaded)} graph types | tracks/jet - median: {int(tracks.median())}, max: {int(tracks.max())}")


def prune_split(data_dict):
    """
    Keep only the node and edge columns that the scoring code reads
    """
    size_before = 0.0
    size_after = 0.0
    dropped_cols = None
    for graph_type, tables in data_dict.items():
        for table_name, keep_cols in TABLE_COLUMNS:
            df = tables.get(table_name)
            if df is None or df.empty:
                continue

            missing = [col for col in keep_cols if col not in df.columns]
            if missing:
                raise KeyError(f"{graph_type}/{table_name} is missing required columns {missing}")
            if dropped_cols is None:
                dropped_cols = [col for col in df.columns if col not in keep_cols]

            size_before += frame_gb(df)
            tables[table_name] = df[keep_cols].copy()
            size_after += frame_gb(tables[table_name])

    gc.collect()
    print(f"[mem] pruned columns {dropped_cols}: {size_before:.1f} GB -> {size_after:.1f} GB; RSS {rss_gb():.1f} GB", flush=True)
    return data_dict


def shrink_dtypes(df):
    """
    Downcast a node or edge frame in place: floats to float32, the ids in
    SMALL_INT_COLS to int16, other integers to the narrowest type that fits.
    """
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            if series.dtype != np.float32:
                df[col] = series.astype(np.float32)
        elif pd.api.types.is_integer_dtype(series):
            if col in SMALL_INT_COLS:
                df[col] = series.astype(np.int16)
            else:
                df[col] = pd.to_numeric(series, downcast="integer")
    return df


def shrink_split(data_dict):
    """Downcast every node and edge table of a loaded split."""
    size_before = 0.0
    size_after = 0.0
    for tables in data_dict.values():
        for table_name in ("nodes", "edges"):
            df = tables.get(table_name)
            if df is None or df.empty:
                continue
            size_before += frame_gb(df)
            shrink_dtypes(df)
            size_after += frame_gb(df)
    gc.collect()
    print(f"[mem] frames: {size_before:.1f} GB -> {size_after:.1f} GB after downcast;RSS {rss_gb():.1f} GB")
    return data_dict


def drop_graph_types(data_dict, keep):
    """Free every construction that is not in keep."""
    keep = set(keep)
    for graph_type in [graph_type for graph_type in data_dict if graph_type not in keep]:
        del data_dict[graph_type]
    gc.collect()
    print(f"[mem] kept {len(data_dict)} constructions for conditioning; RSS {rss_gb():.1f} GB", flush=True)
    return data_dict


def load_kept_jets(kept_ids):
    """
    Pruned, downcast frames of the kept train jets: from the column cache when
    it matches, otherwise from the shards, which then refill the cache.

    :param kept_ids: output of get_kept_ids
    :return: loaded split
    """
    train_dict = read_cached_split(CACHE_DIR, GRAPH_TYPES, kept_ids, KEEP_NODE_COLS, KEEP_EDGE_COLS)
    if train_dict is None:
        train_dict = load_split("train", GRAPH_TYPES, kept_ids, N_WORKERS, node_cols=KEEP_NODE_COLS, edge_cols=KEEP_EDGE_COLS)
        train_dict = prune_split(train_dict)
        train_dict = shrink_split(train_dict)
        write_cached_split(CACHE_DIR, train_dict, kept_ids)
    check_loaded_jets(train_dict, kept_ids, "train")
    return train_dict
