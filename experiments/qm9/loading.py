import glob
import time
from multiprocessing import Pool
import h5py
import numpy as np
import pandas as pd

from experiments.qm9.config import CURVATURES, EDGE_COLS, NODE_COLS


def append_group_columns(group, columns):
    """
    Append every dataset of an h5 group to columns; returns the row count
    """
    n_rows = 0
    for column in group:
        values = group[column][:]
        n_rows = len(values)
        columns.setdefault(column, []).append(values)
    return n_rows


def read_shard_file(job):
    """
    Read every molecule of one graph type from one shard file. Runs in a
    worker process, so it takes a single picklable argument.

    :param job: tuple (path, graph_type)
    :return: tuple (node_cols, edge_cols, mol_rows), where the first two map
        a column name to a list of per-molecule arrays and mol_rows is a list
        of molecule attribute dicts
    """
    path, graph_type = job
    node_cols = {}
    edge_cols = {}
    mol_rows = []
    with h5py.File(path, "r") as shard:
        if graph_type not in shard:
            return node_cols, edge_cols, mol_rows
        group = shard[graph_type]
        for mol_name in group:
            mol = group[mol_name]
            mol_id = int(mol.attrs["mol_id"])

            n_atoms = append_group_columns(mol["nodes"], node_cols)
            node_cols.setdefault("mol_id", []).append(np.full(n_atoms, mol_id, dtype=np.int64))
            # Without a node_id column, row order is atom order.
            if "node_id" not in mol["nodes"]:
                node_cols.setdefault("node_id", []).append(np.arange(n_atoms, dtype=np.int64))

            # A molecule without src has no edges in this construction.
            if "src" in mol["edges"]:
                n_edges = append_group_columns(mol["edges"], edge_cols)
                edge_cols.setdefault("mol_id", []).append(np.full(n_edges, mol_id, dtype=np.int64))

            mol_rows.append(dict(mol.attrs))
    return node_cols, edge_cols, mol_rows


def columns_to_frame(columns):
    """
    Concatenate the per-molecule arrays of every column into one DataFrame.

    :param columns: {column name: list of arrays} from read_shard_file
    :return: DataFrame, empty if columns is empty
    """
    if not columns:
        return pd.DataFrame()

    lengths = {name: sum(len(values) for values in arrays) for name, arrays in columns.items()}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"FATAL: shard columns have different total lengths -- some molecules lack a column that others have: {lengths}")

    df = pd.DataFrame({name: np.concatenate(arrays) for name, arrays in columns.items()})

    # h5py returns strings as bytes.
    for name in df.columns:
        if df[name].dtype != object or len(df) == 0:
            continue
        if isinstance(df[name].iloc[0], bytes):
            df[name] = df[name].str.decode("utf-8")
    return df


def run_jobs(jobs, n_workers):
    """
    Results of read_shard_file for every job, in job order
    """
    start_time = time.time()
    results = []
    if n_workers > 1 and len(jobs) > 1:
        with Pool(min(n_workers, len(jobs))) as pool:
            # imap keeps the results in job order.
            for job_number, result in enumerate(pool.imap(read_shard_file, jobs), start=1):
                results.append(result)
                if job_number % 25 == 0 or job_number == len(jobs):
                    print(f"  {job_number}/{len(jobs)} jobs done ({time.time() - start_time:.0f} s)", flush=True)
    else:
        for job_number, job in enumerate(jobs, start=1):
            results.append(read_shard_file(job))
            print(f"  {job_number}/{len(jobs)} jobs done ({time.time() - start_time:.0f} s)", flush=True)
    return results


def harmonize_columns(nodes, edges):
    """
    Accept both shard writer conventions: is_hydrogen from Z, and edge curvature columns named like the node ones.
    """
    if "is_hydrogen" not in nodes.columns and "Z" in nodes.columns:
        nodes["is_hydrogen"] = nodes["Z"] == 1
    for curvature in CURVATURES:
        node_name = f"{curvature}_curvature"
        edge_name = f"{curvature}_edge_curvature"
        if edge_name not in edges.columns and node_name in edges.columns:
            edges = edges.rename(columns={node_name: edge_name})
    if len(edges) > 0:
        edges["src"] = edges["src"].astype(int)
        edges["dst"] = edges["dst"].astype(int)
    return nodes, edges


def load_shards(file_list, graph_types, n_workers, max_files):
    """
    Read the shards into one set of tables per graph type. Every (file, graph type) pair is a separate job in one worker pool

    :param n_workers: worker processes; 1 reads everything in this process
    :param max_files: only read this many files, or None for all
    :return: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    """
    if max_files is not None:
        file_list = file_list[:max_files]
    jobs = [(path, graph_type) for graph_type in graph_types for path in file_list]

    start_time = time.time()
    print(f"reading {len(file_list)} files x {len(graph_types)} graph types = {len(jobs)} jobs on {min(n_workers, len(jobs))} workers", flush=True)
    results = run_jobs(jobs, n_workers)

    grouped = {graph_type: {"nodes": {}, "edges": {}, "mols": []} for graph_type in graph_types}
    for (_, graph_type), (job_nodes, job_edges, job_mols) in zip(jobs, results):
        target = grouped[graph_type]
        for name, arrays in job_nodes.items():
            target["nodes"].setdefault(name, []).extend(arrays)
        for name, arrays in job_edges.items():
            target["edges"].setdefault(name, []).extend(arrays)
        target["mols"].extend(job_mols)

    data_dict = {}
    for graph_type in graph_types:
        parts = grouped[graph_type]
        if not parts["mols"]:
            raise SystemExit(f"FATAL: no molecules found for graph type '{graph_type}' in {len(file_list)} files. Check GRAPH_TYPES against the yaml names used when saving.")
        nodes = columns_to_frame(parts["nodes"])
        if parts["edges"]:
            edges = columns_to_frame(parts["edges"])
        else:
            edges = pd.DataFrame(columns=["mol_id", "src", "dst"])
        nodes, edges = harmonize_columns(nodes, edges)
        mols = pd.DataFrame(parts["mols"])

        data_dict[graph_type] = {"nodes": nodes, "edges": edges, "mols": mols}
        print(f"{graph_type}: {len(mols)} molecules, {len(nodes)} atoms, {len(edges)} edges", flush=True)
    print(f"load_shards: {time.time() - start_time:.0f} s", flush=True)
    return data_dict


def missing_columns(nodes, edges):
    return ([col for col in NODE_COLS if col not in nodes.columns]
            + [col for col in EDGE_COLS if col not in edges.columns])


def truth_flags(tables):
    """Per-molecule labels_ok flags indexed by mol_id, or None if no table has them."""
    if "mols" in tables and "labels_ok" in tables["mols"].columns:
        return tables["mols"].set_index("mol_id")
    if "labels_ok" in tables["nodes"].columns:
        return tables["nodes"].drop_duplicates("mol_id").set_index("mol_id")
    return None


def check_truth_columns(data_dict):
    """
    Check that the truth and curvature columns survived serialization, and
    drop the molecules whose truth lookup failed (labels_ok != 1).

    :return: the same dict, with the failed molecules removed in place
    """
    for graph_type, tables in data_dict.items():
        nodes = tables["nodes"]
        edges = tables["edges"]

        missing = missing_columns(nodes, edges)
        if missing:
            raise SystemExit(
                f"FATAL [{graph_type}]: shard tables are missing {missing}. Were the graphs written after attach_true_labels() and the curvature pass? Node columns present: {list(nodes.columns)}; edge columns present: {list(edges.columns)}")

        flags = truth_flags(tables)
        if flags is None:
            print(f"[{graph_type}] WARNING: no labels_ok anywhere -- cannot drop molecules whose truth lookup failed; their sentinels will contaminate chain_nodes / nonbond_edges", flush=True)
            continue

        n_before = nodes["mol_id"].nunique()
        bad_ids = set(flags.index[flags["labels_ok"].astype(int) != 1])
        if bad_ids:
            tables["nodes"] = nodes[~nodes["mol_id"].isin(bad_ids)]
            tables["edges"] = edges[~edges["mol_id"].isin(bad_ids)]
            if "mols" in tables:
                mols = tables["mols"]
                tables["mols"] = mols[~mols["mol_id"].isin(bad_ids)]

        extra = ""
        if "smiles_match" in flags.columns:
            n_mismatch = int((flags["smiles_match"].astype(int) == 0).sum())
            extra = f", {n_mismatch} with smiles_match == 0 (kept; investigate if > 0)"
        print(f"[{graph_type}] {n_before} molecules, dropped {len(bad_ids)} with labels_ok != 1{extra}", flush=True)
    return data_dict


def load_qm9(graph_types, shard_glob, n_workers, max_files):
    """
    Load the QM9 shards, check their columns and drop failed molecules

    :param shard_glob: glob pattern matching the shard files
    :param n_workers: worker processes for reading
    :param max_files: only read this many files, or None for all
    :return: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    """
    files = sorted(glob.glob(shard_glob))
    if not files:
        raise SystemExit(f"FATAL: no shard files match {shard_glob}")
    print(f"{len(files)} shard files match {shard_glob}", flush=True)
    data_dict = load_shards(files, graph_types, n_workers, max_files)
    return check_truth_columns(data_dict)


