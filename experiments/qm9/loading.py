import glob
import time
from multiprocessing import Pool
import h5py
import numpy as np
import pandas as pd

from experiments.qm9.config import CURVATURES, EDGE_COLS, NODE_COLS


def read_shard_file(job):
    """
    Read every molecule of one graph type from one shard file. Runs in a
    worker process.

    :param job: tuple (path, graph_type)
    :return: tuple (node_cols, edge_cols, mol_rows), where the first two
        map a column name to a list of per-molecule arrays and mol_rows is
        a list of molecule attribute dicts
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

            node_group = mol["nodes"]
            n_atoms = 0
            for column in node_group:
                values = node_group[column][:]
                n_atoms = len(values)
                node_cols.setdefault(column, []).append(values)
            node_cols.setdefault("mol_id", []).append(np.full(n_atoms, mol_id, dtype=np.int64))
            # Without a node_id column, row order is atom order.
            if "node_id" not in node_group:
                node_cols.setdefault("node_id", []).append(np.arange(n_atoms, dtype=np.int64))

            edge_group = mol["edges"]
            if "src" in edge_group:
                n_edges = 0
                for column in edge_group:
                    values = edge_group[column][:]
                    n_edges = len(values)
                    edge_cols.setdefault(column, []).append(values)
                edge_cols.setdefault("mol_id", []).append( np.full(n_edges, mol_id, dtype=np.int64))

            attributes = {}
            for key in mol.attrs:
                attributes[key] = mol.attrs[key]
            mol_rows.append(attributes)
    return node_cols, edge_cols, mol_rows


def columns_to_frame(columns):
    """
    Concatenate the per molecule arrays of every column into one DataFrame.

    :param columns: {column name: list of arrays} from read_shard_file
    :return: DataFrame, empty if columns is empty
    """
    if not columns:
        return pd.DataFrame()

    lengths = {}
    for name in columns:
        total = 0
        for values in columns[name]:
            total += len(values)
        lengths[name] = total
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"FATAL: shard columns have different total lengths -- some molecules lack a column that others have: {lengths}")

    joined = {}
    for name in columns:
        joined[name] = np.concatenate(columns[name])
    df = pd.DataFrame(joined)

    # h5py returns strings as bytes.
    for name in df.columns:
        if df[name].dtype != object or len(df) == 0:
            continue
        if isinstance(df[name].iloc[0], bytes):
            df[name] = df[name].str.decode("utf-8")
    return df


def load_shards(file_list, graph_types, n_workers=1, max_files=None):
    """
    Read the shards into one set of tables per graph type. Every
    (file, graph type) pair is a separate job in one worker pool.

    :param file_list: shard paths
    :param graph_types: constructions to read from each shard
    :param n_workers: worker processes; 1 runs everything in this process
    :param max_files: only read this many files, or None for all
    :return: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    """
    if max_files is not None:
        file_list = file_list[:max_files]

    jobs = []
    for graph_type in graph_types:
        for path in file_list:
            jobs.append((path, graph_type))

    start_time = time.time()
    n_processes = min(n_workers, len(jobs))
    print(f"reading {len(file_list)} files x {len(graph_types)} graph types = {len(jobs)} jobs on {n_processes} workers", flush=True)

    results = []
    if n_workers > 1 and len(jobs) > 1:
        with Pool(n_processes) as pool:
            # imap returns the results in job order.
            for job_number, result in enumerate(pool.imap(read_shard_file, jobs), start=1):
                results.append(result)
                if job_number % 25 == 0 or job_number == len(jobs):
                    elapsed = time.time() - start_time
                    print(f"  {job_number}/{len(jobs)} jobs done ({elapsed:.0f} s)", flush=True)
    else:
        for job_number, job in enumerate(jobs, start=1):
            results.append(read_shard_file(job))
            elapsed = time.time() - start_time
            print(f"{job_number}/{len(jobs)} jobs done ({elapsed:.0f} s)", flush=True)

    grouped = {}
    for graph_type in graph_types:
        grouped[graph_type] = {"nodes": {}, "edges": {}, "mols": []}
    for job, result in zip(jobs, results):
        target = grouped[job[1]]
        job_nodes, job_edges, job_mols = result
        for name in job_nodes:
            target["nodes"].setdefault(name, []).extend(job_nodes[name])
        for name in job_edges:
            target["edges"].setdefault(name, []).extend(job_edges[name])
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
        mols = pd.DataFrame(parts["mols"])

        # Accept both writer conventions.
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

        data_dict[graph_type] = {"nodes": nodes, "edges": edges, "mols": mols}
        print(f"{graph_type}: {len(mols)} molecules, {len(nodes)} atoms, {len(edges)} edges", flush=True)
    print(f"load_shards: {time.time() - start_time:.0f} s", flush=True)
    return data_dict


def check_truth_columns(data_dict):
    """
    Check that the truth and curvature columns survived serialization and
    drop the molecules whose truth lookup failed (labels_ok != 1).

    :param data_dict: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    :return: the same dict, with the failed molecules removed in place
    """
    for graph_type in data_dict:
        tables = data_dict[graph_type]
        nodes = tables["nodes"]
        edges = tables["edges"]

        missing = []
        for col in NODE_COLS:
            if col not in nodes.columns:
                missing.append(col)
        for col in EDGE_COLS:
            if col not in edges.columns:
                missing.append(col)
        if missing:
            raise SystemExit(
                f"FATAL [{graph_type}]: shard tables are missing {missing}. Were the graphs "
                f"written after attach_true_labels() and the curvature pass? Node columns present: {list(nodes.columns)}; edge columns present: {list(edges.columns)}")

        if "mols" in tables and "labels_ok" in tables["mols"].columns:
            flags = tables["mols"].set_index("mol_id")
        elif "labels_ok" in nodes.columns:
            flags = nodes.drop_duplicates("mol_id").set_index("mol_id")
        else:
            print(f"[{graph_type}] WARNING: no labels_ok anywhere -- cannot drop molecules "
                  f"whose truth lookup failed; their sentinels will contaminate chain_nodes / nonbond_edges", flush=True)
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


def load_qm9(graph_types, shard_glob, n_workers=1, max_files=None):
    """
    Load the QM9 shards, check their columns and drop failed molecules.

    :param graph_types: constructions to load
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

