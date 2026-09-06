import concurrent.futures
import multiprocessing as mp
import h5py
import numpy as np
import pandas as pd

def load_graph_to_arrays(args):
    '''
    read one .h5 file and return per-graph type dicts
    :param args:
    :return:
    '''
    h5_path, file_pos, graph_types = args # file_pos added for unique jet ids
    out = {}

    with h5py.File(h5_path, "r") as f:
        for graph_type in graph_types:
            # dictionary to store properties of a graph's components
            bucket = {"node_comps": {},
                      "edge_comps": {},
                      "node_jet_id": [], # empty lists to store ids linking nodes to specific jets
                      "node_flavor": [], #keeping track of the jets flavor, but mapping it directly to the nodes belonging to that jet
                      "edge_jet_id": [], # empty lists to store ids linking edges to specific jets
                      "edge_flavor": []} #keeping track of the jet's flavor, but mapping it directly to the edges belonging to that jet
            out[graph_type] = bucket

            if graph_type not in f:
                continue
            graph_group = f[graph_type]

            for jet_name in graph_group:
                jet_group = graph_group[jet_name]
                flavor = int(jet_group.attrs.get("label", -1))

                # jet numbering restarts per shard, so make the id globally unique across files (file_pos comes from the sorted file list)
                jet_id = file_pos * 10_000_000 + int(jet_name.split("_", 1)[1])

                if "nodes" in jet_group:
                    nodes = jet_group["nodes"]
                    first_key = next(iter(nodes.keys()), None)
                    if first_key is not None:
                        n = nodes[first_key].shape[0]
                        if n > 0:
                            for k in nodes:
                                bucket["node_comps"].setdefault(k, []).append(nodes[k][:])
                            bucket["node_jet_id"].append(np.full(n, jet_id, dtype=np.int64))
                            bucket["node_flavor"].append(np.full(n, flavor, dtype=np.int16))

                if "edges" in jet_group:
                    edges = jet_group["edges"]
                    first_key = next(iter(edges.keys()), None)
                    if first_key is not None:
                        m = edges[first_key].shape[0]
                        if m > 0:
                            for k in edges:
                                bucket["edge_comps"].setdefault(k, []).append(edges[k][:])
                            bucket["edge_jet_id"].append(np.full(m, jet_id, dtype=np.int64))
                            bucket["edge_flavor"].append(np.full(m, flavor, dtype=np.int16))

    # Pre-concatenate inside the worker so we ship one ndarray per field
    compact = {}
    for graph_type, bucket in out.items():
        compact[graph_type] = {
            "node_comps": {k: np.concatenate(v) for k, v in bucket["node_comps"].items()},
            "edge_comps": {k: np.concatenate(v) for k, v in bucket["edge_comps"].items()},
            "node_jet_id":  np.concatenate(bucket["node_jet_id"])  if bucket["node_jet_id"]  else np.empty(0, dtype=np.int64),
            "node_flavor":  np.concatenate(bucket["node_flavor"])  if bucket["node_flavor"]  else np.empty(0, dtype=np.int16),
            "edge_jet_id":  np.concatenate(bucket["edge_jet_id"])  if bucket["edge_jet_id"]  else np.empty(0, dtype=np.int64),
            "edge_flavor":  np.concatenate(bucket["edge_flavor"])  if bucket["edge_flavor"]  else np.empty(0, dtype=np.int16),
        }
    return h5_path, compact


def load_all_files_parallel(h5_files,graph_types, n_workers):
    '''
    Parallel reader so n_workers read independently and then merge
    :param h5_files:
    :param graph_types:
    :param n_workers:per-graph-type dicts
    :return:
    '''

    print(f"Reading {len(h5_files)} files with {n_workers} workers...", flush=True)

    # Parent-side accumulators: one list per (graph_type, comp) holding worker-returned pre-concatenated arrays
    node_acc = {g: {} for g in graph_types}
    edge_acc = {g: {} for g in graph_types}
    node_jet_id  = {g: [] for g in graph_types}
    node_flavor  = {g: [] for g in graph_types}
    edge_jet_id  = {g: [] for g in graph_types}
    edge_flavor  = {g: [] for g in graph_types}

    tasks = [(fp, i, graph_types) for i, fp in enumerate(h5_files)]

    # using spawn so we dont drag any parent thread state into the workers
    # Read-only hdf5 is safe under spawn
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.get_context
    # when setting up multiprocessing context
    # this explicitly sets the method for child process to start at spawn
    ctx = mp.get_context("spawn")
    # initializing the pool process
    # which creates a pool of worker processes -- so worker processes (executor) that will run tasks in parallel
    # mp_context=ctx: this forces the executor to use the `spawn` context
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        # this launches all task in the background  and maps to file paths
        futures = {executor.submit(load_graph_to_arrays, t): t[0] for t in tasks}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            shard_path = futures[fut]
            try:
                _, file = fut.result()
            except Exception as e:
                print(f" Shard FAILED: {shard_path}: {e!r}", flush=True)
                continue
            #looping through categories of graphs
            for g in graph_types:
                gb = file[g]
                for k, arr in gb["node_comps"].items():
                    node_acc[g].setdefault(k, []).append(arr)
                for k, arr in gb["edge_comps"].items():
                    edge_acc[g].setdefault(k, []).append(arr)
                if gb["node_jet_id"].size:
                    node_jet_id[g].append(gb["node_jet_id"])
                    node_flavor[g].append(gb["node_flavor"])
                if gb["edge_jet_id"].size:
                    edge_jet_id[g].append(gb["edge_jet_id"])
                    edge_flavor[g].append(gb["edge_flavor"])

            done += 1
            if done % 10 == 0 or done == len(h5_files):
                print(f"  merged {done}/{len(h5_files)} files", flush=True)

    # Build one df per graph type at the end
    final = {}
    for g in graph_types:
        if node_acc[g]:
            cols = {k: np.concatenate(v) for k, v in node_acc[g].items()}
            cols["jet_id"] = np.concatenate(node_jet_id[g]) if node_jet_id[g] else np.empty(0, dtype=np.int64)
            cols["flavor"] = np.concatenate(node_flavor[g]) if node_flavor[g] else np.empty(0, dtype=np.int16)
            if "truth_vertex_idx" in cols:
                # 0 is the truth PV (JetSet docs), so SV is strictly > 0;
                # negative means no truth match -- exclude via is_matched in SV/PV tasks
                tvi = cols["truth_vertex_idx"]
                cols["is_sv_node"] = (tvi > 0).astype(np.int8)
                cols["is_matched"] = (tvi >= 0).astype(np.int8)
            nodes_df = pd.DataFrame(cols)
        else:
            nodes_df = pd.DataFrame()

        if edge_acc[g]:
            cols = {k: np.concatenate(v) for k, v in edge_acc[g].items()}
            cols["jet_id"] = np.concatenate(edge_jet_id[g]) if edge_jet_id[g] else np.empty(0, dtype=np.int64)
            cols["flavor"] = np.concatenate(edge_flavor[g]) if edge_flavor[g] else np.empty(0, dtype=np.int16)
            edges_df = pd.DataFrame(cols)
        else:
            edges_df = pd.DataFrame()

        final[g] = {"nodes": nodes_df, "edges": edges_df}

    return final


def check_jet_ids(final, n_files, reference_graph_type=None):
    '''
    Verify unique jet IDs and multi-file stride coverage.

    :param final: dict graph_type -> {"nodes": DataFrame, "edges": DataFrame}
    :param n_files: number of shards loaded (expected file strides)
    :param reference_graph_type: graph type to inspect; defaults to the first key
    :return: number of unique jet IDs
    '''
    g = reference_graph_type or list(final)[0]
    jid = final[g]["nodes"]["jet_id"]
    n_unique = jid.nunique()
    strides_used = (jid // 10_000_000).nunique()
    print(f"[{g}] unique jets: {n_unique:,} | file strides represented: {strides_used}/{n_files}")

    if n_files > 1 and strides_used == 1:
        print("  WARNING: all IDs in one stride -- file_pos not applied?")
    return n_unique


def validate_sv_labels(nodes_df):
    '''
    Compute normalized crosstab of is_sv_node versus origin_label on matched tracks.

    :param nodes_df: node table containing is_sv_node, is_matched, and origin_label
    :return: normalized crosstab DataFrame, or None if required columns are missing
    '''
    needed = ["is_sv_node", "is_matched", "origin_label"]
    missing = [col for col in needed if col not in nodes_df.columns]
    if missing:
        print(f"validate_sv_labels: missing columns {sorted(missing)}")
        return None

    m = nodes_df[nodes_df["is_matched"] == 1]
    ct = pd.crosstab(m["is_sv_node"], m["origin_label"], normalize="index")
    print(ct.round(3).to_string())
    return ct