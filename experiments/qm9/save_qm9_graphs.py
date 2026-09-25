import os
import glob
import random

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import numpy as np
import traceback
import multiprocessing as mp
import concurrent.futures
import time

from src.qm9_graphs import read_qm9_xyz, construct_graphs, TARGET_NAMES, DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES
from src.qm9_true_bonds import load_sdf_bonds
from src.ricci_curvature import forman_ricci_curvature, ollivier_ricci_curvature

from utils.local_paths import local_paths
PATHS = local_paths()
SDF_PATH = os.path.join(PATHS.QM9_DOWNLAODED_DATA, "qm9", "gdb9.sdf")

# this is where the data is 
INPUT_DIR = PATHS.QM9_DOWNLAODED_DATA
OUT_DIR = PATHS.QM9_SHARD_GLOB
VERIFIED_IDS_PATH = os.path.join(PATHS.QM9_DOWNLAODED_DATA, "qm9", "verified_mol_ids.txt")

seed = 44
np.random.seed(seed)
random.seed(seed)

def load_verified_ids(path):
    '''
    Read the molecule ids that verify_qm9.py approved (SMILES agreed, neutral, not author-flagged)

    :param path: path to the id list, one integer per line, or None
    :return: set of mol_ids, or None when path is None, meaning "use every molecule"
    '''
    if path is None:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: verified id list not found at {path}\nRun verify_qm9.py first, or set VERIFIED_IDS_PATH = None to use every molecule.")
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))
    print(f"Loaded {len(ids)} verified molecule ids from {path}")
    return ids


def mol_id_from_filename(filepath):
    '''
    dsgdb9nsd_000042.xyz -> 42

    :param filepath: path to a QM9 xyz file
    :return: the molecule id as an int
    '''
    name = os.path.basename(filepath)
    digits = name.replace("dsgdb9nsd_", "").replace(".xyz", "")
    return int(digits)


def keep_only_verified(file_list, verified_ids):
    '''
    Drop any file whose molecule id is not in the verified set.

    :param file_list: xyz file paths
    :param verified_ids: set from load_verified_ids, or None to keep everything
    :return: the filtered list of paths
    '''
    if verified_ids is None:
        return file_list
    kept = []
    for filepath in file_list:
        try:
            if mol_id_from_filename(filepath) in verified_ids:
                kept.append(filepath)
        except ValueError:
            continue
    return kept


def make_or_load_splits(input_dir, split_dir, fractions, verified_ids):
    '''
    Shuffle all xyz files once (fixed seed) and cut into splits. The
    lists are saved to text files; if they already exist, reuse them so
    the split never changes between runs.

    :param input_dir: folder holding the QM9 .xyz files
    :param split_dir: where the qm9_<split>_files.txt lists are written
    :param fractions: split name : fraction, e.g. {"train": 0.8, ...};
                      the last split takes whatever is left over
    :param verified_ids: set from load_verified_ids, applied before splitting
    :return: dict of split name : list of file paths
    '''
    os.makedirs(split_dir, exist_ok=True)

    existing = {}
    all_found = True
    for split in fractions:
        list_path = os.path.join(split_dir, "qm9_" + split + "_files.txt")
        if os.path.exists(list_path):
            paths = []
            with open(list_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        paths.append(line)
            existing[split] = paths
        else:
            all_found = False
    if all_found:
        for split in existing:
            print(f"Loaded cached split '{split}': {len(existing[split])} files")
        return existing

    all_files = sorted(glob.glob(os.path.join(input_dir, "*.xyz")))
    if len(all_files) == 0:
        raise SystemExit(f"FATAL: no .xyz files found in {input_dir}")
    print(f"Found {len(all_files)} xyz files in {input_dir}")

    all_files = keep_only_verified(all_files, verified_ids)
    print(f"After keeping only verified molecules: {len(all_files)}")
    if len(all_files) == 0:
        raise SystemExit("FATAL: no molecules left after the verified filter")

    rng = random.Random(seed)
    rng.shuffle(all_files)

    splits = {}
    start = 0
    split_names = list(fractions)
    for i, split in enumerate(split_names):
        if i == len(split_names) - 1:
            chunk = all_files[start:]
        else:
            n = int(len(all_files) * fractions[split])
            chunk = all_files[start:start + n]
            start = start + n
        splits[split] = chunk

        list_path = os.path.join(split_dir, "qm9_" + split + "_files.txt")
        with open(list_path, "w") as f:
            for p in chunk:
                f.write(p + "\n")
        print(f"Split '{split}': {len(chunk)} files -> {list_path}")

    return splits


def process_single_mol_all_graphs(mol, graph_types):
    '''
    Build every graph type for ONE molecule, compute the curvatures, and
    pack everything into plain numpy arrays ready for saving.

    :param mol: the mol dict from read_qm9_xyz
    :param graph_types: which constructions to build
    :return: dict of graph_type -> (node_data, edge_data, graph_attrs),
             or None if this molecule should be skipped
    '''
    mol_payload = {}

    for graph_type in graph_types:
        try:
            G = construct_graphs(mol, graph_type=graph_type)

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                return None

            G_orc = ollivier_ricci_curvature(G)
            G_frc = forman_ricci_curvature(G)

            nodes = list(G.nodes())
            N = len(nodes)
            edges = list(G.edges())

            Z = np.zeros(N, dtype=np.int32)
            mass = np.zeros(N, dtype=np.float32)
            mulliken = np.zeros(N, dtype=np.float32)
            pos_x = np.zeros(N, dtype=np.float32)
            pos_y = np.zeros(N, dtype=np.float32)
            pos_z = np.zeros(N, dtype=np.float32)
            orc_nodes = np.zeros(N, dtype=np.float32)
            frc_nodes = np.zeros(N, dtype=np.float32)
            in_ring = np.zeros(N, dtype=np.int8)
            true_degree = np.full(N, -1, dtype=np.int32)
            heavy_degree = np.full(N, -1, dtype=np.int32)
            is_hydrogen = np.zeros(N, dtype=np.int8)

            for i, node in enumerate(nodes):
                nd = G.nodes[node]
                Z[i] = nd.get("Z", 0)
                mass[i] = nd.get("mass", 0.0)
                mulliken[i] = nd.get("mulliken", 0.0)
                pos = nd.get("pos", (0.0, 0.0, 0.0))
                pos_x[i] = pos[0]
                pos_y[i] = pos[1]
                pos_z[i] = pos[2]

                edge_curv = []
                for neighbor in G_orc[node]:
                    edge_curv.append(G_orc[node][neighbor].get("ricciCurvature", 0.0))
                if len(edge_curv) > 0:
                    orc_nodes[i] = float(np.mean(edge_curv))
                frc_nodes[i] = G_frc.nodes[node].get("formanCurvature", 0.0)

                in_ring[i] = int(nd.get("in_ring", False))
                true_degree[i] = int(nd.get("true_degree", -1))
                heavy_degree[i] = int(nd.get("heavy_degree", -1))
                is_hydrogen[i] = int(nd.get("is_hydrogen", Z[i] == 1))

            node_data = {
                "node_id": np.array(nodes, dtype=np.int32),
                "Z": Z, "mass": mass,
                "mulliken": mulliken,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
                "orc_curvature": orc_nodes,
                "frc_curvature": frc_nodes,
                "in_ring": in_ring,
                "true_degree": true_degree,
                "heavy_degree": heavy_degree,
                "is_hydrogen": is_hydrogen,
            }

            edge_data = {}
            if len(edges) > 0:
                n_edges = len(edges)
                src = np.zeros(n_edges, dtype=np.int32)
                dst = np.zeros(n_edges, dtype=np.int32)
                orc_edges = np.zeros(n_edges, dtype=np.float32)
                frc_edges = np.zeros(n_edges, dtype=np.float32)
                true_bond = np.zeros(n_edges, dtype=np.int8)
                for i, edge in enumerate(edges):
                    u = edge[0]
                    v = edge[1]
                    src[i] = u
                    dst[i] = v
                    orc_edges[i] = G_orc[u][v].get("ricciCurvature", 0.0)
                    frc_edges[i] = G_frc[u][v].get("formanCurvature", 0.0)
                    true_bond[i] = int(G.edges[u, v].get("is_true_bond", False))
                edge_data["src"] = src
                edge_data["dst"] = dst
                edge_data["orc_edge_curvature"] = orc_edges
                edge_data["frc_edge_curvature"] = frc_edges
                edge_data["is_true_bond"] = true_bond

            graph_attrs = {
                "labels_ok": int(G.graph.get("labels_ok", 0)),
                "smiles_match": int(G.graph.get("smiles_match", -1)),
                "n_true_bonds": int(G.graph.get("n_true_bonds", 0)),
                "n_tp_bonds": int(G.graph.get("n_tp_bonds", 0)),
                "bond_precision": float(G.graph.get("bond_precision", 0.0)),
                "bond_recall": float(G.graph.get("bond_recall", 0.0)),
                "bond_f1": float(G.graph.get("bond_f1", 0.0)),
            }

            mol_payload[graph_type] = (node_data, edge_data, graph_attrs)

        except Exception as e:
            print(f"[mol {mol.get('mol_id', '?')} / {graph_type}] {repr(e)}", flush=True)
            traceback.print_exc()
            return None

    if len(mol_payload) == 0:
        return None

    return mol_payload


def process_and_save_files(args):
    '''
    One worker: read its list of xyz files, build all graphs per
    molecule, write everything into one .h5 shard.

    :param args: (shard_idx, file_chunk, graph_types, output_dir,
                  shard_prefix) — one tuple so it can be submitted directly
    :return: a status string for the main process to print
    '''
    shard_idx, file_chunk, graph_types, output_dir, shard_prefix = args

    if len(graph_types) == 0:
        return f"Shard {shard_idx} FAILED: graph_types list is empty"

    shard_filepath = os.path.join(output_dir, shard_prefix + "_" + str(shard_idx).zfill(4) + ".h5")
    tmp_filepath = shard_filepath + ".tmp"

    written_count = 0
    mols_seen = 0
    start_time = time.time()

    try:
        with h5py.File(tmp_filepath, "w", libver="latest") as f:
            graph_groups = {}
            for g_type in graph_types:
                graph_groups[g_type] = f.create_group(g_type)

            for filepath in file_chunk:
                mols_seen += 1
                if mols_seen % 500 == 0:
                    elapsed = time.time() - start_time
                    per_mol = elapsed / mols_seen
                    remaining = per_mol * (len(file_chunk) - mols_seen)
                    print(f"Shard {str(shard_idx).zfill(4)}: {mols_seen}/{len(file_chunk)} mols done ({round(per_mol, 3)} s/mol, ~{round(remaining / 3600, 1)} h left)", flush=True)

                mol = read_qm9_xyz(filepath)
                if mol is None:
                    continue

                payload = process_single_mol_all_graphs(mol, graph_types)
                if payload is None:
                    continue

                mol_id = mol['mol_id']

                for graph_type in payload:
                    node_data, edge_data, graph_attrs = payload[graph_type]

                    g_grp = graph_groups[graph_type]
                    m_grp = g_grp.create_group("mol_" + str(mol_id))

                    m_grp.attrs["mol_id"] = mol_id
                    m_grp.attrs["num_atoms"] = mol['num_atoms']
                    m_grp.attrs["smiles"] = mol['smiles']
                    for name in TARGET_NAMES:
                        m_grp.attrs[name] = mol['targets'][name]
                    for k in graph_attrs:
                        m_grp.attrs[k] = graph_attrs[k]

                    ng = m_grp.create_group("nodes")
                    for k in node_data:
                        ng.create_dataset(k, data=node_data[k])

                    eg = m_grp.create_group("edges")
                    for k in edge_data:
                        eg.create_dataset(k, data=edge_data[k])

                if len(payload) > 0:
                    written_count += 1

            f.flush()

        os.replace(tmp_filepath, shard_filepath)

    except Exception as e:
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except OSError:
                pass
        return f"Shard {shard_idx} FAILED ({repr(e)})\n{traceback.format_exc()}"

    return f"Shard {str(shard_idx).zfill(4)} complete. Saved {written_count} molecules -> {shard_filepath}"


def run_pipeline(file_list, graph_types, output_dir, shard_prefix, n_workers=200):
    '''
    Split the file list into chunks and give each chunk to a worker.

    :param file_list: xyz file paths for one split
    :param graph_types: which constructions to build
    :param output_dir: where the .h5 shards go; created if missing
    :param shard_prefix: filename prefix, e.g. "all_train"
    :param n_workers: upper bound on processes; capped by CPU count and by how many files there are
    :return: None; shard status lines are printed as they finish
    '''
    os.makedirs(output_dir, exist_ok=True)

    if len(graph_types) == 0:
        raise ValueError("graph_types is empty — qm9_graph_configs.yaml probably did not load")
    if len(file_list) == 0:
        print("No files to process — nothing to do.", flush=True)
        return

    cpu = os.cpu_count() or 1
    n_workers = min(n_workers, cpu, len(file_list))
    print(f"Using {n_workers} workers (machine has {cpu} CPUs)", flush=True)

    chunk_size = max(1, int(np.ceil(len(file_list) / n_workers)))
    print(f"Splitting {len(file_list)} molecules into shards of size {chunk_size}", flush=True)

    tasks = []
    shard_idx = 0
    for i in range(0, len(file_list), chunk_size):
        tasks.append((shard_idx, file_list[i:i + chunk_size], graph_types, output_dir, shard_prefix))
        shard_idx += 1

    print(f"Starting {n_workers} workers over {len(tasks)} shards...", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers,
                                                mp_context=mp.get_context("fork")) as executor:
        future_to_idx = {}
        for task in tasks:
            fut = executor.submit(process_and_save_files, task)
            future_to_idx[fut] = task[0]

        for fut in concurrent.futures.as_completed(future_to_idx):
            s_idx = future_to_idx[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"Shard {s_idx} crashed hard: {repr(e)}", flush=True)
                traceback.print_exc()


def verify_saved_shards(output_dir, shard_prefix, expected_mols=None):
    '''
    Open the saved shards and check they contain what they should.

    for the SDF truth source + guard fix:
      - labels_ok / smiles_match are now expected to be EXACTLY 0
        failures; any hit is a regression and the offending mol_ids are
        printed (cheap, since the expected count is zero).
      - NEW INVARIANT: fully_connected recall must be exactly 1.0 for
        every molecule. The fully connected graph contains every atom
        pair, so every true bond is necessarily among its edges — if
        recall < 1.0 anywhere, the truth labels themselves are broken
        (misaligned bonds / wrong mol_id lookup), independent of any
        construction. Free end-to-end test of the labeling path.

    :param output_dir: folder holding the shards
    :param shard_prefix: filename prefix the shards were written with
    :param expected_mols: how many molecules went in, or None to skip that comparison
    :return: None; the report is printed
    '''
    bar = "=" * 50
    print(f"\n{bar}")
    print(f"Starting data validation check ({shard_prefix})")
    print(bar, flush=True)

    shard_files = sorted(glob.glob(os.path.join(output_dir, shard_prefix + "_*.h5")))
    leftover_tmp = sorted(glob.glob(os.path.join(output_dir, shard_prefix + "_*.h5.tmp")))

    if len(leftover_tmp) > 0:
        print(f"  WARNING: {len(leftover_tmp)} leftover .tmp file(s) — those shards probably crashed mid-write")
    if len(shard_files) == 0:
        print(f"  ERROR: no shard files found in {output_dir}")
        return

    print(f"Found {len(shard_files)} completed shard files.")

    try:
        with h5py.File(shard_files[0], "r") as f:
            gtypes = list(f.keys())
            print(f"Graph types present ({len(gtypes)}): {gtypes}")
            if len(gtypes) == 0:
                print("WARNING: file has no graph types!")
                return
            mols = list(f[gtypes[0]].keys())
            print(f" Molecules in this shard: {len(mols)}")
            if len(mols) > 0:
                m = f[gtypes[0]][mols[0]]
                print(f"Sample {mols[0]}: smiles={m.attrs.get('smiles', '?')} num_atoms={m.attrs.get('num_atoms', '?')} gap={m.attrs.get('gap', '?')} bond_f1={m.attrs.get('bond_f1', 'MISSING')} labels_ok={m.attrs.get('labels_ok', 'MISSING')}")
                if "nodes" in m:
                    print(f" nodes/Z shape: {m['nodes']['Z'].shape}  nodes/orc_curvature shape: {m['nodes']['orc_curvature'].shape}")
                    for col in ["in_ring", "true_degree", "heavy_degree", "is_hydrogen"]:
                        if col not in m["nodes"]:
                            print(f" WARNING: truth column nodes/{col} MISSING")
                if "edges" in m and "src" in m["edges"]:
                    print(f" edges/src shape: {m['edges']['src'].shape}")
                    if "is_true_bond" not in m["edges"]:
                        print("    WARNING: truth column edges/is_true_bond MISSING")
    except Exception as e:
        print(f"  FAILED TO READ FILE: {e}")

    total = 0
    per_type = {}
    corrupted = 0
    empty = 0
    bad_labels_ids = []
    bad_smiles_ids = []
    bad_recall_ids = []
    fc_checked = 0

    for sf in shard_files:
        try:
            with h5py.File(sf, "r") as f:
                gtypes = list(f.keys())
                if len(gtypes) == 0:
                    empty += 1
                    continue
                total += len(f[gtypes[0]].keys())
                for mk in f[gtypes[0]]:
                    a = f[gtypes[0]][mk].attrs
                    if int(a.get("labels_ok", 0)) != 1:
                        bad_labels_ids.append(int(a.get("mol_id", -1)))
                    if int(a.get("smiles_match", -1)) != 1:
                        bad_smiles_ids.append(int(a.get("mol_id", -1)))
                if "fully_connected" in f:
                    for mk in f["fully_connected"]:
                        a = f["fully_connected"][mk].attrs
                        fc_checked += 1
                        if float(a.get("bond_recall", -1.0)) != 1.0:
                            bad_recall_ids.append(int(a.get("mol_id", -1)))
                for gt in gtypes:
                    per_type[gt] = per_type.get(gt, 0) + len(f[gt].keys())
        except Exception:
            corrupted += 1

    consistent = len(set(per_type.values())) <= 1
    n_bad_labels = len(bad_labels_ids)
    n_bad_smiles = len(bad_smiles_ids)
    n_bad_recall = len(bad_recall_ids)

    consistent_note = "" if consistent else "  <- MISMATCH!"
    labels_note = " (OK — expected 0)" if n_bad_labels == 0 else "  <- REGRESSION: expected 0 after the guard fix. mol_ids: " + str(sorted(bad_labels_ids)[:50])
    smiles_note = " (OK — expected 0)" if n_bad_smiles == 0 else "  <- SDF bonds do not match either xyz SMILES column — investigate. mol_ids: " + str(sorted(bad_smiles_ids)[:50])

    print("-" * 50)
    print("VALIDATION SUMMARY:")
    print(f" - Completed shard files:{len(shard_files)}")
    print(f" - Corrupted files:{corrupted}")
    print(f" - Empty files: {empty}")
    print(f" - Total molecules written: {total}")
    print(f" - Per-graph-type counts: {per_type}")
    print(f" - Graph types consistent: {consistent}{consistent_note}")
    print(f" - labels_ok != 1: {n_bad_labels}{labels_note}")
    print(f" - smiles_match != 1: {n_bad_smiles}{smiles_note}")

    if fc_checked > 0:
        recall_note = "  (OK — truth labels internally consistent)" if n_bad_recall == 0 else "  <- TRUTH LABELS BROKEN (bond misalignment / wrong lookup). mol_ids: " + str(sorted(bad_recall_ids)[:50])
        print(f"- fully_connected recall == 1.0: checked {fc_checked} molecules, {n_bad_recall} violations{recall_note}")
    else:
        print(" - fully_connected recall invariant: SKIPPED (fully_connected not in this run's graph types)")

    if expected_mols is not None:
        diff = expected_mols - total
        if diff == 0:
            print(f"- Expected molecules: {expected_mols}  (OK)")
        else:
            print(f"- Expected molecules:{expected_mols}  ({diff} fewer — unreadable files or same-molecules drops)")
    print(f"{bar}\n", flush=True)


if __name__ == '__main__':
    print(f"os.cpu_count(): {os.cpu_count()}")

    graph_types = list(DEFAULT_GRAPH_TYPES) + list(SWEEP_GRAPH_TYPES)
    RUN_NAME = "all"

    print(f"{RUN_NAME} graph types ({len(graph_types)}): {graph_types}")
    if len(graph_types) == 0:
        raise SystemExit("FATAL: no graph types loaded. Stopping.")

   

    SPLIT_FRACTIONS = {"train": 0.8, "valid": 0.1, "test": 0.1}

    n_workers = 170
    MAX_MOLS = None

    verified_ids = load_verified_ids(VERIFIED_IDS_PATH)
    load_sdf_bonds(SDF_PATH)
    splits = make_or_load_splits(INPUT_DIR, OUT_DIR, SPLIT_FRACTIONS, verified_ids)

    for split in splits:
        file_list = splits[split]
        if MAX_MOLS is not None:
            file_list = file_list[:MAX_MOLS]

        save_path = os.path.join(OUT_DIR, "temp_" + split)
        print(f"\n {split}: {len(file_list)} molecules", flush=True)

        start_time = time.time()
        run_pipeline(
            file_list=file_list,
            graph_types=graph_types,
            output_dir=save_path,
            shard_prefix=RUN_NAME + "_" + split,
            n_workers=n_workers,
        )
        elapsed = time.time() - start_time
        print(f"\nTime elapsed for {split}: {round(elapsed, 2)} seconds")

        verify_saved_shards(save_path, shard_prefix=RUN_NAME + "_" + split, expected_mols=len(file_list))

    print("All tasks finished successfully!")
