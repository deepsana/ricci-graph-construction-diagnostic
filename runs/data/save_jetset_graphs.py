import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import random
import numpy as np
import traceback
import multiprocessing
import concurrent.futures
import time
from data import read_single_jet_from_jetset
from src.graphs import construct_graphs, DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES
from src.ricci_curvature import forman_ricci_curvature, ollivier_ricci_curvature

seed = 44
np.random.seed(seed)
random.seed(seed)


def build_and_save_clean_jets(tracks_arr_all, jet_arr_all, truth_arr_all, save_indices_path="clean_jet_indices.npy"):
    '''
    Find which jets are usable and save their indices, so next time we can
    just load the saved list instead of checking again.

    :param tracks_arr_all: full tracks array from the raw h5 file
    :param jet_arr_all: full jets array from the raw h5 file
    :param truth_arr_all: full truth_hadrons array from the raw h5 file
    :param save_indices_path: where to cache the resulting indices
    :return: numpy array of clean jet indices
    '''
    if os.path.exists(save_indices_path):
        print(f"Loading cached indices from {save_indices_path}")
        return np.load(save_indices_path)

    clean_indices = []
    for i in range(len(jet_arr_all)):
        jet = read_single_jet_from_jetset(i, tracks_arr_all, jet_arr_all, truth_arr_all)
        if jet is None:
            continue
        clean_indices.append(i)

    clean_indices = np.array(clean_indices, dtype=np.int32)
    np.save(save_indices_path, clean_indices)
    print(f"Saved {len(clean_indices)} clean jet indices in {save_indices_path}")
    return clean_indices


GLOBAL_TRACKS = None
GLOBAL_JETS = None
GLOBAL_TRUTH = None


def init_worker(tracks, jets, truth):
    '''
    Runs once when a worker process starts. Sets up the global data arrays
    and pins the math libraries down to 1 thread inside this process.

    :param tracks: tracks array to store globally
    :param jets: jets array to store globally
    :param truth: truth array to store globally
    :return: nothing, sets the module-level globals
    '''
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    global GLOBAL_TRACKS, GLOBAL_JETS, GLOBAL_TRUTH
    GLOBAL_TRACKS = tracks
    GLOBAL_JETS = jets
    GLOBAL_TRUTH = truth


def process_single_jet_all_graphs(jet, jet_idx, graph_types):
    '''
    Build every graph type for ONE jet, compute the curvatures, and pack
    everything into plain numpy arrays ready for saving.

    :param jet: the jet dict returned by read_single_jet_from_jetset
    :param jet_idx: index of this jet, used for logging
    :param graph_types: list of graph type names to build for this jet
    :return: tuple (jet_idx, jet_payload), or None if this jet should be skipped
    '''
    jet_payload = {}

    for graph_type in graph_types:
        try:
            G = construct_graphs(jet, graph_type=graph_type)

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                return None

            G_orc = ollivier_ricci_curvature(G)
            G_frc = forman_ricci_curvature(G)

            nodes = list(G.nodes())
            N = len(nodes)
            edges = list(G.edges())
            E = len(edges)

            node_ids = np.array(nodes, dtype=np.int32)
            t_v_idx = np.zeros(N, dtype=np.int32)
            t_o_lbl = np.zeros(N, dtype=np.float32)
            pt_frac = np.zeros(N, dtype=np.float32)
            t_pt = np.zeros(N, dtype=np.float32)
            pos_deta = np.zeros(N, dtype=np.float32)
            pos_dphi = np.zeros(N, dtype=np.float32)
            pos_d0 = np.zeros(N, dtype=np.float32)
            pos_z0M = np.zeros(N, dtype=np.float32)
            pos_ltD = np.zeros(N, dtype=np.float32)
            pos_ltZ = np.zeros(N, dtype=np.float32)
            orc_nodes = np.zeros(N, dtype=np.float32)
            frc_nodes = np.zeros(N, dtype=np.float32)

            for i, node in enumerate(nodes):
                nd = G.nodes[node]
                t_v_idx[i] = nd.get("truth_vertex_idx", 0)
                t_o_lbl[i] = nd.get("truth_origin_label", 0.0)
                pt_frac[i] = nd.get("track_pt_frac", 0.0)
                t_pt[i] = nd.get("track_pt", 0.0)

                pos = nd.get("pos", (0.0, 0.0))
                pos_deta[i] = pos[0]
                pos_dphi[i] = pos[1]

                pos_imp = nd.get("pos_impact_parameter", (0.0, 0.0))
                pos_d0[i] = pos_imp[0]
                pos_z0M[i] = pos_imp[1]

                pos_lt = nd.get("pos_lifetime_coord", (0.0, 0.0))
                pos_ltD[i] = pos_lt[0]
                pos_ltZ[i] = pos_lt[1]

                edge_curv = [d.get("ricciCurvature", 0.0) for _, d in G_orc[node].items()]
                orc_nodes[i] = float(np.mean(edge_curv)) if edge_curv else 0.0
                frc_nodes[i] = G_frc.nodes[node].get("formanCurvature", 0.0)

            node_data = {
                "node_id": node_ids, "truth_vertex_idx": t_v_idx, "truth_origin_label": t_o_lbl,
                "track_pt_frac": pt_frac, "track_pt": t_pt, "pos_deta": pos_deta, "pos_dphi": pos_dphi,
                "pos_d0": pos_d0, "pos_z0SinTheta": pos_z0M, "pos_lifetimeD0": pos_ltD,
                "pos_lifetimeZ0": pos_ltZ, "orc_curvature": orc_nodes, "frc_curvature": frc_nodes,
            }

            edge_data = {}
            if E > 0:
                src_list = []
                dst_list = []
                orc_edge_list = []
                frc_edge_list = []
                for u, v in edges:
                    src_list.append(u)
                    dst_list.append(v)
                    orc_edge_list.append(G_orc[u][v].get("ricciCurvature", 0.0))
                    frc_edge_list.append(G_frc[u][v].get("formanCurvature", 0.0))

                edge_data["src"] = np.array(src_list, dtype=np.int32)
                edge_data["dst"] = np.array(dst_list, dtype=np.int32)
                edge_data["orc_edge_curvature"] = np.array(orc_edge_list, dtype=np.float32)
                edge_data["frc_edge_curvature"] = np.array(frc_edge_list, dtype=np.float32)

            attrs = {k: v for k, v in G.graph.items() if v is not None}
            jet_payload[graph_type] = (attrs, node_data, edge_data)

        except Exception as e:
            print(f"[jet {jet_idx} / {graph_type}] {repr(e)}", flush=True)
            traceback.print_exc()
            return None

    if len(jet_payload) == 0:
        return None

    return jet_idx, jet_payload


def process_and_save_files(args):
    '''
    One worker runs this. It processes its chunk of jets and writes them
    all into one .h5 shard file.

    :param args: tuple (shard_idx, index_chunk, graph_types, output_dir, shard_prefix)
    :return: a status string describing success or failure for this shard
    '''
    shard_idx, index_chunk, graph_types, output_dir, shard_prefix = args

    if len(graph_types) == 0:
        return f"Shard {shard_idx} FAILED: graph_types list is empty"

    shard_filepath = os.path.join(output_dir, f"{shard_prefix}_{str(shard_idx).zfill(4)}.h5")
    tmp_filepath = shard_filepath + ".tmp"

    graph_scalar_keys = {"label", "frac_from_b", "frac_from_c", "num_sv_tracks", "num_tracks", "max_Lxy", "max_d0_sig"}
    graph_array_keys = {"truth_Lxy", "truth_decayVertexZ", "truth_decayVertexDPhi"}

    written_count = 0
    jets_seen = 0
    start_time = time.time()

    try:
        with h5py.File(tmp_filepath, "w", libver="latest") as f:
            graph_groups = {g_type: f.create_group(g_type) for g_type in graph_types}

            for idx in index_chunk:
                idx = int(idx)
                jets_seen += 1
                if jets_seen % 500 == 0:
                    elapsed = time.time() - start_time
                    per_jet = elapsed / jets_seen
                    remaining = per_jet * (len(index_chunk) - jets_seen)
                    print(f"Shard {str(shard_idx).zfill(4)}: {jets_seen}/{len(index_chunk)} jets done. ({round(per_jet, 2)} s/jet, ~{round(remaining / 3600, 1)} h left)", flush=True)

                jet = read_single_jet_from_jetset(idx, GLOBAL_TRACKS, GLOBAL_JETS, GLOBAL_TRUTH)
                if jet is None:
                    continue

                result = process_single_jet_all_graphs(jet, idx, graph_types)
                if result is None:
                    continue

                jet_idx, jet_payload = result

                for graph_type in jet_payload:
                    attrs, node_data, edge_data = jet_payload[graph_type]

                    g_grp = graph_groups[graph_type]
                    j_grp = g_grp.create_group(f"jet_{jet_idx}")

                    for k in graph_scalar_keys:
                        if k in attrs and attrs[k] is not None:
                            j_grp.attrs[k] = attrs[k]

                    j_grp.attrs["jet_pt"] = jet["jet"]["pt"]
                    j_grp.attrs["jet_eta"] = jet["jet"]["eta"]
                    j_grp.attrs["jet_phi"] = jet["jet"]["phi"]
                    j_grp.attrs["frac_sv"] = jet["physical_substructure"]["frac_sv"]
                    j_grp.attrs["jet_width"] = jet["physical_substructure"]["jet_width"]
                    j_grp.attrs["num_vertices"] = jet["physical_substructure"]["num_vertices"]
                    j_grp.attrs["num_sv_vertices"] = jet["physical_substructure"]["num_sv_vertices"]

                    for k in graph_array_keys:
                        if k in attrs and attrs[k] is not None:
                            j_grp.create_dataset(k, data=np.asarray(attrs[k], dtype=np.float32))

                    ng = j_grp.create_group("nodes")
                    for k in node_data:
                        ng.create_dataset(k, data=node_data[k])

                    eg = j_grp.create_group("edges")
                    for k in edge_data:
                        eg.create_dataset(k, data=edge_data[k])

                if len(jet_payload) > 0:
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

    return f"Shard {str(shard_idx).zfill(4)} complete. Saved {written_count} clean jets to {shard_filepath}"


def run_pipeline_hpc(clean_indices, tracks_arr, jet_arr, truth_arr, graph_types, output_dir, shard_prefix, n_workers=200, mp_start="fork"):
    '''
    Split the jets into chunks and give each chunk to a worker process.

    :param clean_indices: array of jet indices that passed the cleanliness check
    :param tracks_arr: full tracks array
    :param jet_arr: full jets array
    :param truth_arr: full truth array
    :param graph_types: list of graph type names to build
    :param output_dir: folder to write shard files into
    :param shard_prefix: filename prefix for the shard files
    :param n_workers: how many worker processes to use
    :param mp_start: multiprocessing start method, "fork" or "spawn"
    :return: nothing, writes shard files to output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)

    if len(graph_types) == 0:
        raise ValueError("graph_types is empty — nothing to build")

    if len(clean_indices) == 0:
        print("No clean indices to process — nothing to do.", flush=True)
        return

    cpu = os.cpu_count() or 1
    n_workers = min(n_workers, cpu, len(clean_indices))
    print(f"Using {n_workers} workers (machine has {cpu} CPUs)", flush=True)

    chunk_size = max(1, int(np.ceil(len(clean_indices) / n_workers)))
    print(f"Splitting {len(clean_indices)} jets into shards of size {chunk_size}", flush=True)

    tasks = []
    shard_idx = 0
    for i in range(0, len(clean_indices), chunk_size):
        index_chunk = clean_indices[i:i + chunk_size]
        tasks.append((shard_idx, index_chunk, graph_types, output_dir, shard_prefix))
        shard_idx += 1

    print(f"Starting {n_workers} workers over {len(tasks)} shards...", flush=True)
    ctx = multiprocessing.get_context(mp_start)

    pool_kwargs = dict(max_workers=n_workers, mp_context=ctx)
    if mp_start == "spawn":
        # "spawn" starts fresh processes, so each one needs its own copy of the data -> pass it through the initializer
        pool_kwargs["initializer"] = init_worker
        pool_kwargs["initargs"] = (tracks_arr, jet_arr, truth_arr)
    else:
        # "fork" copies the parent process, so if we set the globals here
        # the children automatically get them for free
        init_worker(tracks_arr, jet_arr, truth_arr)

    with concurrent.futures.ProcessPoolExecutor(**pool_kwargs) as executor:
        future_to_idx = {}
        for task in tasks:
            fut = executor.submit(process_and_save_files, task)
            future_to_idx[fut] = task[0]

        for fut in concurrent.futures.as_completed(future_to_idx):
            shard_idx = future_to_idx[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                print(f"Shard {shard_idx} crashed hard: {repr(e)}", flush=True)
                traceback.print_exc()


def verify_saved_shards(output_dir, shard_prefix, expected_jets=None):
    '''
    After a run, open the saved files and check they actually contain what
    they are supposed to contain.

    :param output_dir: folder the shard files were written to
    :param shard_prefix: filename prefix the shard files use
    :param expected_jets: if given, compare the total jet count against this
    :return: nothing, just prints a validation report
    '''
    print("\n" + "=" * 50)
    print(f"STARTING DATA VALIDATION CHECK ({shard_prefix})")
    print("=" * 50, flush=True)

    shard_files = sorted(glob.glob(os.path.join(output_dir, f"{shard_prefix}_*.h5")))
    leftover_tmp = sorted(glob.glob(os.path.join(output_dir, f"{shard_prefix}_*.h5.tmp")))

    if len(leftover_tmp) > 0:
        print(f"  WARNING: found {len(leftover_tmp)} leftover .tmp files — those shards probably crashed mid-write:")
        for t in leftover_tmp:
            print(f"      {t}")

    if len(shard_files) == 0:
        print(f"  ERROR: no shard files found in {output_dir}")
        return

    print(f"Found {len(shard_files)} completed shard files.")

    sample_file = shard_files[0]
    print(f"\nChecking sample file: {sample_file}")
    try:
        with h5py.File(sample_file, "r") as f:
            graph_types_found = list(f.keys())
            print(f"   Graph types present: {graph_types_found}")
            if len(graph_types_found) == 0:
                print("  WARNING: file has no graph types!")
                return

            test_g_type = graph_types_found[0]
            saved_jets = list(f[test_g_type].keys())
            print(f"  Jets in this shard (under '{test_g_type}'): {len(saved_jets)}")

            if len(saved_jets) > 0:
                jet_grp = f[test_g_type][saved_jets[0]]
                print(f"\n  Sample jet {saved_jets[0]}:")

                for ak in ("jet_pt", "jet_eta", "jet_phi", "label", "frac_sv", "jet_width", "num_tracks", "max_Lxy"):
                    print(f"    - attr '{ak}': {jet_grp.attrs.get(ak, 'MISSING')}")

                if "nodes" in jet_grp:
                    ng = jet_grp["nodes"]
                    if "track_pt" in ng:
                        print(f"    - nodes/track_pt shape: {ng['track_pt'].shape}")
                    if "orc_curvature" in ng:
                        print(f"    - nodes/orc_curvature shape: {ng['orc_curvature'].shape}")
                else:
                    print("    - WARNING: 'nodes' group MISSING")

                if "edges" in jet_grp:
                    eg = jet_grp["edges"]
                    if "src" in eg:
                        print(f"    - edges/src shape: {eg['src'].shape}")
                else:
                    print("    - WARNING: 'edges' group MISSING")

                for tk in ("truth_Lxy", "truth_decayVertexZ", "truth_decayVertexDPhi"):
                    if tk in jet_grp:
                        print(f"    - truth dataset '{tk}' shape: {jet_grp[tk].shape}")
                    else:
                        print(f"    - WARNING: truth dataset '{tk}' MISSING")

    except Exception as e:
        print(f"  FAILED TO READ FILE: {e}")

    print("\nCounting jets across all files...")
    corrupted_files = 0
    empty_files = 0
    total_jets = 0
    per_type_counts = {}

    for sf in shard_files:
        try:
            with h5py.File(sf, "r") as f:
                gtypes = list(f.keys())
                if len(gtypes) == 0:
                    empty_files += 1
                    continue
                total_jets += len(f[gtypes[0]].keys())
                for gt in gtypes:
                    per_type_counts[gt] = per_type_counts.get(gt, 0) + len(f[gt].keys())
        except Exception:
            print(f"  File corrupted or unreadable: {sf}")
            corrupted_files += 1

    counts = set(per_type_counts.values())
    types_consistent = len(counts) <= 1

    print("-" * 50)
    print("VALIDATION SUMMARY:")
    print(f" - Completed shard files: {len(shard_files)}")
    print(f" - Corrupted/unreadable files: {corrupted_files}")
    print(f" - Empty files: {empty_files}")
    print(f" - Leftover .tmp files:{len(leftover_tmp)}")
    print(f" - Total unique jets written:  {total_jets}")
    print(f" - Per-graph-type counts: {per_type_counts}")
    print(f" - Graph types consistent: {types_consistent}")
    if expected_jets is not None:
        diff = expected_jets - total_jets
        if diff == 0:
            print(f"  - Expected jets: {expected_jets}  (OK)")
        else:
            print(f"  - Expected jets:  {expected_jets}  ({diff} fewer — dropped by the same jets rule)")
    print("=" * 50 + "\n", flush=True)


if __name__ == '__main__':
    print("os.cpu_count():", os.cpu_count())
    print("multiprocessing.cpu_count():", multiprocessing.cpu_count())

    graph_types = list(SWEEP_GRAPH_TYPES)
    RUN_NAME = "sweep"

    print(f"{RUN_NAME} graph types ({len(graph_types)}): {graph_types}")

    if len(graph_types) == 0:
        raise SystemExit("FATAL: graph_types list is empty. Stopping.")

    OUT_DIR = "/media/mule/scratch/JetSet/07-21/"

    SPLITS = {
        "train": ("/media/mule/scratch/JetSet/mc-flavtag-ttbar-small-train.h5",
                  "/media/mule/scratch/JetSet/07-21/temp_train/",
                  "/media/mule/scratch/JetSet/07-21/temp_train/small_train_clean_jet_indices.npy"),
        "valid": ("/media/mule/scratch/JetSet/mc-flavtag-ttbar-small-valid.h5",
                  "/media/mule/scratch/JetSet/07-21/temp_valid/",
                  "/media/mule/scratch/JetSet/07-21/temp_valid/small_valid_clean_jet_indices.npy"),
        "test":  ("/media/mule/scratch/JetSet/mc-flavtag-ttbar-small-test.h5",
                  "/media/mule/scratch/JetSet/07-21/temp_test/",
                  "/media/mule/scratch/JetSet/07-21/temp_test/small_test_clean_jet_indices.npy"),
    }

    n_workers = 230
    MAX_JETS = None

    for split in SPLITS:
        raw_path, save_path, indices_path = SPLITS[split]

        if not os.path.exists(raw_path):
            print(f"{split}: {raw_path} not found — skipping", flush=True)
            continue

        print(f"\n {split}: {raw_path}", flush=True)
        g = h5py.File(raw_path, "r")
        print(g['jets'].shape)
        print(g.keys())

        jet_arr_all = g['jets'][:]
        tracks_arr_all = g['tracks'][:]
        truth_arr_all = g['truth_hadrons'][:]

        print(f"jet_arr_all.shape: {jet_arr_all.shape}")
        print(f"tracks_arr_all.shape: {tracks_arr_all.shape}")
        print(f"truth_arr_all.shape: {truth_arr_all.shape}")

        clean_indices = build_and_save_clean_jets(tracks_arr_all, jet_arr_all, truth_arr_all, save_indices_path=indices_path)
        if MAX_JETS is not None:
            clean_indices = clean_indices[:MAX_JETS]

        print(f"Processing {len(clean_indices)} jets using up to {n_workers} cores...", flush=True)

        start_time = time.time()
        run_pipeline_hpc(
            clean_indices=clean_indices,
            tracks_arr=tracks_arr_all,
            jet_arr=jet_arr_all,
            truth_arr=truth_arr_all,
            graph_types=graph_types,
            output_dir=save_path,
            shard_prefix=f"{RUN_NAME}_{split}",
            n_workers=n_workers,
        )
        elapsed = time.time() - start_time
        print(f"\nTime elapsed for {split}: {round(elapsed, 2)} seconds")

        verify_saved_shards(save_path, shard_prefix=f"{RUN_NAME}_{split}", expected_jets=len(clean_indices))

    print("All tasks finished successfully!")