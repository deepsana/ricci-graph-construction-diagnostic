import os
import time
import pickle
import numpy as np
import pandas as pd
import torch

from src.jetset_graphs import DEFAULT_GRAPH_TYPES
from utils.diminishing_returns_deep_models import build_padded_tensors, build_padded_edges, standardize_features, SimpleGNN, train_and_score
from sklearn.metrics import roc_auc_score, mean_absolute_error
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import load_split

from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_ROOT = PATHS.JETSET_SHARD_ROOT
CACHE_ROOT = PATHS.JETSET_CACHE_DIR
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "jetset")

RUN_TAG = "2026-09-22_default_gnn_orc_frc_only_300k_50k"    # change per run
CSV_PATH = os.path.join(OUT_DIR, f"diminishing_returns-{RUN_TAG}.csv")
PREDS_PATH = os.path.join(OUT_DIR, f"test_predictions-{RUN_TAG}.csv")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints", RUN_TAG)

MAX_TRAIN_JETS = 300000
MAX_TEST_JETS = 50000
SUBSAMPLE_SEED = 0
N_WORKERS = 16

GRAPH_TYPES = list(DEFAULT_GRAPH_TYPES)

TASK = "classification"
# (name, signal flavor, background flavor)
TARGETS = [
    ("b_vs_light", 5, 0),
    ("c_vs_light", 4, 0),
    ("tau_vs_light", 15, 0),
]

POINT_FEATURES = ["pos_deta", "pos_dphi", "track_pt"]
DEEP_POINT_FEATURES = POINT_FEATURES + ["n_tracks"]
CURV_POINT_FEATURES = ["orc_curvature", "frc_curvature"]

MODEL_SEEDS = [0, 1, 2, 4, 44]
N_BOOT = 50
DEEP_EPOCHS = 30
DEEP_PATIENCE = 5
DEEP_BATCH = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def select_task_jets(nodes_df, signal_flavor, background_flavor):
    d = nodes_df[nodes_df["flavor"].isin([signal_flavor, background_flavor])].copy()
    d["label"] = (d["flavor"] == signal_flavor).astype(np.float32)
    d["n_tracks"] = d.groupby("jet_id")["jet_id"].transform("size").astype(np.float32)
    return d


def bootstrap_score(y, preds, task, n_boot, rng):
    scores = []
    n = len(y)
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        yb, pb = y[pick], preds[pick]
        if task == "classification":
            if len(np.unique(yb)) < 2:
                continue
            scores.append(roc_auc_score(yb, pb))
        else:
            scores.append(mean_absolute_error(yb, pb))
    return float(np.mean(scores)), float(np.std(scores))


def append_result(rows, dataset, graph_type, model, setting, score, error,
                  n_train, n_test, n_features, model_seed):
    rows.append({"dataset": dataset, "graph_type": graph_type, "task": TASK,
                 "target": "label", "model": model, "setting": setting,
                 "score": score, "error": error, "n_train": n_train, "n_test": n_test,
                 "n_features": n_features, "model_seed": model_seed})


def save_predictions(path, dataset, graph_type, setting, model_seed, jet_ids, n, y, preds):
    df = pd.DataFrame({"dataset": dataset, "graph_type": graph_type, "setting": setting,
                       "model_seed": model_seed, "jet_id": jet_ids,
                       "n": np.asarray(n).astype(int), "y": y, "pred": preds})
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(model, dataset, graph_type, model_name, setting, model_seed):
    tag = f"{dataset}__{graph_type}__{model_name}__{setting.replace(' ', '')}__seed{model_seed}.pt"
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, tag))


def holdout_split(n, model_seed):
    order = np.random.default_rng(model_seed).permutation(n)
    n_stop = max(1, int(0.1 * n))
    return order[:n_stop], order[n_stop:]


def fit_eval_deep(model, fit, stop, test, y_fit, y_stop, y_test):
    train_and_score(model, fit, stop, y_fit, y_stop,
                    epochs=DEEP_EPOCHS, batch_size=DEEP_BATCH,
                    patience=DEEP_PATIENCE, device=DEVICE, verbose=False, task=TASK)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(y_test), 4096):
            inputs = [t[start:start + 4096].to(DEVICE) for t in test]
            out = model(*inputs)
            if TASK == "classification":
                out = torch.sigmoid(out)
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds)
    score, error = bootstrap_score(y_test, preds, TASK, N_BOOT, np.random.default_rng(0))
    return score, error, preds


def to_tensor(a, dt=torch.float32):
    return torch.tensor(a, dtype=dt)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(PREDS_PATH):
        os.remove(PREDS_PATH)

    start = time.time()
    common_tr = get_kept_ids("train", MAX_TRAIN_JETS, SUBSAMPLE_SEED)
    common_te = get_kept_ids("test", MAX_TEST_JETS, SUBSAMPLE_SEED)
    CACHE = os.path.join(CACHE_ROOT, f"cache_{MAX_TRAIN_JETS}_default_{MAX_TEST_JETS}_seed{SUBSAMPLE_SEED}.pkl")
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            train_dict, test_dict = pickle.load(f)
    else:
        # None reads every column, as before; the models need pos_deta, pos_dphi, track_pt, ...
        train_dict = load_split("train", GRAPH_TYPES, common_tr, N_WORKERS, node_cols=None, edge_cols=None)
        test_dict = load_split("test", GRAPH_TYPES, common_te, N_WORKERS, node_cols=None, edge_cols=None)
        with open(CACHE, "wb") as f:
            pickle.dump((train_dict, test_dict), f, protocol=4)
    print(f"loaded in {round(time.time() - start, 1)} s")

    missing = [g for g in GRAPH_TYPES if g not in train_dict or g not in test_dict]
    if missing:
        raise KeyError(f"cache is missing graph types: {missing}")

    all_feats = DEEP_POINT_FEATURES + CURV_POINT_FEATURES
    n_base = len(DEEP_POINT_FEATURES)
    base_cols = list(range(n_base))
    i_orc = n_base + CURV_POINT_FEATURES.index("orc_curvature")
    i_frc = n_base + CURV_POINT_FEATURES.index("frc_curvature")
    deep_runs = [("GNN", "baseline + ntracks", base_cols),
                 ("GNN", "+ ntracks + orc", base_cols + [i_orc]),
                 ("GNN", "+ ntracks + frc", base_cols + [i_frc]),
                 ("GNN", "+ ntracks + curvature", base_cols + [i_orc, i_frc])]

    all_rows = []

    for target_name, signal, background in TARGETS:
        dataset_name = f"jets_{target_name}"

        for graph_type in GRAPH_TYPES:
            print(f"\n{'=' * 60}\n{dataset_name} -- {graph_type}\n{'=' * 60}")

            tr_nodes = select_task_jets(train_dict[graph_type]["nodes"], signal, background)
            te_nodes = select_task_jets(test_dict[graph_type]["nodes"], signal, background)
            tr_edges = train_dict[graph_type]["edges"]
            te_edges = test_dict[graph_type]["edges"]

            X_tr, m_tr, y_tr_g, ids_tr_g = build_padded_tensors(tr_nodes, all_feats, label_col="label")
            X_te, m_te, y_te_g, ids_te_g = build_padded_tensors(te_nodes, all_feats, label_col="label",
                                                                max_nodes=X_tr.shape[1])
            X_tr, X_te = standardize_features(X_tr, m_tr, X_te)
            n_te_g = m_te.sum(axis=1)
            ei_tr, em_tr = build_padded_edges(tr_edges, ids_tr_g, max_nodes=X_tr.shape[1])
            ei_te, em_te = build_padded_edges(te_edges, ids_te_g, max_nodes=X_tr.shape[1])

            for model_seed in MODEL_SEEDS:
                stop_idx, fit_idx = holdout_split(len(y_tr_g), model_seed)
                for model_name, setting_name, cols in deep_runs:
                    torch.manual_seed(model_seed)
                    n_cols = len(cols)
                    model = SimpleGNN(n_cols)
                    fit = (to_tensor(X_tr[fit_idx][:, :, cols]), to_tensor(m_tr[fit_idx]),
                           to_tensor(ei_tr[fit_idx], torch.int32), to_tensor(em_tr[fit_idx]))
                    stop = (to_tensor(X_tr[stop_idx][:, :, cols]), to_tensor(m_tr[stop_idx]),
                            to_tensor(ei_tr[stop_idx], torch.int32), to_tensor(em_tr[stop_idx]))
                    test = (to_tensor(X_te[:, :, cols]), to_tensor(m_te),
                            to_tensor(ei_te, torch.int32), to_tensor(em_te))

                    score, error, preds = fit_eval_deep(model, fit, stop, test,
                                                        y_tr_g[fit_idx], y_tr_g[stop_idx], y_te_g)
                    append_result(all_rows, dataset_name, graph_type, model_name, setting_name, score, error,
                                  len(fit_idx), len(y_te_g), n_cols, model_seed)
                    save_predictions(PREDS_PATH, dataset_name, graph_type, setting_name, model_seed,
                                     ids_te_g, n_te_g, y_te_g, preds)
                    save_checkpoint(model, dataset_name, graph_type, model_name, setting_name, model_seed)
                    print(f"  {model_name:<6}{setting_name:<20}seed {model_seed}  "
                          f"score = {score:.4f} +/- {error:.4f}")

            # rewrite after every construction so a crash loses at most one
            pd.DataFrame(all_rows).to_csv(CSV_PATH, index=False)

    pd.DataFrame(all_rows).to_csv(CSV_PATH, index=False)
    print(f"\nresults -> {CSV_PATH}")
    print(f"predictions -> {PREDS_PATH}")
    print(f"checkpoints -> {CKPT_DIR}")