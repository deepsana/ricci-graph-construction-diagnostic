import os
import glob
import time
import h5py
import numpy as np
import pandas as pd
import torch

from src.jetset_graphs import DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES
from utils.diminishing_returns import run_experiment
from utils.diminishing_returns_deep_models import build_padded_tensors, build_padded_edges, standardize_features,SmallTransformer, SimpleGNN, train_and_score
from sklearn.metrics import roc_auc_score, mean_absolute_error
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import load_split

from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_ROOT = PATHS.JETSET_SHARD_ROOT
CACHE_ROOT = PATHS.JETSET_CACHE_DIR
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "jetset")


RUN_TAG = "2026-09-07_default_deep_degree_300k_50k"       # change per run
CSV_PATH = os.path.join(OUT_DIR, f"diminishing_returns-{RUN_TAG}.csv")
PREDS_PATH = os.path.join(OUT_DIR, f"test_predictions-{RUN_TAG}.csv")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints", RUN_TAG)

# deliberate subsample; None = all common jets. Recorded in the csv via n_train / n_test
MAX_TRAIN_JETS = 300000
MAX_TEST_JETS = 50000
SUBSAMPLE_SEED = 0
N_WORKERS = 16 #200 #240

GRAPH_TYPES =  list(DEFAULT_GRAPH_TYPES)
#GRAPH_TYPES = ["knn_3", "fully_connected", "radius_based", "laman_p_neg1", "unique_3_p_neg1"]
#GRAPH_TYPES = ["kt_threshold_quantile_p_pos1", "kt_threshold_quantile_p_neg1", "kt_threshold_quantile_p_zero", "laman_p_pos1", "laman_p_neg1", "laman_p_zero", "unique_3_p_pos1", "unique_3_p_neg1", "unique_3_p_zero"]
# ["knn_3", "fully_connected", "radius_based"]
FC_NAME = GRAPH_TYPES[0] #"fully_connected"

TASK = "classification"
 # (name, signal flavor, background flavor)
TARGETS= [
    ("b_vs_light", 5, 0),
    ("c_vs_light", 4, 0),
    ("tau_vs_light",15, 0),
    #("b_vs_c", 5, 4),
    #("tau_vs_b", 15, 5),
    #("tau_vs_c", 15, 4),
]

# per-track features. Curvature enters the deep models as two extra channels.
POINT_FEATURES =  ["pos_deta", "pos_dphi", "track_pt"]
DEEP_POINT_FEATURES = POINT_FEATURES + ["n_tracks"]
CURV_POINT_FEATURES = ["orc_curvature", "frc_curvature"]

# summary statistics fed to the sklearn ladder (run_experiment)
SUMMARY_STATS = ["mean", "median", "std", "min", "max", "skew", "p10", "p25", "p75", "p90"]

MODEL_SEEDS =  [0, 1, 2, 4, 44]
N_BOOT = 50
DEEP_EPOCHS = 30
DEEP_PATIENCE = 5
DEEP_BATCH = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# The baseline inputs of the sklearn ladder and the transformer do not depend on the
# graph (points only), so they are trained once per seed under this tag and the result
# rows are copied to every construction afterwards. The GNN and every "+ curvature" run
# depend on the construction and are trained per graph type.
SHARED_TAG = "shared_baseline"
import pickle

def summarize_per_jet(nodes_df, feature_cols, jet_col="jet_id"):
    '''
    Turn per track features into per jet summary stats, one row per jet. For every
    column in feature_cols you get <col>_mean, <col>_median, <col>_std, and so on
    (the list is SUMMARY_STATS). A column n_tracks with the number of tracks in the
    jet gets added at the end.

    std and skew are NaN for jets with a single track, those get set to 0.

    :param nodes_df: DataFrame with one row per track, needs jet_col and every column in feature_cols
    :param feature_cols: list of per track columns to summarize
    :param jet_col: name of the column holding the jet id
    :return: DataFrame indexed by jet id, one column per feature/stat pair plus n_tracks
    '''
    g = nodes_df.groupby(jet_col)
    pieces = {}
    for c in feature_cols:
        col = g[c]
        pieces[f"{c}_mean"] = col.mean()
        pieces[f"{c}_median"] = col.median()
        pieces[f"{c}_std"] = col.std().fillna(0.0)
        pieces[f"{c}_min"] = col.min()
        pieces[f"{c}_max"] = col.max()
        pieces[f"{c}_skew"] = col.skew().fillna(0.0)
        for q, name in [(0.10, "p10"), (0.25, "p25"), (0.75, "p75"), (0.90, "p90")]:
            pieces[f"{c}_{name}"] = col.quantile(q)
    out = pd.DataFrame(pieces)
    out["n_tracks"] = g.size()
    return out


def select_task_jets(nodes_df, signal_flavor, background_flavor):
    '''
    Keep only the jets of the two flavors we care about and add a "label" column,
    1 for signal and 0 for background. Every track already carries its jet's
    flavor so this works row by row.

    :param nodes_df: one row per track, needs a "flavor" column
    :param signal_flavor: flavor code that becomes label 1
    :param background_flavor: flavor code that becomes label 0
    :return: filtered copy of nodes_df with a new "label" column (float32)
    '''
    d = nodes_df[nodes_df["flavor"].isin([signal_flavor, background_flavor])].copy()
    d["label"] = (d["flavor"] == signal_flavor).astype(np.float32)
    d["n_tracks"] = d.groupby("jet_id")["jet_id"].transform("size").astype(np.float32)

    return d


def bootstrap_score(y, preds, task, n_boot, rng):
    '''
    Score the test set n_boot times on random resamples (with replacement) and
    report the mean and the spread. That spread is the error bar in the csv.

    A resample that happens to contain only one class can't be scored with AUC so
    it just gets skipped.

    :param y: true labels (or true values for regression), 1d array
    :param preds: model predictions, same length and order as y
    :param task: "classification" scores with AUC, "regression" with MAE
    :param n_boot: how many resamples to draw
    :param rng: numpy Generator, e.g. np.random.default_rng(0), so the resamples
        are the same every time
    :return: (score, error), the mean and std over the resamples as floats
    '''
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
    '''
    Add one result row (a dict) to the rows list. These dicts become the rows of
    diminishing_returns-1.csv at the end.

    :param rows: the list to append to (all_rows in main)
    :param dataset: dataset name, e.g. "jets_b_vs_light"
    :param graph_type: graph construction name, or SHARED_TAG for shared baselines
    :param model: model name, e.g. "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature"
    :param score: test score from bootstrap_score
    :param error: its error bar from bootstrap_score
    :param n_train: number of jets the model was fit on
    :param n_test: number of test jets
    :param n_features: number of input features the model got
    :param model_seed: the seed used for this run
    :return: nothing, rows is changed in place
    '''
    rows.append({"dataset": dataset, "graph_type": graph_type, "task": TASK,
                 "target": "label", "model": model, "setting": setting,
                 "score": score, "error": error, "n_train": n_train, "n_test": n_test,
                 "n_features": n_features, "model_seed": model_seed})


def replicate_shared_rows(rows, graph_types):
    '''
    The shared baselines were saved under SHARED_TAG instead of a real graph type.
    The analysis scripts match baseline and "+ curvature" rows on graph_type, so
    here we copy every SHARED_TAG row once per graph type and drop the originals.
    The copies are identical across graph types, which is right, it really is the
    same model.

    :param rows: list of result dicts, some with graph_type == SHARED_TAG
    :param graph_types: list of constructions to copy the shared rows to
    :return: new list of rows, the shared ones replaced by one copy per graph type
    '''
    shared = []
    kept = []
    for r in rows:
        if r["graph_type"] == SHARED_TAG:
            shared.append(r)
        else:
            kept.append(r)

    for r in shared:
        for g in graph_types:
            copy = dict(r)
            copy["graph_type"] = g
            kept.append(copy)
    return kept


def save_predictions(path, dataset, graph_type, setting, model_seed, jet_ids, n, y, preds):
    '''
    Write one row per test jet to the predictions csv. Having these means you can
    score any subset later (for example only jets with n <= k+1 tracks) without
    retraining anything. Shared baselines get saved once under SHARED_TAG.

    The file is opened in append mode. The header only gets written the first time,
    after that rows just get added to the bottom.

    :param path: csv to append to (PREDS_PATH)
    :param dataset: dataset name
    :param graph_type: graph construction name, or SHARED_TAG
    :param setting: "baseline" or "+ curvature"
    :param model_seed: the seed used for this run
    :param jet_ids: jet id of every test jet, in prediction order
    :param n: number of real tracks in every test jet, same order
    :param y: true labels, same order
    :param preds: model predictions, same order
    :return: nothing, writes to disk
    '''
    df = pd.DataFrame({"dataset": dataset, "graph_type": graph_type, "setting": setting,
                       "model_seed": model_seed, "jet_id": jet_ids,
                       "n": np.asarray(n).astype(int), "y": y, "pred": preds})
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(model, dataset, graph_type, model_name, setting, model_seed):
    '''
    Save the model weights to CKPT_DIR. train_and_score already restores the best
    epoch before it returns, so what gets saved here is the best epoch. About 1 MB
    per file.

    :param model: the trained torch model
    :param dataset: dataset name
    :param graph_type: graph construction name, or SHARED_TAG
    :param model_name: "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature" (the space is removed in the filename)
    :param model_seed: the seed used for this run
    :return: nothing, writes a .pt file to CKPT_DIR
    '''
    tag = f"{dataset}__{graph_type}__{model_name}__{setting.replace(' ', '')}__seed{model_seed}.pt"
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, tag))


def holdout_split(n, model_seed):
    '''
    Split the training jets into a 10% holdout (used to pick the best epoch and
    to stop early) and the 90% we actually fit on. The split is random but fixed by
    model_seed, so every seed gets its own holdout.

    :param n: number of training jets
    :param model_seed: seed for the shuffle
    :return: (stop_idx, fit_idx), arrays of row indices for the holdout and for
        the fit set
    '''
    order = np.random.default_rng(model_seed).permutation(n)
    n_stop = max(1, int(0.1 * n))
    return order[:n_stop], order[n_stop:]


def fit_eval_deep(model, fit, stop, test, y_fit, y_stop, y_test):
    '''
    Train a model with early stopping on the holdout, then score the test set
    using the best epoch's weights (restored before returning).

    :param model: SmallTransformer or SimpleGNN
    :param fit: training tensors in forward() order: (X, mask) for transformers, or (X, mask, edge_index, edge_mask) for GNNs.
    :param stop: holdout tensors used for early stopping
    :param test: test tensors
    :param y_fit: labels for the training tensors
    :param y_stop: labels for the holdout tensors
    :param y_test: labels for the test tensors
    :return: Tuple of (score, error, preds). Score and error come from  bootstrap_score
            preds is a probability per test jet.
    '''
    train_and_score(model, fit, stop, y_fit, y_stop, epochs=DEEP_EPOCHS, batch_size=DEEP_BATCH, patience=DEEP_PATIENCE, device=DEVICE, verbose=False, task=TASK)
    model.eval()
    preds = []
    with torch.no_grad():
        # predict in chunks of 4096 jets so we don't blow up gpu memory
        for start in range(0, len(y_test), 4096):
            inputs = []
            for t in test:
                inputs.append(t[start:start + 4096].to(DEVICE))
            out = model(*inputs)
            if TASK == "classification":
                # model gives logits, turn them into probabilities for AUC
                out = torch.sigmoid(out)
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds)
    score, error = bootstrap_score(y_test, preds, TASK, N_BOOT, np.random.default_rng(0))
    return score, error, preds

def to_tensor(a):
    '''
    numpy to torch tensor
    '''
    return torch.tensor(a, dtype=torch.float32)


# main

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # the predictions file gets appended to as we go, so if a previous run left one
    # behind we would end up with doubled rows. start clean.
    if os.path.exists(PREDS_PATH):
        os.remove(PREDS_PATH)

    start = time.time()

    start = time.time()
    # Jets present in both the default and sweep families. The intersection script
    # normally computes these once and saves them next to the shards. If the files
    # are not there we compute them here and save them for next time.
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
    #print(train_dict[FC_NAME]["nodes"].drop_duplicates("jet_id")["flavor"].value_counts())
    #print(test_dict[FC_NAME]["nodes"].drop_duplicates("jet_id")["flavor"].value_counts())




    # every result row from every task ends up in here
    all_rows = []

    for target_name, signal, background in TARGETS:
        dataset_name = f"jets_{target_name}"

        # Point features and labels are the same under every construction (only the
        # edges and the curvature differ), so take them from FC_NAME once.
        tr_nodes_fc = select_task_jets(train_dict[FC_NAME]["nodes"], signal, background)
        te_nodes_fc = select_task_jets(test_dict[FC_NAME]["nodes"], signal, background)

        # shared baselines, once per seed, copied to every construction later
        print(f"\n{'=' * 60}\n{dataset_name} -- shared baselines\n{'=' * 60}")

        # per jet summary stats of the point features + the label, this is the
        # table the sklearn models train on
        tr_sum = summarize_per_jet(tr_nodes_fc, POINT_FEATURES)
        te_sum = summarize_per_jet(te_nodes_fc, POINT_FEATURES)
        labels_tr = tr_nodes_fc.groupby("jet_id")["label"].first()
        labels_te = te_nodes_fc.groupby("jet_id")["label"].first()
        base_train_df = tr_sum.assign(label=labels_tr).reset_index()
        base_test_df = te_sum.assign(label=labels_te).reset_index()
        baseline_features = list(tr_sum.columns)

        # sklearn ladder baseline. run_experiment trains its own set of sklearn
        # models and hands back a DataFrame of result rows.
        '''for model_seed in MODEL_SEEDS:
            df = run_experiment(base_train_df, base_test_df, baseline_features, [],
                                target_col="label", task=TASK, n_boot=N_BOOT,
                                dataset_name=dataset_name, graph_type=SHARED_TAG,
                                csv_path=None, model_seed=model_seed,
                                settings=("baseline",))
            all_rows.extend(df.to_dict("records"))'''


        Xb_tr, mb_tr, y_tr, ids_tr = build_padded_tensors(tr_nodes_fc, DEEP_POINT_FEATURES, label_col="label")
        Xb_te, mb_te, y_te, ids_te = build_padded_tensors(te_nodes_fc, DEEP_POINT_FEATURES, label_col="label",
                                                          max_nodes=Xb_tr.shape[1])
        # z-score using train stats only, ignoring the padding
        Xb_tr, Xb_te = standardize_features(Xb_tr, mb_tr, Xb_te)
        # number of real tracks per test jet, saved with the predictions
        n_te = mb_te.sum(axis=1)

        for model_seed in MODEL_SEEDS:
            stop_idx, fit_idx = holdout_split(len(y_tr), model_seed)
            torch.manual_seed(model_seed)
            model = SmallTransformer(len(DEEP_POINT_FEATURES))
            score, error, preds = fit_eval_deep(
                model,
                (to_tensor(Xb_tr[fit_idx]), to_tensor(mb_tr[fit_idx])),
                (to_tensor(Xb_tr[stop_idx]), to_tensor(mb_tr[stop_idx])),
                (to_tensor(Xb_te), to_tensor(mb_te)),
                y_tr[fit_idx], y_tr[stop_idx], y_te)
            append_result(all_rows, dataset_name, SHARED_TAG, "transformer", "baseline",
                          score, error, len(fit_idx), len(y_te), len(DEEP_POINT_FEATURES), model_seed)
            save_predictions(PREDS_PATH, dataset_name, SHARED_TAG, "baseline", model_seed, ids_te, n_te, y_te, preds)
            save_checkpoint(model, dataset_name, SHARED_TAG, "transformer", "baseline", model_seed)
            print(f"  transformer baseline seed {model_seed}  score = {score:.4f} +/- {error:.4f}")

        # per construction runs
        for graph_type in GRAPH_TYPES:
            print(f"\n{'=' * 60}\n{dataset_name} -- {graph_type}\n{'=' * 60}")

            tr_nodes = select_task_jets(train_dict[graph_type]["nodes"], signal, background)
            te_nodes = select_task_jets(test_dict[graph_type]["nodes"], signal, background)
            tr_edges = train_dict[graph_type]["edges"]
            te_edges = test_dict[graph_type]["edges"]

            # sklearn ladder, "+ curvature" only (the baseline is shared). Summarize
            # the curvature per jet, glue it onto the baseline table, and let
            # run_experiment measure what the extra columns buy you.
            tr_curv = summarize_per_jet(tr_nodes, CURV_POINT_FEATURES).drop(columns="n_tracks")
            te_curv = summarize_per_jet(te_nodes, CURV_POINT_FEATURES).drop(columns="n_tracks")
            train_df = base_train_df.merge(tr_curv, on="jet_id")
            test_df = base_test_df.merge(te_curv, on="jet_id")
            curvature_features = list(tr_curv.columns)

            '''for model_seed in MODEL_SEEDS:
                df = run_experiment(train_df, test_df, baseline_features, curvature_features,
                                    target_col="label", task=TASK, n_boot=N_BOOT,
                                    dataset_name=dataset_name, graph_type=graph_type,
                                    csv_path=None, model_seed=model_seed,
                                    settings=("+ curvature",))
                all_rows.extend(df.to_dict("records"))'''

            # deep tensors for this construction. Same as above but with the two
            # curvature columns tacked on after the point features, plus the edges.
            all_feats = DEEP_POINT_FEATURES + CURV_POINT_FEATURES
            X_tr, m_tr, y_tr_g, ids_tr_g = build_padded_tensors(tr_nodes, all_feats, label_col="label")
            X_te, m_te, y_te_g, ids_te_g = build_padded_tensors(te_nodes, all_feats, label_col="label",
                                                                max_nodes=X_tr.shape[1])
            X_tr, X_te = standardize_features(X_tr, m_tr, X_te)
            n_te_g = m_te.sum(axis=1)
            # edge_index (n_jets, max_edges, 2) and a mask saying which edges are real
            ei_tr, em_tr = build_padded_edges(tr_edges, ids_tr_g, max_nodes=X_tr.shape[1])
            ei_te, em_te = build_padded_edges(te_edges, ids_te_g, max_nodes=X_tr.shape[1])

            # The three deep runs that depend on the graph. Each entry is
            # (model name, setting name, how many feature columns to feed it).
            # The first n_base columns of X are the point features and the two after
            # them are curvature, so "baseline" just uses the first n_base columns.
            # The transformer baseline is not here because it was done above.
            n_base = len(DEEP_POINT_FEATURES)
            n_all = len(all_feats)
            deep_runs = [("transformer", "+ ntracks + curvature", n_all),
                         ("GNN", "baseline + ntracks", n_base),
                         ("GNN", "+ ntracks + curvature", n_all)]

            for model_seed in MODEL_SEEDS:
                stop_idx, fit_idx = holdout_split(len(y_tr_g), model_seed)
                for model_name, setting_name, n_cols in deep_runs:
                    # reseed before building each model so the init weights only
                    # depend on the seed, not on what ran before
                    torch.manual_seed(model_seed)
                    if model_name == "transformer":
                        model = SmallTransformer(n_cols)
                        fit = (to_tensor(X_tr[fit_idx][:, :, :n_cols]), to_tensor(m_tr[fit_idx]))
                        stop = (to_tensor(X_tr[stop_idx][:, :, :n_cols]), to_tensor(m_tr[stop_idx]))
                        test = (to_tensor(X_te[:, :, :n_cols]), to_tensor(m_te))
                    else:
                        model = SimpleGNN(n_cols)
                        fit = (to_tensor(X_tr[fit_idx][:, :, :n_cols]), to_tensor(m_tr[fit_idx]),
                               to_tensor(ei_tr[fit_idx], torch.int32), to_tensor(em_tr[fit_idx]))
                        stop = (to_tensor(X_tr[stop_idx][:, :, :n_cols]), to_tensor(m_tr[stop_idx]),
                                to_tensor(ei_tr[stop_idx], torch.int32), to_tensor(em_tr[stop_idx]))
                        test = (to_tensor(X_te[:, :, :n_cols]), to_tensor(m_te), to_tensor(ei_te, torch.int32),
                                to_tensor(em_te))

                    score, error, preds = fit_eval_deep(model, fit, stop, test, y_tr_g[fit_idx], y_tr_g[stop_idx],
                                                        y_te_g)
                    append_result(all_rows, dataset_name, graph_type, model_name, setting_name, score, error,
                                  len(fit_idx), len(y_te_g), n_cols, model_seed)
                    save_predictions(PREDS_PATH, dataset_name, graph_type, setting_name, model_seed, ids_te_g, n_te_g,
                                     y_te_g, preds)
                    save_checkpoint(model, dataset_name, graph_type, model_name, setting_name, model_seed)
                    print(f"  {model_name:<12}{setting_name:<14}seed {model_seed}  score = {score:.4f} +/- {error:.4f}")

            # rewrite the csv after every construction so a crash loses at most one
            pd.DataFrame(replicate_shared_rows(all_rows, GRAPH_TYPES)).to_csv(CSV_PATH, index=False)

    # final write, same thing but with everything in it
    pd.DataFrame(replicate_shared_rows(all_rows, GRAPH_TYPES)).to_csv(CSV_PATH, index=False)
    print(f"\nresults -> {CSV_PATH}")
    print(f"predictions -> {PREDS_PATH}   (shared baselines under '{SHARED_TAG}')")
    print(f"checkpoints -> {CKPT_DIR}")