import os
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score

from experiments.jetset.config import CACHE_ROOT, PATHS
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import load_split
from src.jetset_graphs import DEFAULT_GRAPH_TYPES

from utils.diminishing_returns import run_experiment
from utils.diminishing_returns_deep_models import SimpleGNN, SmallTransformer, build_padded_edges, build_padded_tensors, standardize_features, train_and_score

from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_ROOT = PATHS.JETSET_SHARD_ROOT
CACHE_ROOT = PATHS.JETSET_CACHE_DIR
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "jetset")


OUT_DIR = "results-jetset/"
RUN_TAG = "2026-09-07_default_300k_50k"  # change per run
CSV_PATH = os.path.join(OUT_DIR, f"diminishing_returns-{RUN_TAG}.csv")
PREDS_PATH = os.path.join(OUT_DIR, f"test_predictions-{RUN_TAG}.csv")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints", RUN_TAG)

# Deliberate subsample (None uses all common jets). The sizes end up in the
# output CSV as n_train and n_test.
MAX_TRAIN_JETS = 300000
MAX_TEST_JETS = 50000
SUBSAMPLE_SEED = 0
N_WORKERS = 16

GRAPH_TYPES = list(DEFAULT_GRAPH_TYPES)
# Point features and labels do not depend on the construction, so the
# shared baselines take them from the first one.
POINT_GRAPH = GRAPH_TYPES[0]

TASK = "classification"
# (name, signal flavor, background flavor)
TARGETS = [
    ("b_vs_light", 5, 0),
    ("c_vs_light", 4, 0),
    ("tau_vs_light", 15, 0),
    # ("b_vs_c", 5, 4),
    # ("tau_vs_b", 15, 5),
    # ("tau_vs_c", 15, 4),
]

POINT_FEATURES = ["pos_deta", "pos_dphi", "track_pt"]
# Curvature enters the deep models as two extra channels placed after the
# point features.
CURV_POINT_FEATURES = ["orc_curvature", "frc_curvature"]
QUANTILES = [(0.10, "p10"), (0.25, "p25"), (0.75, "p75"), (0.90, "p90")]

MODEL_SEEDS = [0, 1, 2, 4, 44]
N_BOOT = 50
DEEP_EPOCHS = 30
DEEP_PATIENCE = 5
DEEP_BATCH = 512
PREDICT_BATCH = 4096
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda:0"

# The sklearn and transformer baselines use point features only, so they
# are trained once per seed under this tag and copied to every construction.
SHARED_TAG = "shared_baseline"
DIVIDER = "=" * 60


def summarize_per_jet(nodes_df, feature_cols, jet_col="jet_id"):
    """
    Summarize per-track features per jet (mean, median, std, min, max, skew
    and four quantiles) and add an n_tracks column. The std and skew of
    single-track jets are set to 0 instead of NaN.

    :param nodes_df: one row per track with jet_col and every feature column
    :param feature_cols: per-track columns to summarize
    :param jet_col: name of the jet id column
    :return: DataFrame indexed by jet id with one column per feature and
        statistic, plus n_tracks
    """
    grouped = nodes_df.groupby(jet_col)
    columns = {}
    for feature in feature_cols:
        values = grouped[feature]
        columns[f"{feature}_mean"] = values.mean()
        columns[f"{feature}_median"] = values.median()
        columns[f"{feature}_std"] = values.std().fillna(0.0)
        columns[f"{feature}_min"] = values.min()
        columns[f"{feature}_max"] = values.max()
        columns[f"{feature}_skew"] = values.skew().fillna(0.0)
        for quantile, name in QUANTILES:
            columns[f"{feature}_{name}"] = values.quantile(quantile)
    summary = pd.DataFrame(columns)
    summary["n_tracks"] = grouped.size()
    return summary


def select_task_jets(nodes_df, signal_flavor, background_flavor):
    """
    Keep the tracks of signal and background jets and add a binary label.

    :param nodes_df: one row per track with a flavor column
    :param signal_flavor: flavor code labeled 1
    :param background_flavor: flavor code labeled 0
    :return: filtered copy of nodes_df with a float32 label column
    """
    keep = nodes_df["flavor"].isin([signal_flavor, background_flavor])
    task_nodes = nodes_df[keep].copy()
    is_signal = task_nodes["flavor"] == signal_flavor
    task_nodes["label"] = is_signal.astype(np.float32)
    return task_nodes


def bootstrap_score(y, preds, task, n_boot, rng):
    """
    Compute the mean and standard deviation of the test score over n_boot
    resamples drawn with replacement. For AUC, resamples that contain a
    single class are skipped.

    :param y: true labels (or values for regression)
    :param preds: predictions in the same order as y
    :param task: "classification" scores with AUC, "regression" with MAE
    :param n_boot: number of resamples
    :param rng: numpy Generator, so the resamples are reproducible
    :return: tuple (score, error)
    """
    scores = []
    n_samples = len(y)
    for _ in range(n_boot):
        pick = rng.integers(0, n_samples, n_samples)
        y_sample = y[pick]
        pred_sample = preds[pick]
        if task == "classification":
            if len(np.unique(y_sample)) < 2:
                continue
            scores.append(roc_auc_score(y_sample, pred_sample))
        else:
            scores.append(mean_absolute_error(y_sample, pred_sample))
    return float(np.mean(scores)), float(np.std(scores))


def append_result(rows, dataset, graph_type, model, setting, score, error, n_train, n_test, n_features, model_seed):
    """
    Add one result row with the model metrics and run metadata.

    :param rows: result list to append to
    :param dataset: dataset name, e.g. "jets_b_vs_light"
    :param graph_type: construction name, or SHARED_TAG
    :param model: model name, "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature"
    :param score: test score from bootstrap_score
    :param error: its error from bootstrap_score
    :param n_train: number of jets the model was fit on
    :param n_test: number of test jets
    :param n_features: number of input features
    :param model_seed: seed of this run
    :return: None; rows is extended in place
    """
    rows.append({"dataset": dataset,
                 "graph_type": graph_type,
                 "task": TASK,
                 "target": "label",
                 "model": model,
                 "setting": setting,
                 "score": score,
                 "error": error,
                 "n_train": n_train,
                 "n_test": n_test,
                 "n_features": n_features,
                 "model_seed": model_seed})


def replicate_shared_rows(rows, graph_types):
    """
    Replace every SHARED_TAG row with one copy per construction.

    :param rows: result rows, some tagged SHARED_TAG
    :param graph_types: constructions to copy the shared rows to
    :return: new list with the construction rows first, then the copies
    """
    shared = []
    kept = []
    for row in rows:
        if row["graph_type"] == SHARED_TAG:
            shared.append(row)
        else:
            kept.append(row)
    for row in shared:
        for graph_type in graph_types:
            row_copy = dict(row)
            row_copy["graph_type"] = graph_type
            kept.append(row_copy)
    return kept


def save_predictions(path, dataset, graph_type, setting, model_seed, jet_ids, n, y, preds):
    """
    Append the test predictions, true labels and track counts of one run
    to a CSV, writing the header only when the file is new. The file has no
    model column, so runs of different models with the same setting are
    told apart only by their order.

    :param path: CSV to append to
    :param dataset: dataset name
    :param graph_type: construction name, or SHARED_TAG
    :param setting: "baseline" or "+ curvature"
    :param model_seed: seed of this run
    :param jet_ids: id of every test jet, in prediction order
    :param n: number of real tracks per test jet, same order
    :param y: true labels, same order
    :param preds: predictions, same order
    :return: None
    """
    predictions = pd.DataFrame({"dataset": dataset,
                                "graph_type": graph_type,
                                "setting": setting,
                                "model_seed": model_seed,
                                "jet_id": jet_ids,
                                "n": np.asarray(n).astype(int),
                                "y": y,
                                "pred": preds})
    predictions.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def save_checkpoint(model, dataset, graph_type, model_name, setting,
                    model_seed):
    """
    Save the best-epoch weights of a trained model to CKPT_DIR.

    :param model: trained torch model
    :param dataset: dataset name
    :param graph_type: construction name, or SHARED_TAG
    :param model_name: "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature"
    :param model_seed: seed of this run
    :return: None
    """
    setting_tag = setting.replace(" ", "")
    file_name = (f"{dataset}__{graph_type}__{model_name}__{setting_tag}__seed{model_seed}.pt")
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, file_name))


def holdout_split(n_jets, model_seed):
    """
    Shuffle the training jets with a fixed seed and split off 10% as the
    early-stopping holdout.

    :param n_jets: number of training jets
    :param model_seed: seed for the shuffle
    :return: tuple (stop_idx, fit_idx) of row indices
    """
    order = np.random.default_rng(model_seed).permutation(n_jets)
    n_stop = max(1, int(0.1 * n_jets))
    return order[:n_stop], order[n_stop:]


def deep_inputs(features, mask, n_cols, edge_index=None, edge_mask=None):
    """
    Build the input tensors of a model in forward() order: (X, mask) for
    the transformer and (X, mask, edge_index, edge_mask) for the GNN.

    :param features: padded features, shape (n_jets, max_nodes, n_features)
    :param mask: padded node mask
    :param n_cols: number of leading feature columns to keep
    :param edge_index: padded edge array, or None for the transformer
    :param edge_mask: padded edge mask, or None for the transformer
    :return: tuple of tensors
    """
    inputs = [torch.tensor(features[:, :, :n_cols], dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)]
    if edge_index is not None:
        inputs.append(torch.tensor(edge_index, dtype=torch.int32))
        inputs.append(torch.tensor(edge_mask, dtype=torch.float32))
    return tuple(inputs)


def fit_and_evaluate(model, fit_inputs, stop_inputs, test_inputs, y_fit,
                     y_stop, y_test):
    """
    Train a model with early stopping on the holdout and score the
    best-epoch weights on the test set.

    :param model: SmallTransformer or SimpleGNN
    :param fit_inputs: training tensors from deep_inputs
    :param stop_inputs: holdout tensors used for early stopping
    :param test_inputs: test tensors
    :param y_fit: training labels
    :param y_stop: holdout labels
    :param y_test: test labels
    :return: tuple (score, error, preds) with the bootstrap score and the
        test predictions
    """
    train_and_score(model, fit_inputs, stop_inputs, y_fit, y_stop,
                    epochs=DEEP_EPOCHS,
                    batch_size=DEEP_BATCH,
                    patience=DEEP_PATIENCE,
                    device=DEVICE,
                    verbose=False,
                    task=TASK)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(y_test), PREDICT_BATCH):
            batch = []
            for tensor in test_inputs:
                batch.append(tensor[start:start + PREDICT_BATCH].to(DEVICE))
            output = model(*batch)
            # The model returns logits; the saved predictions are
            # probabilities.
            if TASK == "classification":
                output = torch.sigmoid(output)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions)
    score, error = bootstrap_score(y_test, predictions, TASK, N_BOOT, np.random.default_rng(0))
    return score, error, predictions


def load_common_ids(split):
    """
    Load the ids of the jets present in both the default and sweep shard
    families. If the cached file is missing, the ids are computed and saved
    next to the shards.

    :param split: "train" or "test"
    :return: set of jet ids
    """
    path = os.path.join(SHARD_ROOT, f"common_{split}_jet_ids.npy")
    if not os.path.exists(path):
        jet_ids = common_jet_ids(split, INTERSECT_FAMILIES)
        np.save(path, np.array(sorted(jet_ids), dtype=np.int64))
        return jet_ids
    jet_ids = set()
    for jet_id in np.load(path):
        jet_ids.add(int(jet_id))
    print(f"{split}: {len(jet_ids)} common jet ids loaded from {path}")
    return jet_ids


def run_shared_baselines(all_rows, dataset_name, train_nodes, test_nodes):
    """
    Train the construction-independent baselines once per seed: the sklearn
    ladder on per-jet point summaries and the transformer on the point
    features.

    :param all_rows: result list, extended in place
    :param dataset_name: dataset name, e.g. "jets_b_vs_light"
    :param train_nodes: training tracks of the task, with a label column
    :param test_nodes: test tracks of the task, with a label column
    :return: tuple (base_train_df, base_test_df, baseline_features) with the
        per-jet summary tables and their feature columns
    """
    print(f"\n{DIVIDER}\n{dataset_name} -- shared baselines\n{DIVIDER}")

    train_summary = summarize_per_jet(train_nodes, POINT_FEATURES)
    test_summary = summarize_per_jet(test_nodes, POINT_FEATURES)
    train_labels = train_nodes.groupby("jet_id")["label"].first()
    test_labels = test_nodes.groupby("jet_id")["label"].first()
    base_train_df = train_summary.assign(label=train_labels).reset_index()
    base_test_df = test_summary.assign(label=test_labels).reset_index()
    baseline_features = list(train_summary.columns)

    for model_seed in MODEL_SEEDS:
        results = run_experiment(base_train_df, base_test_df,
                                 baseline_features=baseline_features,
                                 curvature_features=[],
                                 target_col="label",
                                 task=TASK,
                                 n_boot=N_BOOT,
                                 dataset_name=dataset_name,
                                 graph_type=SHARED_TAG,
                                 csv_path=None,
                                 model_seed=model_seed,
                                 settings=("baseline",))
        all_rows.extend(results.to_dict("records"))

    # The test jets are padded to the training width, and the features are
    # standardized with training statistics only.
    x_train, mask_train, y_train, _ = build_padded_tensors(train_nodes, POINT_FEATURES, label_col="label")
    x_test, mask_test, y_test, test_ids = build_padded_tensors(test_nodes, POINT_FEATURES, label_col="label", max_nodes=x_train.shape[1])
    x_train, x_test = standardize_features(x_train, mask_train, x_test)
    n_tracks_test = mask_test.sum(axis=1)
    n_features = len(POINT_FEATURES)

    for model_seed in MODEL_SEEDS:
        stop_idx, fit_idx = holdout_split(len(y_train), model_seed)
        torch.manual_seed(model_seed)
        model = SmallTransformer(n_features)
        score, error, preds = fit_and_evaluate(model,
                                               deep_inputs(x_train[fit_idx], mask_train[fit_idx], n_features),
                                               deep_inputs(x_train[stop_idx], mask_train[stop_idx], n_features),
                                               deep_inputs(x_test, mask_test, n_features),
                                               y_train[fit_idx], y_train[stop_idx], y_test)
        append_result(all_rows, dataset_name, SHARED_TAG, "transformer","baseline", score, error, len(fit_idx), len(y_test),n_features, model_seed)
        save_predictions(PREDS_PATH, dataset_name, SHARED_TAG, "baseline",model_seed, test_ids, n_tracks_test, y_test, preds)
        save_checkpoint(model, dataset_name, SHARED_TAG, "transformer","baseline", model_seed)
        print(f"  transformer baseline seed {model_seed}  score = {score:.4f} +/- {error:.4f}")

    return base_train_df, base_test_df, baseline_features


def run_sklearn_curvature(all_rows, dataset_name, graph_type, train_nodes, test_nodes, base_train_df, base_test_df, baseline_features):
    """
    Run the sklearn ladder with the per-jet curvature summaries added to
    the shared point summaries. Only the "+ curvature" setting is trained,
    since the baseline is shared.

    :param all_rows: result list, extended in place
    :param dataset_name: dataset name
    :param graph_type: construction name
    :param train_nodes: training tracks of the task for this construction
    :param test_nodes: test tracks of the task for this construction
    :param base_train_df: shared per-jet training summary table
    :param base_test_df: shared per-jet test summary table
    :param baseline_features: feature columns of the shared tables
    :return: None
    """
    train_curvature = summarize_per_jet(train_nodes, CURV_POINT_FEATURES)
    train_curvature = train_curvature.drop(columns="n_tracks")
    test_curvature = summarize_per_jet(test_nodes, CURV_POINT_FEATURES)
    test_curvature = test_curvature.drop(columns="n_tracks")
    train_df = base_train_df.merge(train_curvature, on="jet_id")
    test_df = base_test_df.merge(test_curvature, on="jet_id")
    curvature_features = list(train_curvature.columns)

    for model_seed in MODEL_SEEDS:
        results = run_experiment(train_df, test_df,
                                 baseline_features=baseline_features,
                                 curvature_features=curvature_features,
                                 target_col="label",
                                 task=TASK,
                                 n_boot=N_BOOT,
                                 dataset_name=dataset_name,
                                 graph_type=graph_type,
                                 csv_path=None,
                                 model_seed=model_seed,
                                 settings=("+ curvature",))
        all_rows.extend(results.to_dict("records"))


def run_deep_models(all_rows, dataset_name, graph_type, train_nodes, test_nodes, train_edges, test_edges):
    """
    Train the construction-dependent deep models once per seed: the
    transformer with curvature and the GNN with and without it. Baselines
    keep only the leading point-feature columns.

    :param all_rows: result list, extended in place
    :param dataset_name: dataset name
    :param graph_type: construction name
    :param train_nodes: training tracks of the task for this construction
    :param test_nodes: test tracks of the task for this construction
    :param train_edges: training edge table of this construction
    :param test_edges: test edge table of this construction
    :return: None
    """
    all_features = POINT_FEATURES + CURV_POINT_FEATURES
    x_train, mask_train, y_train, train_ids = build_padded_tensors(train_nodes, all_features, label_col="label")
    max_nodes = x_train.shape[1]

    x_test, mask_test, y_test, test_ids = build_padded_tensors(test_nodes, all_features, label_col="label", max_nodes=max_nodes)
    x_train, x_test = standardize_features(x_train, mask_train, x_test)

    n_tracks_test = mask_test.sum(axis=1)
    edges_train, edge_mask_train = build_padded_edges(train_edges, train_ids, max_nodes=max_nodes)
    edges_test, edge_mask_test = build_padded_edges(test_edges, test_ids, max_nodes=max_nodes)

    n_base = len(POINT_FEATURES)
    n_all = len(all_features)
    deep_runs = [ ("transformer", "+ curvature", n_all),
                  ("GNN", "baseline", n_base),
                  ("GNN", "+ curvature", n_all),
                  ]

    for model_seed in MODEL_SEEDS:
        stop_idx, fit_idx = holdout_split(len(y_train), model_seed)
        for model_name, setting, n_cols in deep_runs:
            # Reseeding before each model makes its initial weights depend
            # only on the seed, not on what ran before.
            torch.manual_seed(model_seed)
            if model_name == "transformer":
                model = SmallTransformer(n_cols)
                fit_inputs = deep_inputs(x_train[fit_idx], mask_train[fit_idx], n_cols)
                stop_inputs = deep_inputs(x_train[stop_idx], mask_train[stop_idx], n_cols)
                test_inputs = deep_inputs(x_test, mask_test, n_cols)
            else:
                model = SimpleGNN(n_cols)
                fit_inputs = deep_inputs(x_train[fit_idx],
                                         mask_train[fit_idx],
                                         n_cols,
                                         edges_train[fit_idx],
                                         edge_mask_train[fit_idx])
                stop_inputs = deep_inputs(x_train[stop_idx],
                                          mask_train[stop_idx],
                                          n_cols,
                                          edges_train[stop_idx],
                                          edge_mask_train[stop_idx])
                test_inputs = deep_inputs(x_test, mask_test, n_cols, edges_test, edge_mask_test)

            score, error, preds = fit_and_evaluate(model, fit_inputs, stop_inputs, test_inputs,y_train[fit_idx], y_train[stop_idx], y_test)
            append_result(all_rows, dataset_name, graph_type, model_name, setting, score, error, len(fit_idx), len(y_test),n_cols, model_seed)
            save_predictions(PREDS_PATH, dataset_name, graph_type, setting,model_seed, test_ids, n_tracks_test, y_test, preds)
            save_checkpoint(model, dataset_name, graph_type, model_name,setting, model_seed)
            print(f"  {model_name:<12}{setting:<14}seed {model_seed}  "
                  f"score = {score:.4f} +/- {error:.4f}")


def main():
    """
    Train the sklearn and deep models for every target and construction,
    and write the results, test predictions and checkpoints.

    :return: None
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Predictions are appended as the run goes, so a file left by an
    # earlier run would double the rows.
    if os.path.exists(PREDS_PATH):
        os.remove(PREDS_PATH)

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

    all_rows = []
    for target_name, signal, background in TARGETS:
        dataset_name = f"jets_{target_name}"
        train_points = select_task_jets(train_dict[POINT_GRAPH]["nodes"], signal, background)
        test_points = select_task_jets(test_dict[POINT_GRAPH]["nodes"], signal, background)
        base_train_df, base_test_df, baseline_features = run_shared_baselines(all_rows, dataset_name, train_points, test_points)

        for graph_type in GRAPH_TYPES:
            print(f"\n{DIVIDER}\n{dataset_name} -- {graph_type}\n{DIVIDER}")
            train_nodes = select_task_jets(train_dict[graph_type]["nodes"],signal, background)
            test_nodes = select_task_jets(test_dict[graph_type]["nodes"], signal, background)
            run_sklearn_curvature(all_rows, dataset_name, graph_type,train_nodes, test_nodes, base_train_df,base_test_df, baseline_features)
            run_deep_models(all_rows, dataset_name, graph_type, train_nodes,test_nodes, train_dict[graph_type]["edges"], test_dict[graph_type]["edges"])

            # Rewriting the csv after every construction means a crash loses at most one construction.
            results = pd.DataFrame(replicate_shared_rows(all_rows, GRAPH_TYPES))
            results.to_csv(CSV_PATH, index=False)

    results = pd.DataFrame(replicate_shared_rows(all_rows, GRAPH_TYPES))
    results.to_csv(CSV_PATH, index=False)
    print(f"\nresults in {CSV_PATH}")
    print(f"predictions in {PREDS_PATH}(shared baselines under '{SHARED_TAG}')")
    print(f"checkpoints in {CKPT_DIR}")


if __name__ == "__main__":
    main()