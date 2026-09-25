import glob
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp

from src.qm9_graphs import ALL_GRAPH_TYPES
from utils.diminishing_returns import run_experiment
from utils.diminishing_returns_deep_models import (
    SimpleGNN,
    SmallTransformer,
    build_padded_edges,
    build_padded_tensors,
    standardize_features,
    train_and_score,
)

from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_ROOT = PATHS.QM9_SHARD_GLOB
VALID_ROOT = PATHS.QM9_VALID_GLOB
CACHE_ROOT = PATHS.QM9_CACHE_DIR
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "qm9")

CSV_PATH = os.path.join(OUT_DIR, "diminishing_returns_qm9.csv")
PER_SEED_CSV = os.path.join(OUT_DIR, "qm9_deep_per_seed.csv")
PAIRED_CSV = os.path.join(OUT_DIR, "qm9_deep_paired_stats.csv")
GAIN_CSV = os.path.join(OUT_DIR, "gain_summary_qm9.csv")

RUN_SKLEARN = True
RUN_DEEP = True

GRAPH_TYPES = ALL_GRAPH_TYPES

# Intensive targets only. Extensive targets such as U0 are nearly
# deterministic functions of the atom count, which is the confound this
# project is about.
TARGETS = ["gap", "mu"]
DEEP_TARGETS = ["gap"]

BASELINE_MODE = "full"  # or "num_atoms"

MAX_TRAIN_MOLS = 200000
MAX_VALID_MOLS = 100000
SEED = 44
MODEL_SEEDS = [0, 1, 2, 4, 44]  # sklearn ladder
SEEDS = [0, 1, 2, 4, 44]  # deep models
N_BOOT = 100
# Fewer training molecules than this means stale debug shards were loaded.
MIN_TRAIN_MOLS = 10000

MAX_NODES = 29  # QM9 molecules have at most 29 atoms
EPOCHS = 150
PATIENCE = 25
BATCH_SIZE = 512
PREDICT_BATCH = 4096
DEFAULT_LR = {"transformer": 3e-4, "GNN": 1e-3}
DEEP_MODELS = ("transformer", "GNN")

# 3D position and mass are the QM9 analog of deta, dphi and pt, and Z tells
# the model the element.
BASE_NODE_COLS = ["pos_x", "pos_y", "pos_z", "mass", "Z"]
CURV_NODE_COLS = ["orc_curvature", "frc_curvature"]
DEEP_SETTINGS = [
    ("baseline", BASE_NODE_COLS),
    ("+ curvature", BASE_NODE_COLS + CURV_NODE_COLS),
]

ELEMENT_COUNTS = [("n_H", 1), ("n_C", 6), ("n_N", 7), ("n_O", 8), ("n_F", 9)]
MODEL_ORDER = ["linear", "small MLP", "large MLP", "grad boosting",
               "transformer", "GNN"]
CELL_KEYS = ["dataset", "graph_type", "model", "setting"]
SEED_KEYS = CELL_KEYS + ["seed"]
PAIR_KEYS = ["dataset", "graph_type", "model"]
DIVIDER = "=" * 60


def load_shards(file_list, graph_types):
    """
    Read the nodes, edges and molecule attributes of every construction
    from the shard files.

    :param file_list: shard paths
    :param graph_types: constructions to read from each shard
    :return: {graph_type: {"nodes": df, "edges": df, "mols": df}}
    """
    data = {}
    for graph_type in graph_types:
        node_frames = []
        edge_frames = []
        mol_rows = []
        for path in file_list:
            with h5py.File(path, "r") as shard:
                if graph_type not in shard:
                    continue
                group = shard[graph_type]
                for mol_name in group:
                    mol = group[mol_name]
                    mol_id = int(mol.attrs["mol_id"])

                    node_group = mol["nodes"]
                    node_columns = {}
                    for column in node_group:
                        node_columns[column] = node_group[column][:]
                    node_frame = pd.DataFrame(node_columns)
                    node_frame["mol_id"] = mol_id
                    node_frames.append(node_frame)

                    edge_group = mol["edges"]
                    if "src" in edge_group:
                        edge_frame = pd.DataFrame({
                            "src": edge_group["src"][:].astype(int),
                            "dst": edge_group["dst"][:].astype(int),
                        })
                        edge_frame["mol_id"] = mol_id
                        edge_frames.append(edge_frame)

                    attributes = {}
                    for key in mol.attrs:
                        attributes[key] = mol.attrs[key]
                    mol_rows.append(attributes)

        if not node_frames:
            raise SystemExit(f"FATAL: no molecules found for graph type "
                             f"'{graph_type}' in {len(file_list)} files. "
                             f"Check GRAPH_TYPES against the yaml names "
                             f"used when saving.")
        if edge_frames:
            edges = pd.concat(edge_frames, ignore_index=True)
        else:
            edges = pd.DataFrame(columns=["mol_id", "src", "dst"])
        nodes = pd.concat(node_frames, ignore_index=True)
        mols = pd.DataFrame(mol_rows)
        data[graph_type] = {"nodes": nodes, "edges": edges, "mols": mols}
        print(f"{graph_type}: {len(mols)} molecules, {len(nodes)} atoms")
    return data


def build_molecule_table(nodes_df, mols_df):
    """
    Build the per-molecule feature table for the sklearn ladder: atom
    counts and point-cloud statistics as the baseline, and curvature
    statistics as the extra features.

    :param nodes_df: one row per atom, with mol_id
    :param mols_df: one row per molecule, with the regression targets
    :return: tuple (table, baseline_features, curvature_features), where
        the two lists are column names of table
    """
    grouped = nodes_df.groupby("mol_id")
    table = pd.DataFrame({"num_atoms": grouped.size()})
    for count_name, atomic_number in ELEMENT_COUNTS:
        is_element = nodes_df["Z"] == atomic_number
        table[count_name] = is_element.groupby(nodes_df["mol_id"]).sum()

    baseline_stat_cols = []
    for column in ("pos_x", "pos_y", "pos_z", "mass", "mulliken"):
        if column not in nodes_df.columns:
            continue
        column_stats = grouped[column].agg(["mean", "std", "min", "max"])
        column_stats = column_stats.fillna(0.0).add_prefix(f"{column}_")
        table = table.join(column_stats)
        baseline_stat_cols.extend(column_stats.columns)

    curvature_features = []
    for column in CURV_NODE_COLS:
        if column not in nodes_df.columns:
            raise SystemExit(f"FATAL: node table has no '{column}' -- were "
                             f"curvatures computed when saving? Columns "
                             f"present: {list(nodes_df.columns)}")
        column_stats = grouped[column].agg(["mean", "std", "min", "max",
                                            "median", "skew"])
        column_stats = column_stats.fillna(0.0).add_prefix(f"{column}_")
        table = table.join(column_stats)
        curvature_features.extend(column_stats.columns)

    table = table.reset_index()
    attributes = mols_df.drop(columns=["num_atoms", "smiles"],
                              errors="ignore")
    table = table.merge(attributes, on="mol_id", how="inner")

    baseline_features = ["num_atoms", "n_H", "n_C", "n_N", "n_O", "n_F"]
    baseline_features.extend(baseline_stat_cols)
    return table, baseline_features, curvature_features


def prepare_tensors(train_nodes, train_edges, valid_nodes, valid_edges,
                    feature_cols, need_edges):
    """
    Pad both splits to MAX_NODES atoms and standardize the features with
    training statistics.

    :param train_nodes: training atoms with mol_id and the target column y
    :param train_edges: training edge list with mol_id
    :param valid_nodes: validation atoms, same columns
    :param valid_edges: validation edge list, same columns
    :param feature_cols: per-atom feature columns
    :param need_edges: True for the GNN, which also needs edge tensors
    :return: tuple (train_tensors, y_train, valid_tensors, y_valid), with
        the tensors in forward() order
    """
    x_train, mask_train, y_train, train_ids = build_padded_tensors(
        train_nodes, feature_cols, jet_col="mol_id", label_col="y",
        max_nodes=MAX_NODES)
    x_valid, mask_valid, y_valid, valid_ids = build_padded_tensors(
        valid_nodes, feature_cols, jet_col="mol_id", label_col="y",
        max_nodes=MAX_NODES)
    x_train, x_valid = standardize_features(x_train, mask_train, x_valid)

    train_tensors = [torch.tensor(x_train), torch.tensor(mask_train)]
    valid_tensors = [torch.tensor(x_valid), torch.tensor(mask_valid)]
    if need_edges:
        edges_train, edge_mask_train = build_padded_edges(
            train_edges, train_ids, jet_col="mol_id", max_nodes=MAX_NODES)
        edges_valid, edge_mask_valid = build_padded_edges(
            valid_edges, valid_ids, jet_col="mol_id", max_nodes=MAX_NODES)
        train_tensors.append(torch.tensor(edges_train))
        train_tensors.append(torch.tensor(edge_mask_train))
        valid_tensors.append(torch.tensor(edges_valid))
        valid_tensors.append(torch.tensor(edge_mask_valid))
    return train_tensors, y_train, valid_tensors, y_valid


def holdout_split(n_mols, model_seed):
    """
    Split off 10% of the training molecules for epoch selection, with a
    fresh shuffle per seed. The validation set is never used for early
    stopping.

    :param n_mols: number of training molecules
    :param model_seed: seed for the shuffle
    :return: tuple (stop_idx, fit_idx) of row indices
    """
    order = np.random.default_rng(model_seed).permutation(n_mols)
    n_stop = max(1, int(0.1 * n_mols))
    return order[:n_stop], order[n_stop:]


def bootstrap_mae(y_true, predictions, n_boot=N_BOOT):
    """
    Compute the MAE and its error bar over bootstrap resamples of the
    validation set.

    :param y_true: true values
    :param predictions: predicted values in the same units
    :param n_boot: number of resamples
    :return: tuple (mean MAE, standard deviation of the MAE)
    """
    rng = np.random.default_rng(0)
    scores = []
    n_samples = len(y_true)
    for _ in range(n_boot):
        pick = rng.integers(0, n_samples, n_samples)
        errors = np.abs(y_true[pick] - predictions[pick])
        scores.append(float(np.mean(errors)))
    return float(np.mean(scores)), float(np.std(scores))


def fit_and_evaluate(model, fit_inputs, stop_inputs, valid_inputs, y_fit,
                     y_stop, y_valid, y_mean, y_std, lr, device):
    """
    Train a model with early stopping on the holdout and score the
    best-epoch weights on the validation set in the target's own units.

    :param model: SmallTransformer or SimpleGNN
    :param fit_inputs: training tensors without the holdout
    :param stop_inputs: holdout tensors
    :param valid_inputs: validation tensors
    :param y_fit: standardized training targets
    :param y_stop: standardized holdout targets
    :param y_valid: standardized validation targets
    :param y_mean: training target mean
    :param y_std: training target standard deviation
    :param lr: AdamW learning rate
    :param device: compute device
    :return: tuple (mae, error) from bootstrap_mae
    """
    train_and_score(model, fit_inputs, stop_inputs, y_fit, y_stop,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=lr,
                    patience=PATIENCE, device=device, verbose=False,
                    task="regression")
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(y_valid), PREDICT_BATCH):
            batch = []
            for tensor in valid_inputs:
                batch.append(tensor[start:start + PREDICT_BATCH].to(device))
            predictions.append(model(*batch).cpu().numpy())
    predictions = np.concatenate(predictions)

    y_valid_units = y_valid * y_std + y_mean
    predictions_units = predictions * y_std + y_mean
    return bootstrap_mae(y_valid_units, predictions_units)


def save_row_to_csv(results, model_name, setting, graph_type, dataset):
    """
    Write the last deep-model result into the main CSV, replacing any
    older row of the same cell.

    :param results: in-memory deep-model rows; only the last is written
    :param model_name: model of the row
    :param setting: "baseline" or "+ curvature"
    :param graph_type: construction of the row
    :param dataset: dataset label of the row
    :return: None
    """
    results_df = pd.DataFrame(results)
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        same_cell = ((old["model"] == model_name)
                     & (old["setting"] == setting)
                     & (old["graph_type"] == graph_type)
                     & (old["dataset"] == dataset))
        combined = pd.concat([old[~same_cell], results_df.tail(1)],
                             ignore_index=True)
    else:
        combined = results_df
    combined.to_csv(CSV_PATH, index=False)


def row_key(row, keys):
    """
    Build the identifying tuple of a row.

    :param row: result dict
    :param keys: column names that identify one cell
    :return: tuple of the row's values for keys
    """
    values = []
    for key in keys:
        values.append(row[key])
    return tuple(values)


def replace_cell_rows(rows, new_rows, keys):
    """
    Replace the rows that share their keys with a new row.

    :param rows: existing rows
    :param new_rows: rows to write in
    :param keys: column names that identify one cell
    :return: kept old rows followed by new_rows
    """
    new_keys = set()
    for row in new_rows:
        new_keys.add(row_key(row, keys))
    kept = []
    for row in rows:
        if row_key(row, keys) not in new_keys:
            kept.append(row)
    return kept + new_rows


def cell_seed_maes(per_seed_rows, dataset, graph_type, model_name, setting):
    """
    Look up the per-seed MAEs of one cell, ordered by seed.

    :param per_seed_rows: per-seed result rows
    :param dataset: dataset label
    :param graph_type: construction name
    :param model_name: "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature"
    :return: list of MAEs, or None if the cell has no rows
    """
    seed_maes = []
    for row in per_seed_rows:
        if (row["dataset"] == dataset and row["graph_type"] == graph_type
                and row["model"] == model_name
                and row["setting"] == setting):
            seed_maes.append((row["seed"], row["mae"]))
    if not seed_maes:
        return None
    seed_maes.sort()
    maes = []
    for _, mae in seed_maes:
        maes.append(mae)
    return maes


def find_transformer_baseline(results, dataset):
    """
    Find a transformer baseline row of a dataset among the results.

    :param results: deep-model result rows
    :param dataset: dataset label
    :return: the first matching row, or None
    """
    for row in results:
        if (row["model"] == "transformer" and row["setting"] == "baseline"
                and row["dataset"] == dataset):
            return row
    return None


def run_sklearn_ladder(graph_type, train_tables, valid_tables, done_labels):
    """
    Run the sklearn ladder for every target of one construction.
    run_experiment appends its rows to CSV_PATH itself.

    :param graph_type: construction name
    :param train_tables: training nodes, edges and mols of the construction
    :param valid_tables: validation nodes, edges and mols
    :param done_labels: (dataset, graph_type) pairs already in the CSV
    :return: None
    """
    train_table, full_baseline, curvature_features = build_molecule_table(
        train_tables["nodes"], train_tables["mols"])
    valid_table = build_molecule_table(valid_tables["nodes"],
                                       valid_tables["mols"])[0]

    if BASELINE_MODE == "num_atoms":
        baseline_features = ["num_atoms"]
    else:
        baseline_features = list(full_baseline)

    if graph_type == GRAPH_TYPES[0]:
        print(f"\nbaseline mode: {BASELINE_MODE}")
        print(f"baseline features ({len(baseline_features)}): "
              f"{baseline_features}")
        print(f"curvature features ({len(curvature_features)}): "
              f"{curvature_features}")

    # A fixed seed keeps the subsample the same for every construction.
    if MAX_TRAIN_MOLS is not None and len(train_table) > MAX_TRAIN_MOLS:
        train_table = train_table.sample(MAX_TRAIN_MOLS, random_state=SEED)
    if MAX_VALID_MOLS is not None and len(valid_table) > MAX_VALID_MOLS:
        valid_table = valid_table.sample(MAX_VALID_MOLS, random_state=SEED)

    for target in TARGETS:
        if target not in train_table.columns:
            print(f"target '{target}' not in the shard attrs -- skipping")
            continue
        label = f"qm9_{target}__base_{BASELINE_MODE}__n{len(train_table)}"
        if (label, graph_type) in done_labels:
            print(f"{label}  {graph_type} -- already in csv, skipping")
            continue

        print(f"\n{DIVIDER}")
        print(f"{graph_type}   target = {target}   train: "
              f"{len(train_table)}   valid: {len(valid_table)}")
        print(DIVIDER)
        for model_seed in MODEL_SEEDS:
            if len(MODEL_SEEDS) > 1:
                print(f"model seed:{model_seed}")
            run_experiment(train_table, valid_table,
                           baseline_features=baseline_features,
                           curvature_features=curvature_features,
                           target_col=target,
                           task="regression",
                           n_boot=N_BOOT,
                           dataset_name=label,
                           graph_type=graph_type,
                           csv_path=CSV_PATH,
                           model_seed=model_seed)


def target_by_mol(mols, target):
    """
    Map every molecule id to its target value.

    :param mols: molecule attribute table
    :param target: target column
    :return: {mol_id: target value}
    """
    values = {}
    for mol_id, value in zip(mols["mol_id"], mols[target]):
        values[mol_id] = value
    return values


def prepare_deep_data(train_tables, valid_tables, target):
    """
    Attach the standardized target to every atom and subsample the
    molecules. The target is standardized with training statistics for a
    stable MSE scale and converted back to its units for reporting.

    :param train_tables: training nodes, edges and mols of a construction
    :param valid_tables: validation nodes, edges and mols
    :param target: target column
    :return: dict with the node and edge tables of both splits, y_mean,
        y_std, n_train and n_valid
    """
    train_targets = target_by_mol(train_tables["mols"], target)
    valid_targets = target_by_mol(valid_tables["mols"], target)
    train_values = list(train_targets.values())
    y_mean = float(np.mean(train_values))
    y_std = float(np.std(train_values))
    if y_std < 1e-12:
        raise SystemExit(f"target '{target}' is constant?")
    for mol_id in train_targets:
        train_targets[mol_id] = (train_targets[mol_id] - y_mean) / y_std
    for mol_id in valid_targets:
        valid_targets[mol_id] = (valid_targets[mol_id] - y_mean) / y_std

    train_nodes = train_tables["nodes"].copy()
    valid_nodes = valid_tables["nodes"].copy()
    train_nodes["y"] = train_nodes["mol_id"].map(train_targets)
    valid_nodes["y"] = valid_nodes["mol_id"].map(valid_targets)

    rng = np.random.default_rng(SEED)
    train_ids = train_nodes["mol_id"].unique()
    valid_ids = valid_nodes["mol_id"].unique()
    if MAX_TRAIN_MOLS is not None and len(train_ids) > MAX_TRAIN_MOLS:
        train_ids = rng.choice(train_ids, MAX_TRAIN_MOLS, replace=False)
    if MAX_VALID_MOLS is not None and len(valid_ids) > MAX_VALID_MOLS:
        valid_ids = rng.choice(valid_ids, MAX_VALID_MOLS, replace=False)

    train_edges = train_tables["edges"]
    valid_edges = valid_tables["edges"]
    return {
        "train_nodes": train_nodes[train_nodes["mol_id"].isin(train_ids)],
        "train_edges": train_edges[train_edges["mol_id"].isin(train_ids)],
        "valid_nodes": valid_nodes[valid_nodes["mol_id"].isin(valid_ids)],
        "valid_edges": valid_edges[valid_edges["mol_id"].isin(valid_ids)],
        "y_mean": y_mean,
        "y_std": y_std,
        "n_train": len(train_ids),
        "n_valid": len(valid_ids),
    }


def train_deep_cell(model_name, setting, feature_cols, graph_type,
                    deep_label, target, deep_data, device):
    """
    Train one deep model in one setting for every seed. The reported error
    combines the mean bootstrap error with the spread over seeds.

    :param model_name: "transformer" or "GNN"
    :param setting: "baseline" or "+ curvature"
    :param feature_cols: per-atom feature columns
    :param graph_type: construction name
    :param deep_label: dataset label
    :param target: target column
    :param deep_data: output of prepare_deep_data
    :param device: compute device
    :return: tuple (maes, seed_rows, result_row) with the per-seed MAEs,
        the per-seed CSV rows and the main CSV row
    """
    start_time = time.time()
    train_tensors, y_train, valid_tensors, y_valid = prepare_tensors(
        deep_data["train_nodes"], deep_data["train_edges"],
        deep_data["valid_nodes"], deep_data["valid_edges"], feature_cols,
        model_name == "GNN")
    lr = DEFAULT_LR[model_name]
    print(f"\n  {model_name}  {setting}  (lr={lr})")

    maes = []
    boot_errors = []
    seed_rows = []
    n_fit = len(y_train)
    for run_seed in SEEDS:
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        stop_idx, fit_idx = holdout_split(len(y_train), run_seed)
        fit_rows = torch.from_numpy(fit_idx)
        stop_rows = torch.from_numpy(stop_idx)
        fit_inputs = []
        stop_inputs = []
        for tensor in train_tensors:
            fit_inputs.append(tensor[fit_rows])
            stop_inputs.append(tensor[stop_rows])
        n_fit = len(fit_idx)

        if model_name == "transformer":
            model = SmallTransformer(len(feature_cols))
        else:
            model = SimpleGNN(len(feature_cols))
        mae, boot_error = fit_and_evaluate(
            model, tuple(fit_inputs), tuple(stop_inputs), valid_tensors,
            y_train[fit_idx], y_train[stop_idx], y_valid,
            deep_data["y_mean"], deep_data["y_std"], lr, device)
        maes.append(mae)
        boot_errors.append(boot_error)
        print(f"    seed {run_seed}  MAE = {round(mae, 5)}")
        seed_rows.append({"dataset": deep_label,
                          "graph_type": graph_type,
                          "model": model_name,
                          "setting": setting,
                          "seed": run_seed,
                          "lr": lr,
                          "mae": mae,
                          "boot_error": boot_error})

    score = float(np.mean(maes))
    mean_boot_error = float(np.mean(boot_errors))
    seed_std = float(np.std(maes))
    error = float(np.sqrt(mean_boot_error ** 2 + seed_std ** 2))
    elapsed = time.time() - start_time
    print(f"  -> MAE = {round(score, 5)} +/- {round(error, 5)}   "
          f"({round(elapsed, 1)}s)")

    result_row = {"dataset": deep_label,
                  "graph_type": graph_type,
                  "task": "regression",
                  "target": target,
                  "model": model_name,
                  "setting": setting,
                  "score": score,
                  "error": error,
                  "n_train": n_fit,
                  "n_test": len(y_valid),
                  "n_features": len(feature_cols),
                  "lr": lr,
                  "n_seeds": len(SEEDS),
                  "seed_std": seed_std}
    return maes, seed_rows, result_row


def paired_stats(per_seed, deep_label, graph_type):
    """
    Compare the per-seed MAEs with and without curvature for each deep
    model. The difference is MAE(baseline) - MAE(+ curvature), so a
    positive value means curvature reduced the error.

    :param per_seed: {(model, setting): list of per-seed MAEs}
    :param deep_label: dataset label
    :param graph_type: construction name
    :return: list of paired-statistics rows
    """
    rows = []
    for model_name in DEEP_MODELS:
        base_key = (model_name, "baseline")
        curv_key = (model_name, "+ curvature")
        if base_key not in per_seed or curv_key not in per_seed:
            continue
        base_maes = np.array(per_seed[base_key])
        curv_maes = np.array(per_seed[curv_key])
        if len(base_maes) != len(curv_maes):
            continue

        deltas = base_maes - curv_maes
        mean_delta = float(np.mean(deltas))
        std_delta = 0.0
        stderr = 0.0
        p_value = float("nan")
        if len(deltas) > 1:
            std_delta = float(np.std(deltas, ddof=1))
            stderr = std_delta / np.sqrt(len(deltas))
            if std_delta > 0:
                p_value = float(ttest_1samp(deltas, 0.0).pvalue)

        rows.append({"dataset": deep_label,
                     "graph_type": graph_type,
                     "model": model_name,
                     "n_seeds": len(deltas),
                     "mean_baseline_mae": float(np.mean(base_maes)),
                     "mean_curvature_mae": float(np.mean(curv_maes)),
                     "mae_reduced": mean_delta,
                     "std_delta": std_delta,
                     "stderr_delta": stderr,
                     "p_value": p_value})
        print(f"\n  PAIRED  {model_name}  {graph_type}   MAE reduced by "
              f"curvature = {round(mean_delta, 5)} +/- {round(stderr, 5)}  "
              f"(p={round(p_value, 4)})")
    return rows


def run_deep_models(graph_type, target, train_tables, valid_tables, device,
                    results, done_cells, per_seed_rows, paired_rows):
    """
    Train the transformer and GNN with and without curvature on one
    construction and target. Cells already in the CSV are skipped, and the
    transformer baseline, whose features do not depend on the
    construction, is copied from the first construction.

    :param graph_type: construction name
    :param target: target column
    :param train_tables: training nodes, edges and mols of the construction
    :param valid_tables: validation nodes, edges and mols
    :param device: compute device
    :param results: deep-model result rows, extended in place
    :param done_cells: finished (dataset, graph, model, setting) cells,
        extended in place
    :param per_seed_rows: per-seed result rows
    :param paired_rows: paired-statistics rows
    :return: tuple (per_seed_rows, paired_rows), updated
    """
    if target not in train_tables["mols"].columns:
        print(f"deep target '{target}' not in the shard attrs -- skipping")
        return per_seed_rows, paired_rows

    deep_data = prepare_deep_data(train_tables, valid_tables, target)
    deep_label = f"qm9_{target}__base_full__n{deep_data['n_train']}"
    print(f"\n{DIVIDER}")
    print(f"DEEP  {graph_type}   {target}   -> {deep_label}")
    print(f"    train mols: {deep_data['n_train']}   "
          f"valid mols: {deep_data['n_valid']}")
    print(DIVIDER)

    per_seed = {}
    for setting, feature_cols in DEEP_SETTINGS:
        for model_name in DEEP_MODELS:
            cell = (deep_label, graph_type, model_name, setting)
            if cell in done_cells:
                print(f"\n  {model_name}  {setting}  -- already in csv, "
                      f"skipping")
                maes = cell_seed_maes(per_seed_rows, deep_label, graph_type,
                                      model_name, setting)
                if maes is not None:
                    per_seed[(model_name, setting)] = maes
                continue

            is_shared = (model_name == "transformer"
                         and setting == "baseline"
                         and graph_type != GRAPH_TYPES[0])
            source_row = None
            if is_shared:
                source_row = find_transformer_baseline(results, deep_label)
            if source_row is not None:
                copied_row = dict(source_row)
                copied_row["graph_type"] = graph_type
                results.append(copied_row)
                print(f"\n  transformer  {setting}  -> reusing the "
                      f"{GRAPH_TYPES[0]} result (baseline features are "
                      f"graph-independent)")
                save_row_to_csv(results, model_name, setting, graph_type,
                                deep_label)
                done_cells.add(cell)

                source_maes = cell_seed_maes(per_seed_rows, deep_label,
                                             GRAPH_TYPES[0], model_name,
                                             setting)
                if source_maes is None:
                    continue
                per_seed[(model_name, setting)] = source_maes
                copied_seed_rows = []
                for index, mae in enumerate(source_maes):
                    seed_value = index
                    if index < len(SEEDS):
                        seed_value = SEEDS[index]
                    copied_seed_rows.append({
                        "dataset": deep_label,
                        "graph_type": graph_type,
                        "model": model_name,
                        "setting": setting,
                        "seed": seed_value,
                        "lr": copied_row.get("lr", DEFAULT_LR[model_name]),
                        "mae": mae,
                        "boot_error": np.nan,
                    })
                per_seed_rows = replace_cell_rows(per_seed_rows,
                                                  copied_seed_rows,
                                                  SEED_KEYS)
                pd.DataFrame(per_seed_rows).to_csv(PER_SEED_CSV, index=False)
                continue

            maes, seed_rows, result_row = train_deep_cell(
                model_name, setting, feature_cols, graph_type, deep_label,
                target, deep_data, device)
            per_seed[(model_name, setting)] = maes
            per_seed_rows = replace_cell_rows(per_seed_rows, seed_rows,
                                              SEED_KEYS)
            pd.DataFrame(per_seed_rows).to_csv(PER_SEED_CSV, index=False)
            results.append(result_row)
            save_row_to_csv(results, model_name, setting, graph_type,
                            deep_label)
            done_cells.add(cell)

    new_paired = paired_stats(per_seed, deep_label, graph_type)
    if new_paired:
        paired_rows = replace_cell_rows(paired_rows, new_paired, PAIR_KEYS)
        pd.DataFrame(paired_rows).to_csv(PAIRED_CSV, index=False)
    return per_seed_rows, paired_rows


def write_gain_summary():
    """
    Average every cell of the main CSV over seeds, write the MAE reduction
    from adding curvature per dataset, construction and model, and print it
    as one table per dataset.

    :return: None
    """
    all_results = pd.read_csv(CSV_PATH)
    cell_scores = all_results.groupby(CELL_KEYS, as_index=False)["score"]
    cell_scores = cell_scores.mean()

    base = cell_scores[cell_scores["setting"] == "baseline"]
    curv = cell_scores[cell_scores["setting"] == "+ curvature"]
    summary = base.merge(curv, on=PAIR_KEYS, suffixes=("_base", "_curv"))
    # Lower MAE is better, so the gain is baseline minus curvature.
    summary["gain"] = summary["score_base"] - summary["score_curv"]
    summary.to_csv(GAIN_CSV, index=False)

    print(f"\n{DIVIDER}")
    print("MAE REDUCED BY ADDING CURVATURE (positive = curvature helped,")
    print("mean over seeds)")
    print(DIVIDER)
    for dataset_name in sorted(summary["dataset"].unique()):
        dataset_rows = summary[summary["dataset"] == dataset_name]
        print(f"\n--- {dataset_name} ---")
        table = dataset_rows.pivot(index="graph_type", columns="model",values="gain")
        model_order = []
        for model_name in MODEL_ORDER:
            if model_name in table.columns:
                model_order.append(model_name)
        print(table[model_order].round(4).to_string())


def main():
    """
    Run the sklearn ladder and the deep models on every QM9 construction,
    resuming from the CSVs of an earlier run, and write the gain summary.

    :return: None
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
    print(f"device: {device}")
    if RUN_DEEP and device == "cpu":
        print("WARNING: no GPU found")

    done_labels = set()
    done_cells = set()
    results = []
    if os.path.exists(CSV_PATH):
        previous = pd.read_csv(CSV_PATH)
        is_deep = previous["model"].isin(DEEP_MODELS)
        sklearn_previous = previous[~is_deep]
        deep_previous = previous[is_deep]
        done_labels = set(zip(sklearn_previous["dataset"],
                              sklearn_previous["graph_type"]))
        done_cells = set(zip(deep_previous["dataset"],
                             deep_previous["graph_type"],
                             deep_previous["model"],
                             deep_previous["setting"]))
        results = deep_previous.to_dict("records")
        print(f"resuming: {len(done_labels)} sklearn cells, {len(done_cells)} deep-model cells already in csv")

    per_seed_rows = []
    if os.path.exists(PER_SEED_CSV):
        per_seed_rows = pd.read_csv(PER_SEED_CSV).to_dict("records")
    paired_rows = []
    if os.path.exists(PAIRED_CSV):
        paired_rows = pd.read_csv(PAIRED_CSV).to_dict("records")

    train_files = sorted(glob.glob(SHARD_GLOB))
    valid_files = sorted(glob.glob(VALID_GLOB))
    print(f"train shards: {len(train_files)}   "
          f"valid shards: {len(valid_files)}")
    if not train_files or not valid_files:
        raise SystemExit("FATAL: no shards matched. Check SHARD_GLOB /VALID_GLOB.")

    start_time = time.time()
    train_data = load_shards(train_files, GRAPH_TYPES)
    valid_data = load_shards(valid_files, GRAPH_TYPES)
    print(f"loaded in {round(time.time() - start_time, 1)} s")

    n_loaded = len(train_data[GRAPH_TYPES[0]]["mols"])
    if n_loaded < MIN_TRAIN_MOLS:
        raise SystemExit(f"FATAL: only {n_loaded} train molecules loaded -- these look like stale debug shards")

    for graph_type in GRAPH_TYPES:
        train_tables = train_data[graph_type]
        valid_tables = valid_data[graph_type]
        if RUN_SKLEARN:
            run_sklearn_ladder(graph_type, train_tables, valid_tables,
                               done_labels)
        if not RUN_DEEP:
            continue
        for target in DEEP_TARGETS:
            per_seed_rows, paired_rows = run_deep_models(
                graph_type, target, train_tables, valid_tables, device,
                results, done_cells, per_seed_rows, paired_rows)

    write_gain_summary()
    print("\nFigures: point make_plots_and_summary.py at this csv.")
    print(f"\nresults in {CSV_PATH}")
    print(f"per-seed in {PER_SEED_CSV}")
    print(f"paired in {PAIRED_CSV}")
    print(f"gain summary in {GAIN_CSV}")


if __name__ == "__main__":
    main()