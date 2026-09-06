import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

MLP_HIDDEN = {"small MLP": (64,),
              "large MLP": (256, 128, 64)}
MLP_EPOCHS = 200
MLP_PATIENCE = 15
MLP_BATCH = 1024
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4

GBM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MLP_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class FFN(nn.Module):
    def __init__(self, n_in, hidden):
        '''
        Feed-forward net: Linear+ReLU blocks and one linear output unit.

        :param n_in: number of input features
        :param hidden: hidden layer sizes, e.g. (256, 128, 64)
        '''
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)


def fit_ffn(X_train, y_train, X_valid, hidden, task, model_seed):
    '''
    Fit an MLP and predict on X_valid.

    :param X_train: (n_train, n_features), standardized
    :param y_train: targets, {0,1} for classification or real values for regression
    :param X_valid: (n_valid, n_features), standardized with the training scaler
    :param hidden: hidden layer sizes, e.g. MLP_HIDDEN["small MLP"]
    :param task: "classification" or "regression"
    :param model_seed: seeds torch init/shuffling and the early-stopping split
    :return: (n_valid,) probabilities for classification, raw predictions for regression
    '''
    torch.manual_seed(model_seed)
    rng = np.random.default_rng(model_seed)

    n = len(y_train)
    order = rng.permutation(n)
    n_stop = max(1, int(0.1 * n))
    stop_idx, fit_idx = order[:n_stop], order[n_stop:]

    X_fit = torch.tensor(X_train[fit_idx], dtype=torch.float32)
    y_fit = torch.tensor(np.asarray(y_train)[fit_idx], dtype=torch.float32)
    X_stop = torch.tensor(X_train[stop_idx], dtype=torch.float32).to(MLP_DEVICE)
    y_stop = np.asarray(y_train)[stop_idx].astype(np.float64)
    X_val = torch.tensor(X_valid, dtype=torch.float32)

    model = FFN(X_train.shape[1], hidden).to(MLP_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    loss_func = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    n_fit = len(y_fit)
    best_metric = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(n_fit)
        for start in range(0, n_fit, MLP_BATCH):
            idx = perm[start:start + MLP_BATCH]
            xb = X_fit[idx].to(MLP_DEVICE)
            yb = y_fit[idx].to(MLP_DEVICE)
            optimizer.zero_grad()
            loss = loss_func(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X_stop)
            if task == "classification":
                p = torch.sigmoid(out).cpu().numpy()
                if len(np.unique(y_stop)) < 2:
                    metric = float(np.mean(np.abs(p - y_stop)))
                else:
                    metric = -roc_auc_score(y_stop, p)
            else:
                p = out.cpu().numpy()
                metric = float(np.mean(np.abs(p - y_stop)))

        if metric < best_metric - 1e-5:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= MLP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_val), 8192):
            out = model(X_val[start:start + 8192].to(MLP_DEVICE))
            if task == "classification":
                out = torch.sigmoid(out)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def fit_linear(X_train, y_train, X_valid, task, model_seed):
    '''
    LogisticRegression for classification, Ridge for regression.

    :param X_train: (n_train, n_features), standardized
    :param y_train: targets
    :param X_valid: (n_valid, n_features), standardized with the training scaler
    :param task: "classification" or "regression"
    :param model_seed: unused
    :return: (n_valid,) probabilities for classification, raw predictions for regression
    '''
    if task == "classification":
        m = LogisticRegression(max_iter=1000)
        m.fit(X_train, y_train)
        return m.predict_proba(X_valid)[:, 1]
    m = Ridge()
    m.fit(X_train, y_train)
    return m.predict(X_valid)


def fit_gbm(X_train, y_train, X_valid, task, model_seed):
    '''
    XGBoost classifier or regressor.

    :param X_train: (n_train, n_features), raw
    :param y_train: targets
    :param X_valid: (n_valid, n_features), raw
    :param task: "classification" or "regression"
    :param model_seed: random_state for tree building
    :return: (n_valid,) probabilities for classification, raw predictions for regression
    '''
    common = dict(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist", device=GBM_DEVICE, random_state=model_seed, n_jobs=-1)
    if task == "classification":
        m = XGBClassifier(eval_metric="logloss", **common)
        m.fit(X_train, y_train)
        return m.predict_proba(X_valid)[:, 1]
    m = XGBRegressor(**common)
    m.fit(X_train, y_train)
    return m.predict(X_valid)


MODELS = (
    ("linear", fit_linear, True),
    ("small MLP", None, True),
    ("large MLP", None, True),
    ("grad boosting", fit_gbm, False),
)


def run_experiment(train_df, test_df, baseline_features, curvature_features, target_col, task="classification",
                   n_boot=50, dataset_name="", graph_type="", csv_path=None, model_seed=0, settings=("baseline", "+ curvature")):
    '''
    Train every model in MODELS and evaluate on test data.

    :param train_df: training rows
    :param test_df: held-out rows
    :param baseline_features: column names of the baseline feature set
    :param curvature_features: column names appended in the "+ curvature" setting
    :param target_col: target column
    :param task: "classification" or "regression"
    :param n_boot: bootstrap resamples
    :param dataset_name: label saved into the csv
    :param graph_type: label for the construction the curvature came from
    :param csv_path: if given, results are appended to this csv
    :param model_seed: seeds the MLPs and the GBM
    :param settings: which settings to run
    :return: DataFrame with one row per (model, setting)
    '''
    both_features = list(baseline_features) + list(curvature_features)

    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    rng = np.random.default_rng(0)
    results = []

    for model_name, fit_func, wants_scaled in MODELS:
        setting_features = {"baseline": list(baseline_features), "+ curvature": both_features}
        for setting_name in settings:
            features = setting_features[setting_name]

            X_train_raw = np.nan_to_num(train_df[features].to_numpy(dtype=np.float64))
            X_test_raw = np.nan_to_num(test_df[features].to_numpy(dtype=np.float64))

            if wants_scaled:
                scaler = StandardScaler().fit(X_train_raw)
                X_train = scaler.transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
            else:
                X_train, X_test = X_train_raw, X_test_raw

            if model_name in MLP_HIDDEN:
                predictions = fit_ffn(X_train, y_train, X_test, hidden=MLP_HIDDEN[model_name], task=task, model_seed=model_seed)
            else:
                predictions = fit_func(X_train, y_train, X_test, task, model_seed)

            boot_scores = []
            n_test = len(y_test)
            for _ in range(n_boot):
                pick = rng.integers(0, n_test, n_test)
                y_boot = y_test[pick]
                p_boot = predictions[pick]
                if task == "classification":
                    if len(np.unique(y_boot)) < 2:
                        continue
                    boot_scores.append(roc_auc_score(y_boot, p_boot))
                else:
                    boot_scores.append(mean_absolute_error(y_boot, p_boot))

            score = float(np.mean(boot_scores))
            error = float(np.std(boot_scores))

            results.append({
                "dataset": dataset_name,
                "graph_type": graph_type,
                "task": task,
                "target": target_col,
                "model": model_name,
                "setting": setting_name,
                "score": score,
                "error": error,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": len(features),
                "model_seed": model_seed,
            })
            print(f"{model_name:<15}{setting_name:<14}score = {score:.4f} +/- {error:.4f}")

    results_df = pd.DataFrame(results)

    if csv_path is not None:
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            results_df = pd.concat([old, results_df], ignore_index=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved results in {csv_path}")

    return results_df