import numpy as np
import warnings
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
from sklearn.metrics import roc_auc_score


def build_padded_tensors(nodes_df, feature_cols, jet_col="jet_id", label_col=None, max_nodes=None):
    '''
    Pad the long node into fixed size arrays

    :param nodes_df: one row per track
    :param feature_cols: columns to use as per-track features
    :param jet_col: column identifying the jet
    :param label_col: per-jet label column, or None
    :param max_nodes: pad/truncate to this many tracks; default = largest jet
    :return: (X, mask, y, jet_ids) with X (n_jets, max_nodes, n_features),
             mask (n_jets, max_nodes) 1 = real track 0 = padding,
             y (n_jets,) or None, jet_ids (n_jets,)
    '''
    nodes_df = nodes_df.sort_values([jet_col])
    grouped = nodes_df.groupby(jet_col, sort=True)

    jet_ids = np.array(list(grouped.groups.keys()))
    sizes = grouped.size().to_numpy()
    if max_nodes is None:
        max_nodes = int(sizes.max())

    n_jets = len(jet_ids)
    n_features = len(feature_cols)
    X = np.zeros((n_jets, max_nodes, n_features), dtype=np.float32)
    mask = np.zeros((n_jets, max_nodes), dtype=np.float32)
    values = nodes_df[feature_cols].to_numpy(dtype=np.float32)

    start = 0
    for row, size in enumerate(sizes):
        keep = min(size, max_nodes)
        X[row, :keep, :] = values[start:start + keep, :]
        mask[row, :keep] = 1.0
        start = start + size

    y = None
    if label_col is not None:
        y = grouped[label_col].first().to_numpy().astype(np.float32)

    return X, mask, y, jet_ids


def build_padded_edges(edges_df, jet_ids, jet_col="jet_id", max_nodes=None, max_edges=None):
    '''
    Pad edge lists to (n_jets, max_edges, 2) plus a mask.

    :param edges_df: one row per edge with src, dst, jet_col
    :param jet_ids: jet order from build_padded_tensors
    :param jet_col: column identifying the jet
    :param max_nodes: drop edges pointing at truncated nodes, if given
    :param max_edges: pad to this many; default = 2 x the largest edge list
    :return: (edge_index, edge_mask) with edge_index (n_jets, max_edges, 2) int32,
             edge_mask (n_jets, max_edges) 1 = real edge 0 = padding
    '''
    grouped = dict(list(edges_df.groupby(jet_col, sort=True)))
    if max_edges is None:
        longest = 0
        for g in grouped.values():
            if len(g) > longest:
                longest = len(g)
        max_edges = 2 * int(longest)

    n_jets = len(jet_ids)
    edge_index = np.zeros((n_jets, max_edges, 2), dtype=np.int32)
    edge_mask = np.zeros((n_jets, max_edges), dtype=np.float32)

    for row, jet_id in enumerate(jet_ids):
        if jet_id not in grouped:
            continue
        g = grouped[jet_id]
        src_one_way = g["src"].to_numpy()
        dst_one_way = g["dst"].to_numpy()
        src = np.concatenate([src_one_way, dst_one_way])
        dst = np.concatenate([dst_one_way, src_one_way])

        keep = min(len(src), max_edges)
        src_k = src[:keep]
        dst_k = dst[:keep]
        if max_nodes is not None:
            ok = (src_k < max_nodes) & (dst_k < max_nodes)
            src_k = src_k[ok]
            dst_k = dst_k[ok]
            keep = len(src_k)

        edge_index[row, :keep, 0] = src_k
        edge_index[row, :keep, 1] = dst_k
        edge_mask[row, :keep] = 1.0

    return edge_index, edge_mask


def standardize_features(X_train, mask_train, *other_X):
    '''
    Standardize every feature to mean 0, std 1 using training statistics.

    :param X_train: (n_jets, max_nodes, n_features)
    :param mask_train: (n_jets, max_nodes)
    :param other_X: further arrays to transform with the same statistics
    :return: list of standardized arrays, X_train first
    '''
    real = mask_train > 0.5
    mean = X_train[real].mean(axis=0)
    std = X_train[real].std(axis=0)
    std[std < 1e-6] = 1.0

    out = [((X_train - mean) / std).astype(np.float32)]
    for X in other_X:
        out.append(((X - mean) / std).astype(np.float32))
    return out


class SmallTransformer(nn.Module):
    '''
    Set transformer for per jet classification or regression.
    '''

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x, mask):
        h = self.embed(x)
        h = self.encoder(h, src_key_padding_mask=(mask < 0.5))
        h = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(h).squeeze(-1)


class SimpleGNN(nn.Module):
    '''
    Simple message passing GNN for graph level targets
    '''

    def __init__(self, n_features, d_model=64, n_layers=3):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.messages = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(n_layers):
            self.messages.append(nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU()))
            self.updates.append(nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU()))
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, x, mask, edge_index, edge_mask):
        batch, n_nodes, _ = x.shape
        h = self.embed(x)
        src = edge_index[:, :, 0].long()
        dst = edge_index[:, :, 1].long()

        for message_net, update_net in zip(self.messages, self.updates):
            idx_src = src.unsqueeze(-1).expand(-1, -1, h.size(-1))
            idx_dst = dst.unsqueeze(-1).expand(-1, -1, h.size(-1))
            h_src = torch.gather(h, 1, idx_src)
            h_dst = torch.gather(h, 1, idx_dst)

            msg = message_net(torch.cat([h_src, h_dst], dim=-1))
            msg = msg * edge_mask.unsqueeze(-1)

            agg = torch.zeros_like(h).scatter_add(1, idx_dst, msg)
            counts = torch.zeros(batch, n_nodes, 1, device=h.device)
            counts = counts.scatter_add(1, dst.unsqueeze(-1), edge_mask.unsqueeze(-1))
            agg = agg / counts.clamp(min=1)

            h = update_net(torch.cat([h, agg], dim=-1))
            h = h * mask.unsqueeze(-1)

        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_and_score(model, train_tensors, valid_tensors, y_train, y_valid,
                    epochs=30, batch_size=512, lr=1e-3, patience=5, device=None,
                    verbose=True, grad_clip=1.0, use_scheduler=True, weight_decay=1e-4,
                    task="classification"):
    '''
    Train with early stopping on validation score and return best score and predictions.

    :param model: SmallTransformer or SimpleGNN
    :param train_tensors: tuple of training tensors in model's forward order
    :param valid_tensors: same for validation
    :param y_train: (n_train,) targets
    :param y_valid: (n_valid,) targets, same units as y_train
    :param epochs: max epochs
    :param batch_size: batch size
    :param lr: AdamW learning rate
    :param patience: epochs without improvement before stopping
    :param device: compute device; default cuda if available
    :param verbose: print per-epoch score
    :param grad_clip: max grad norm, None to disable
    :param use_scheduler: cosine LR schedule over epochs
    :param weight_decay: AdamW weight decay
    :param task: "classification" (BCE, AUC) or "regression" (MSE, MAE)
    :return: (best_score, best_predictions); AUC or MAE in the units y was passed in
    '''
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    if task == "regression":
        loss_func = nn.MSELoss()
        metric_name = "MAE"
    else:
        loss_func = nn.BCEWithLogitsLoss()
        metric_name = "AUC"

    data_device = train_tensors[0].device
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(data_device)
    n_train = len(y_train)

    best_metric = float("inf")
    best_score = None
    best_predictions = None
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train, device=data_device)
        for start in range(0, n_train, batch_size):
            batch_idx = order[start:start + batch_size]
            inputs = [t[batch_idx].to(device) for t in train_tensors]
            target = y_train_t[batch_idx].to(device)
            optimizer.zero_grad()
            loss = loss_func(model(*inputs), target)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(y_valid), 4096):
                inputs = [t[start:start + 4096].to(device) for t in valid_tensors]
                out = model(*inputs)
                if task == "classification":
                    out = torch.sigmoid(out)
                predictions.append(out.cpu().numpy())
        predictions = np.concatenate(predictions)

        if task == "regression":
            score = float(np.mean(np.abs(y_valid - predictions)))
            metric = score
        else:
            score = roc_auc_score(y_valid, predictions)
            metric = -score

        if scheduler is not None:
            scheduler.step()
        if verbose:
            print(f"Epoch {epoch + 1}, valid {metric_name} = {score:.4f}")

        if metric < best_metric - 1e-4:
            best_metric = metric
            best_score = score
            best_predictions = predictions
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_score, best_predictions