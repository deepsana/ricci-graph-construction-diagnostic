import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from experiments.jetset import config
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import load_split

ALL_GRAPH_TYPES = list(config.GRAPH_TYPES)
MAX_TRAIN_JETS = config.MAX_TRAIN_JETS  
SUBSAMPLE_SEED = config.SUBSAMPLE_SEED
OUT_DIR = os.path.join(config.PATHS.RESULTS_ROOT, "jetset", "vertex-recovery")
SIGNAL_FLAVORS = (5, 4, 15)# b, c, tau
SAME_VERTEX_LABELS = (3, 4, 5, 6)
NODE_COLS = ["jet_id", "node_id", "flavor", "truth_vertex_idx", "truth_origin_label"]
EDGE_COLS = ["jet_id", "src", "dst"]
REFERENCES = ("same_vertex",)
FLAVOR_SCOPES = {"b": 5, "c": 4, "tau": 15}
SCOPES = ("signal", "all") + tuple(FLAVOR_SCOPES)
STRIDE = 1000 # used to pack several numbers into one works because node_id < STRIDE (at most 40 tracks per jet)



def reference_group(nodes, reference):
    """
    Give every track a group number.
    Two tracks in the same jet are a reference pair when they have the same group number and that number is >= 0

    :param nodes: node table with truth_vertex_idx and truth_origin_label
    :param reference: "same_vertex"
    :return: int64 array, -1 for tracks that are in no group
    """
    vertex = nodes["truth_vertex_idx"].to_numpy().astype(np.int64)
    label = np.rint(nodes["truth_origin_label"].to_numpy()).astype(np.int64)

    group = np.full(len(nodes), -1, dtype=np.int64)

    if reference == "same_vertex":
        # secondary vertex number but only for tracks from b, c and tau decays
        ok = (vertex > 0) & np.isin(label, SAME_VERTEX_LABELS)
        group[ok] = vertex[ok]
    else:
        raise ValueError(reference)
    return group


def reference_pair_count(jet_ids, group):
    """
    Count the reference pairs. 
    A group with k tracks has k * (k - 1) / 2 pairs and we add that up over every (jet, group).
    """
    ok = group >= 0
    if not np.any(ok):
        return 0

    keys = jet_ids[ok] * STRIDE + group[ok]
    _, counts = np.unique(keys, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def unique_edges(edges):
    """
    Clean up the edge list: treat a-b and b-a as the same edge, remove self
    loops (a-a), and remove duplicates.

    :return: three arrays (jet_id, i, j) with i < j
    """
    jet = edges["jet_id"].to_numpy().astype(np.int64)
    src = edges["src"].to_numpy().astype(np.int64)
    dst = edges["dst"].to_numpy().astype(np.int64)

    # smaller node first so a-b and b-a look the same
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)

    keep = low != high

    # pack (jet, low, high) into one number, e.g. jet 7, low 3, high 12 gives 7003012
    keys = np.unique((jet[keep] * STRIDE + low[keep]) * STRIDE + high[keep])

    high = keys % STRIDE
    low = (keys // STRIDE) % STRIDE
    jet = keys // (STRIDE * STRIDE)
    return jet, low, high


def lookup(nodes_keys, order, values, keys):
    """
    Find the value of each node we ask for using binary search on the
    sorted node keys (jet_id * STRIDE + node_id).

    :param nodes_keys: key of every node
    :param order: positions that sort nodes_keys
    :param values: one value per node
    :param keys: keys of the nodes we want
    :return: the value of each wanted node; -2 if the node is not there
    """
    sorted_keys = nodes_keys[order]
    position = np.searchsorted(sorted_keys, keys)
    position = np.clip(position, 0, len(order) - 1)
    found = sorted_keys[position] == keys

    out = np.full(len(keys), -2, dtype=np.int64)
    out[found] = values[order][position[found]]
    return out


def score_construction(nodes, edges):
    """
    Precision, recall and F1 of one construction for every reference and scope.

    :param nodes: node table (NODE_COLS)
    :param edges: edge table (EDGE_COLS)
    :return: list of row dicts
    """
    node_jet = nodes["jet_id"].to_numpy().astype(np.int64)
    node_key = node_jet * STRIDE + nodes["node_id"].to_numpy().astype(np.int64)
    order = np.argsort(node_key, kind="stable")

    flavor_of_jet = nodes.groupby("jet_id")["flavor"].first()

    jet, low, high = unique_edges(edges)
    edge_flavor = flavor_of_jet.reindex(jet).to_numpy()
    if np.any(pd.isna(edge_flavor)):
        raise SystemExit("an edge belongs to a jet with no nodes")

    # both ends of every edge must be real nodes, the zeros are dummies and only -2 matters
    zeros = np.zeros(len(node_key), dtype=np.int64)
    missing = lookup(node_key, order, zeros, jet * STRIDE + low) == -2
    missing |= lookup(node_key, order, zeros, jet * STRIDE + high) == -2
    if np.any(missing):
        raise SystemExit(f"{int(missing.sum())} edges point to a node_id that is not in the node table")

    rows = []
    node_flavor = nodes["flavor"].to_numpy()
    for reference in REFERENCES:
        group = reference_group(nodes, reference)
        group_low = lookup(node_key, order, group, jet * STRIDE + low)
        group_high = lookup(node_key, order, group, jet * STRIDE + high)

        # an edge is right if both ends are in the same real group
        is_true = (group_low >= 0) & (group_low == group_high)

        for scope in SCOPES:
            if scope == "signal":
                edge_in = np.isin(edge_flavor, SIGNAL_FLAVORS)
                node_in = np.isin(node_flavor, SIGNAL_FLAVORS)
            elif scope in FLAVOR_SCOPES:
                edge_in = edge_flavor == FLAVOR_SCOPES[scope]
                node_in = node_flavor == FLAVOR_SCOPES[scope]
            else:
                edge_in = np.ones(len(jet), dtype=bool)
                node_in = np.ones(len(node_flavor), dtype=bool)

            n_edges = int(edge_in.sum())
            n_true_edges = int((is_true & edge_in).sum())
            n_reference = reference_pair_count(node_jet[node_in], group[node_in])

            if n_edges:
                precision = n_true_edges / n_edges
            else:
                precision = np.nan

            if n_reference:
                recall = n_true_edges / n_reference
            else:
                recall = np.nan

            # if precision or recall is nan this test is False, so f1 becomes 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            rows.append({
                "reference": reference,
                "scope": scope,
                "n_edges": n_edges,
                "n_reference_pairs": n_reference,
                "n_true_edges": n_true_edges,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_jets": int(np.unique(node_jet[node_in]).size),
            })
    return rows


def quick_look(nodes):
    """
    Two quick counts with the same_vertex reference: how many secondary
    vertices each signal jet has, and how many tracks each secondary vertex has.
    """
    signal = nodes[np.isin(nodes["flavor"].to_numpy(), SIGNAL_FLAVORS)]
    group = reference_group(signal, "same_vertex")
    jet = signal["jet_id"].to_numpy().astype(np.int64)
    ok = group >= 0

    # one key per (jet, vertex) and how many tracks each one has
    keys, tracks = np.unique(jet[ok] * STRIDE + group[ok], return_counts=True)

    # jets with no vertex are not in this list
    vertices_per_jet = pd.Series(keys // STRIDE).value_counts()
    n_signal_jets = signal["jet_id"].nunique()

    rows = []

    # secondary vertices per signal jet: 0, 1, 2, 3, 4, >=5
    for k in range(0, 6):
        if k == 0:
            n = n_signal_jets - len(vertices_per_jet)
        elif k < 5:
            n = int((vertices_per_jet == k).sum())
        else:
            n = int((vertices_per_jet >= 5).sum())

        if k < 5:
            value = k
        else:
            value = ">=5"
        rows.append({"quantity": "secondary vertices per signal jet", "value": value,
                     "count": n, "share": n / n_signal_jets})

    # tracks per secondary vertex: 1, 2, 3, 4, 5, >=6
    for k in range(1, 7):
        if k < 6:
            n = int((tracks == k).sum())
            value = k
        else:
            n = int((tracks >= 6).sum())
            value = ">=6"
        rows.append({"quantity": "tracks per secondary vertex",
                         "value": value,
                         "count": n,
                         "share": n / max(len(tracks), 1)})

    return pd.DataFrame(rows)


def rank_correlations(results, separation_path):
    """
    Spearman and Kendall rank correlation between each separation column and F1.
    """
    separation = pd.read_csv(separation_path)

    columns = []
    for c in ("S_sep_cond", "S_sep_kept", "preserved", "excess_over_exact_null"):
        if c in separation.columns:
            columns.append(c)

    # newer null files have one row per descriptor and aggregation so score each piece on its own
    slice_cols = []
    for c in ("descriptor", "aggregation"):
        if c in separation.columns:
            slice_cols.append(c)

    if slice_cols:
        slices = separation.groupby(slice_cols)
    else:
        slices = [((), separation)]

    rows = []
    for slice_key, sep_part in slices:
        if not isinstance(slice_key, tuple):
            slice_key = (slice_key,)
        tags = dict(zip(slice_cols, slice_key))

        for row in _rank_rows(results, sep_part, columns):
            # tag columns first then the correlation columns
            new_row = dict(tags)
            new_row.update(row)
            rows.append(new_row)
    return pd.DataFrame(rows)


def _rank_rows(results, separation, columns):
    """
    Rank correlations of one piece of the separation file against F1, per reference and scope.
    """
    rows = []
    for (reference, scope), part in results.groupby(["reference", "scope"]):
        # A per-pair separation file: pick the pair that matches the scope, or
        # the mean over the three pairs for the pooled scopes.
        sep = separation
        if "pair" in separation.columns:
            if scope in FLAVOR_SCOPES:
                sep = separation[separation["pair"] == f"{scope}_vs_light"]
            else:
                sep = separation.groupby("construction", as_index=False)[columns].mean()
        merged = part.merge(sep[["construction"] + columns], on="construction", how="inner")

        for subset in ("all", "no_fully_connected"):
            if subset == "all":
                data = merged
            else:
                data = merged[merged["construction"] != "fully_connected"]

            for column in columns:
                ok = data[[column, "f1"]].dropna()
                if len(ok) < 3:
                    continue
                rho, rho_p = spearmanr(ok[column], ok["f1"])
                tau, tau_p = kendalltau(ok[column], ok["f1"])
                rows.append({
                    "reference": reference,
                    "scope": scope,
                    "constructions": subset,
                    "separation": column,
                    "n": len(ok),
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "kendall_tau": tau,
                    "kendall_p": tau_p,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--constructions", nargs="*", default=None, help="default: every construction")
    parser.add_argument("--quick-look", action="store_true", help="only the vertex counts, on one construction")
    parser.add_argument("--separation", default=None, help="construction_null.csv to correlate with F1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-jets", type=int, default=MAX_TRAIN_JETS)
    parser.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.constructions:
        names = args.constructions
    else:
        names = list(ALL_GRAPH_TYPES)

    kept_ids = get_kept_ids("train", args.max_jets, args.seed)

    # quick look only: count vertices on the first construction and stop
    if args.quick_look:
        loaded = load_split("train", names[:1], kept_ids, args.workers, node_cols=NODE_COLS, edge_cols=EDGE_COLS)
        table = quick_look(loaded[names[0]]["nodes"])
        table.to_csv(os.path.join(args.out_dir, "vertex_quick_look.csv"), index=False)
        print(table.to_string(index=False))
        return

    # read old results so finished constructions are skipped
    out_csv = os.path.join(args.out_dir, "vertex_recovery.csv")
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        # runs before 2026-11-09 called the signal scope "heavy"
        old["scope"] = old["scope"].replace({"heavy": "signal"})
        # drop rows of references that no longer exist, like "decay_chain"
        stale = ~old["reference"].isin(REFERENCES)
        if stale.any():
            print(f"dropping {int(stale.sum())} rows of removed references: {sorted(old.loc[stale, 'reference'].unique())}")
            old = old[~stale]
        # a construction is done only if it has every scope (older runs lack b/c/tau)
        have = old.groupby("construction")["scope"].apply(set)
        done = {c for c, scopes in have.items() if set(SCOPES) <= scopes}
        partial = sorted(set(have.index) - done)
        if partial:
            print(f"rescoring {len(partial)} constructions that lack the per-flavor scopes: {partial}")
            old = old[old["construction"].isin(done)]
        rows = old.to_dict("records")
    else:
        rows = []
        done = set()

    # the reference pair counts only depend on the jets so they must match for every construction
    reference_counts = None

    for name in names:
        if name in done:
            print(f"{name}: done, skipping", flush=True)
            continue

        start = time.time()
        loaded = load_split("train", [name], kept_ids, args.workers, node_cols=NODE_COLS, edge_cols=EDGE_COLS)
        nodes = loaded[name]["nodes"]
        edges = loaded[name]["edges"]

        if reference_counts is None and not rows:
            quick_look(nodes).to_csv(os.path.join(args.out_dir, "vertex_quick_look.csv"), index=False)

        new_rows = score_construction(nodes, edges)

        counts_list = []
        for row in new_rows:
            counts_list.append(row["n_reference_pairs"])
        counts = tuple(counts_list)

        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise SystemExit(f"{name}: reference pair counts differ from the first construction (different jets?)")

        for row in new_rows:
            row["construction"] = name
            row["seconds"] = time.time() - start
        rows.extend(new_rows)

        # save after every construction so nothing is lost if the run stops
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        summary = pd.DataFrame(new_rows)[["reference", "scope", "precision", "recall", "f1"]]
        print(summary.to_string(index=False, float_format="{:.3f}".format), flush=True)

        del loaded, nodes, edges

    # optional rank correlations with the separation scores
    results = pd.DataFrame(rows)
    if args.separation:
        correlations = rank_correlations(results, args.separation)
        correlations.to_csv(os.path.join(args.out_dir, "vertex_recovery_vs_sep.csv"), index=False)
        print("\n" + correlations.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nresults in {out_csv}")


if __name__ == "__main__":
    main()

