import hashlib
from itertools import combinations
import numpy as np

def node_curvature_digest(nodes, id_col):
    """
    Hash of the node ORC and FRC values, ordered by (cloud id, node id), so
     two constructions share a digest exactly when every node agrees.

    :param nodes: Node table containing id_col, node_id, orc_curvature, frc_curvature
    :param id_col: Point-cloud ID column ("jet_id" or "mol_id")
    :return: Hex digest representing node curvature state
    """
    cloud_id = nodes[id_col].to_numpy(dtype=np.int64)
    node_id = nodes["node_id"].to_numpy(dtype=np.int64)
    order = np.lexsort((node_id, cloud_id))

    digest = hashlib.blake2b(digest_size=16)
    digest.update(cloud_id[order].tobytes())
    digest.update(node_id[order].tobytes())
    for col in ("orc_curvature", "frc_curvature"):
        # Adding 0.0 maps -0.0 to +0.0 to prevent hash discrepancies
        values = nodes[col].to_numpy()[order] + 0.0
        digest.update(values.tobytes())
    return digest.hexdigest()


def duplicate_pairs_from_values(data_dict, graph_types, id_col):
    """
     Construction pairs with identical node curvature, found without any W1

    :param data_dict: Dict mapping graph types to node/edge tables
    :param graph_types: Graph constructions to evaluate
    :param id_col: Point-cloud ID column ("jet_id" or "mol_id")
    :return: List of (graph_a, graph_b) name pairs
    """
    by_digest = {}
    for graph_type in graph_types:
        if graph_type not in data_dict:
            continue
        nodes = data_dict[graph_type]["nodes"]
        if nodes.empty:
            continue
        digest = node_curvature_digest(nodes, id_col)
        by_digest.setdefault(digest, []).append(graph_type)

    pairs = []
    for members in by_digest.values():
        pairs.extend(combinations(members, 2))
    return pairs


def duplicate_pairs_from_table(df_across):
    """
    Construction pairs whose across-construction W1 is 0 on every row

    :param df_across: Across-construction W1 table (Graph_A, Graph_B, W_Distance)
    :return: List of (graph_a, graph_b) name pairs
    """
    if df_across is None or df_across.empty or "W_Distance" not in df_across.columns:
        return []
    max_w1 = df_across.groupby(["Graph_A", "Graph_B"])["W_Distance"].max()
    pairs = []
    for pair, value in max_w1.items():
        if value <= 0.0:
            pairs.append(pair)
    return pairs


def register_duplicate_pairs(duplicates, pairs, reason, pick_drop):
    """
     Add one member of every identical pair to the skip list, in place.

    :param duplicates: lower-cased names to skip; extended in place
    :param pairs: list of (graph_a, graph_b) pairs
    :param reason: why the pair counts as identical, for the log
    :param pick_drop: function (graph_a, graph_b) -> lower-cased name to drop
    """
    for graph_a, graph_b in pairs:
        drop = pick_drop(graph_a, graph_b)
        already_listed = (drop in duplicates or graph_a.lower() in duplicates or graph_b.lower() in duplicates)
        if already_listed:
            print(f"[dup] {graph_a} == {graph_b} ({reason}); already listed", flush=True)
        else:
            print(f"[dup] {graph_a} == {graph_b} ({reason}); adding {drop} to duplicate list", flush=True)
            duplicates.append(drop)
    
    if pairs:
        print(f"[dup] duplicates = {duplicates}", flush=True)
    else:
        print("[dup] no identical construction pairs found", flush=True)
