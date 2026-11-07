import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr


NODE_ID_STRIDE = 64 # every node_id, src and dst must be smaller than this numbe -- jets have  at most 40 tracks, molecules at most 29 atoms
DESCRIPTORS = ["deg", "tri", "clu"]



def add_graph_descriptors(data_dict, graph_types, id_col="jet_id"):
    """
    Add the degree, triangle and clustering columns for every graph type.

    :param data_dict: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    :param graph_types: list of graph types to process
    :param id_col: column that says which point cloud a row belongs to ("jet_id" for jets, "mol_id" for molecules)
    :return: the same dict, with the six new columns added
    """
    start = time.time()

    for graph_type in graph_types:
        # skip graph types that are missing or have no nodes
        if graph_type not in data_dict:
            continue
        nodes = data_dict[graph_type]["nodes"]
        edges = data_dict[graph_type]["edges"]
        if nodes.empty:
            continue

        # node_id starts over at 0 in every point cloud (jet or molecule) so
        # "node 3 in cloud A" and "node 3 in cloud B" have same node_id
        # so to tell them apart we use: key = cloud_number * NODE_ID_STRIDE + node_id
        # where cloud_number is 0, 1, 2, ... in the order the clouds first appear
        # this only works if every node_id is smaller than NODE_ID_STRIDE
        cloud_list = pd.Index(pd.unique(nodes[id_col].to_numpy()))
        total_keys = len(cloud_list) * NODE_ID_STRIDE

        node_id = nodes["node_id"].to_numpy(dtype=np.int64)
        if node_id.min() < 0 or node_id.max() >= NODE_ID_STRIDE:
            raise SystemExit(f"FATAL [{graph_type}]: node_id outside [0, {NODE_ID_STRIDE}); raise NODE_ID_STRIDE")
        node_cloud_number = cloud_list.get_indexer(nodes[id_col].to_numpy()).astype(np.int64)
        node_key = node_cloud_number * NODE_ID_STRIDE + node_id

        # do the same for both ends of every edge
        # get_indexer gives -1 when an edge's cloud is not in the node table
        edge_cloud_number = cloud_list.get_indexer(edges[id_col].to_numpy()).astype(np.int64)
        if (edge_cloud_number < 0).any():
            raise SystemExit(f"FATAL [{graph_type}]: {int((edge_cloud_number < 0).sum())} edges belong to "
                             f"point clouds absent from the node table")

        src = edges["src"].to_numpy(dtype=np.int64)
        dst = edges["dst"].to_numpy(dtype=np.int64)
        # only check src/dst if there are any edges (min/max of an empty array fails)
        if len(src):
            smallest = min(src.min(), dst.min())
            largest = max(src.max(), dst.max())
            if smallest < 0 or largest >= NODE_ID_STRIDE:
                raise SystemExit(f"FATAL [{graph_type}]: src/dst outside [0, {NODE_ID_STRIDE})")
        src_key = edge_cloud_number * NODE_ID_STRIDE + src
        dst_key = edge_cloud_number * NODE_ID_STRIDE + dst

        # build the adjacency matrix 
        # adjacency[i, j] = 1 if there is an edge between i and j, else 0.
        # all point clouds go into one big matrix. Clouds never connect to each other so each cloud just sits in its own square block along the diagonal
        # we use a sparse matrix which only stores the 1s because almost every entry is 0
        not_self_loop = src_key != dst_key
        good_src = src_key[not_self_loop]
        good_dst = dst_key[not_self_loop]

        # put every edge in both directions so the matrix is symmetric
        rows = np.concatenate([good_src, good_dst])
        cols = np.concatenate([good_dst, good_src])
        ones = np.ones(len(rows), dtype=np.int32)
        adjacency = sp.csr_matrix((ones, (rows, cols)), shape=(total_keys, total_keys))

        # if an edge was listed more than once its entry got added up to 2 or more
        # set every stored entry back to 1 so each edge only counts once
        adjacency.data[:] = 1

        # Degree = number of neighbors = sum of the node's row
        degree = np.asarray(adjacency.sum(axis=1)).ravel()

        # count common neighbors on every edge (adjacency @ adjacency)[i, j] = number of nodes connected to both i and j
        # Multiplying entry-by-entry with adjacency keeps that number only where i and j are connected themselves (real edges) and sets the rest to 0
        # Each common neighbor k of an edge (i, j) is one triangle i-j-k
        shared = (adjacency @ adjacency).multiply(adjacency).tocsr()

        # node triangles and clustering 
        # Adding up a node's row of `shared` counts each of its triangles twice
        # (once through each of its two edges in that triangle) so divide by 2
        triangles = np.asarray(shared.sum(axis=1)).ravel() // 2

        # Number of pairs of neighbors = deg * (deg - 1) / 2.
        neighbor_pairs = degree * (degree - 1) / 2

        # clustering = triangles/pairs and 0 where there are no pairs (deg < 2)
        clustering = np.zeros(total_keys)
        has_pairs = neighbor_pairs > 0
        clustering[has_pairs] = triangles[has_pairs] / neighbor_pairs[has_pairs]

        # edge triangles and clustering 
        # look up the common neighbor count for each row of the edge table
        # seld loops get 0since matrix has nothing on its diagonal
        edge_triangles = np.asarray(shared[src_key, dst_key]).ravel()

        # Edge clustering = common neighbors / (smaller of the two degrees - 1) and 0 where that bottom number is 0 or less
        smaller_minus_one = np.minimum(degree[src_key], degree[dst_key]) - 1
        edge_clustering = np.zeros(len(smaller_minus_one))
        can_divide = smaller_minus_one > 0
        edge_clustering[can_divide] = edge_triangles[can_divide] / smaller_minus_one[can_divide]

        # node_key / src_key / dst_key pick out the value for each row of tables
        node_deg = degree[node_key]
        nodes["deg_curvature"] = node_deg.astype(np.int16)
        nodes["tri_curvature"] = triangles[node_key].astype(np.int32)
        nodes["clu_curvature"] = clustering[node_key].astype(np.float32)
        edges["deg_edge_curvature"] = (degree[src_key] + degree[dst_key]).astype(np.int16)
        edges["tri_edge_curvature"] = edge_triangles.astype(np.int32)
        edges["clu_edge_curvature"] = edge_clustering.astype(np.float32)
        data_dict[graph_type]["nodes"] = nodes
        data_dict[graph_type]["edges"] = edges

        print(f"[{graph_type:34s}] degree mean {node_deg.mean():.2f} (max {node_deg.max()}, {int((node_deg == 0).sum())} isolated); triangles/node {triangles[node_key].mean():.2f}; clustering mean {clustering[node_key].mean():.3f}", flush=True)

    print(f"[desc] degree, triangles, clustering added in {time.time() - start:.1f} s", flush=True)
    return data_dict


def check_against_frc(data_dict, graph_types):
    """
    Check degree and triangle columns against the FRC values we already have

    For every edge FRC should equal 4 - deg_edge + 3 * tri_edge
    For each graph type, this rebuilds FRC from that formula and prints biggest difference from the stored FRC. 
    If it's 0 everywhere then the degree and triangle columns are correct.

    :param data_dict: {graph_type: {"nodes": DataFrame, "edges": DataFrame}}
    :param graph_types: list of graph types to check
    :return: {graph_type: largest absolute difference}
    """
    worst = {}
    for graph_type in graph_types:
        # Skip graph types that are missing or have no edges
        if graph_type not in data_dict or data_dict[graph_type]["edges"].empty:
            continue
        edges = data_dict[graph_type]["edges"]

        frc = edges["frc_edge_curvature"].to_numpy(dtype=np.float64)
        deg_edge = edges["deg_edge_curvature"].to_numpy(dtype=np.float64)
        tri_edge = edges["tri_edge_curvature"].to_numpy(dtype=np.float64)

        rebuilt_frc = 4 - deg_edge + 3 * tri_edge
        worst[graph_type] = float(np.abs(frc - rebuilt_frc).max())
        print(f"[{graph_type}] max |FRC - (4 - deg_edge + 3 tri_edge)| = {worst[graph_type]:g}", flush=True)
    return worst


def degree_vs_curvature_table(all_scores_df, curvatures, save_dir, sep_col="Separability (higher better)"):
    """
    Does each descriptor rank the graph constructions the same way degree does?

    1. Make a "wide" table: one row per (construction, comparison, aggregation),
       one column per descriptor, holding its separation score S_sep
       (without size control). Saved as sep_degree_vs_curvature_wide.csv.
    2. For each (comparison, aggregation), compute the Spearman rank correlation
       across constructions between degree's scores and each other descriptor's
       scores. Near 1 means "ranks the constructions just like degree does".
       Saved as sep_degree_vs_curvature_spearman.csv.

    Does nothing if "deg" is not one of the descriptors

    :param all_scores_df: scores without size control (columns Graph, Comparison, Curvature, Aggregation, sep_col)
    :param curvatures: the descriptors that were scored, e.g. config.CURVATURES
    :param save_dir: folder to write the CSVs to
    :return: the rank table, or None if "deg" is missing
    """
    if "deg" not in curvatures:
        return None

    # one column per descriptor 
    wide = all_scores_df.pivot_table(index=["Graph", "Comparison", "Aggregation"], columns="Curvature", values=sep_col, aggfunc="first").reset_index()
    wide.columns.name = None   # remove leftover "Curvature" label above the columns
    wide.to_csv(os.path.join(save_dir, "sep_degree_vs_curvature_wide.csv"), index=False)

    # rank correlation with degree
    other_descriptors = [c for c in curvatures if c != "deg"]

    rows = []
    for (comparison, aggregation), group in wide.groupby(["Comparison", "Aggregation"]):
        row = {
            "Comparison": comparison,
            "Aggregation": aggregation,
            "n_graphs": int(group["deg"].notna().sum()),
        }
        for curvature in other_descriptors:
            if curvature not in group.columns:
                continue

            # only use constructions that have a score for both descriptors
            both = group["deg"].notna() & group[curvature].notna()

            # need at least 3 constructions for a correlation to mean anything
            rho, p_value = np.nan, np.nan
            if both.sum() >= 3:
                rho, p_value = spearmanr(group.loc[both, "deg"], group.loc[both, curvature])

            row[f"rho_deg_vs_{curvature}"] = float(rho)
            row[f"p_deg_vs_{curvature}"] = float(p_value)
        rows.append(row)

    rank_table = pd.DataFrame(rows)
    rank_table.to_csv(os.path.join(save_dir, "sep_degree_vs_curvature_spearman.csv"), index=False)
    print("\nSPEARMAN across constructions: degree S_sep vs each descriptor's S_sep")
    print(rank_table.to_string(index=False))
    return rank_table

