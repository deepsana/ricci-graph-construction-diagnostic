import os

from src.qm9_graphs import ALL_GRAPH_TYPES
from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_GLOB = PATHS.QM9_SHARD_GLOB
RUN_NAME = "separation"
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "qm9", RUN_NAME)

N_WORKERS = 60
MAX_FILES = None

# Every pair of constructions is compared in the sensitivity stage, so it
# grows like M^2: 22 constructions give 231 pairs.
GRAPH_TYPES = ALL_GRAPH_TYPES

# Bins are inclusive on both ends. A value on a shared edge goes to the
# first bin that contains it.
MULT_BINS = [(3, 6), (7, 10), (11, 14), (15, 18), (19, 22), (23, 29)]
NH_BINS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 20)]
NHEAVY_BINS = [(1, 4), (5, 6), (7, 7), (8, 8), (9, 9)]
HFRAC_BINS = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
# (n_heavy, n_H) carries the same information as (n, n_H), but the
# heavy-atom grid is much better populated.
JOINT_COND = "n_heavy_x_n_H"

# Empirical W1 is biased upward roughly like 1/sqrt(N), so thin bins and
# sparse distributions are left out.
MIN_MOLS_PER_BIN = 200
MIN_VALUES_PER_DIST = 50

# FRC on the complete graph equals n, which would dominate every other
# construction's sensitivity average.
EXCLUDE_FC_FROM_SENS_FOR = {"frc"}
FC_GRAPH = "fully_connected"
# Skip the across-construction W1 (the M^2 stage) and S_sens. S_sep, the
# share and the unconditioned scores do not use it.
SKIP_SENSITIVITY = True

NODE_SUBSETS = ["all_nodes","H_nodes", "heavy_nodes", "ring_nodes", "chain_nodes" "terminal_nodes", "branch_nodes"]
EDGE_SUBSETS = ["all_edges","bond_edges", "nonbond_edges", "HH_edges", "Hheavy_edges", "heavyheavy_edges"]
SUBSET_PAIRS = {
    "H_vs_heavy": ("H_nodes", "heavy_nodes"),
    "ring_vs_chain": ("ring_nodes", "chain_nodes"),
    "terminal_vs_branch": ("terminal_nodes", "branch_nodes"),
    "bond_vs_nonbond": ("bond_edges", "nonbond_edges"),
    "Hheavy_vs_heavyheavy": ("Hheavy_edges", "heavyheavy_edges"),
    "HH_vs_heavyheavy": ("HH_edges", "heavyheavy_edges"),
}
SENS_ONLY_COMPARISONS = {"all_nodes": ("all_nodes",),"all_edges": ("all_edges",),}
# Store the union arrays of all_nodes and all_edges too, not only their
# per-molecule means.
ALL_CONCAT = True

NODE_COMPARISONS = ["H_vs_heavy", "ring_vs_chain", "terminal_vs_branch"]
EDGE_COMPARISONS = ["bond_vs_nonbond", "Hheavy_vs_heavyheavy", "HH_vs_heavyheavy"]
NODE_SCORED = ["all_nodes"] + NODE_COMPARISONS
EDGE_SCORED = ["all_edges"] + EDGE_COMPARISONS
COMPARISON_ORDER = NODE_SCORED + EDGE_SCORED

# S_sep of the "all" comparisons pools every pair of that kind.
SEP_POOL = {
    "all_nodes": NODE_COMPARISONS,
    "all_edges": EDGE_COMPARISONS,
}
CURVATURES = ["orc", "frc"]
AGGREGATIONS = ["union", "per_mol_mean"]
AGGREGATION_SUFFIX = {"union": "", "per_mol_mean": "_mean"}

NODE_COLS = ["mol_id", "node_id", "is_hydrogen", "in_ring", "heavy_degree", "orc_curvature", "frc_curvature"]
EDGE_COLS = ["mol_id", "src", "dst", "is_true_bond", "orc_edge_curvature", "frc_edge_curvature"]


def dist_key(subset, curvature, aggregation):
    """
    Key of one distribution, e.g. "ring_nodes_orc_mean".

    :param subset: subset name
    :param curvature: "orc" or "frc"
    :param aggregation: "union" or "per_mol_mean"
    :return: the distribution key
    """
    return f"{subset}_{curvature}{AGGREGATION_SUFFIX[aggregation]}"


def comparison_subsets(comparison):
    """
    Subsets whose across-construction W1 rows make up S_sens.

    :param comparison: a key of SUBSET_PAIRS or SENS_ONLY_COMPARISONS
    :return: list of subset names
    """
    if comparison in SENS_ONLY_COMPARISONS:
        return list(SENS_ONLY_COMPARISONS[comparison])
    return list(SUBSET_PAIRS[comparison])


def build_specs(subsets):
    """
    The (subset, distribution key, curvature, aggregation) tuples of a list
    of subsets.

    :param subsets: subset names, e.g. NODE_SUBSETS
    :return: list of (subset, key, curvature, aggregation) tuples
    """
    specs = []
    for subset in subsets:
        for curvature in CURVATURES:
            for aggregation in AGGREGATIONS:
                key = dist_key(subset, curvature, aggregation)
                specs.append((subset, key, curvature, aggregation))
    return specs


NODE_SPECS = build_specs(NODE_SUBSETS)
EDGE_SPECS = build_specs(EDGE_SUBSETS)

