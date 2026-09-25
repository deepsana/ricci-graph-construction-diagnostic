import os

from src.graphs import DEFAULT_GRAPH_TYPES, SWEEP_GRAPH_TYPES
from utils.local_paths import local_paths

PATHS = local_paths()
SHARD_ROOT = PATHS.JETSET_SHARD_ROOT
CACHE_ROOT = PATHS.JETSET_CACHE_DIR
RUN_NAME = "separation"
OUT_DIR = os.path.join(PATHS.RESULTS_ROOT, "jetset", RUN_NAME)

N_WORKERS = 8

# Keep W1 serial. Handing the distribution dicts to joblib workers memory-maps
# them into /dev/shm, and that got the job OOM-killed; n_jobs=1 runs in-process.
W1_N_JOBS = 1
PROGRESS_EVERY = 250

# The default and sweep shards were written by separate runs that dropped
# different jets, so only jets present in both families are scored. Jet ids
# are row indices of the source JetSet split ("jet_<idx>").
FAMILIES = {
    "default": list(DEFAULT_GRAPH_TYPES),
    "sweep": list(SWEEP_GRAPH_TYPES),
}
# One graph type per family, read only for the jet ids in its group names.
ID_PROBE = {"default": "knn_3", "sweep": "knn_4"}
INTERSECT_FAMILIES = ["default", "sweep"]
# The parallel loader adds a per-file multiple of this offset to every jet_id.
LOADER_FILE_OFFSET = 10000000

GRAPH_TYPES = list(DEFAULT_GRAPH_TYPES) + list(SWEEP_GRAPH_TYPES)

# Constructions of the conditioned passes; their cost grows linearly with this count.
COND_GRAPH_TYPES = GRAPH_TYPES

# The full 4.27M-jet sample does not fit in memory. W1 error scales like
# 1/sqrt(N), so 300k jets is precise everywhere except the thinnest joint cells.
MAX_TRAIN_JETS = 300000
SUBSAMPLE_SEED = 0
# Pruned, downcast frames of the kept jets. Delete this directory after
# changing KEEP_NODE_COLS or KEEP_EDGE_COLS.
CACHE_DIR = os.path.join(CACHE_ROOT, f"cache_train_{MAX_TRAIN_JETS}_seed{SUBSAMPLE_SEED}")

# The only columns the scoring code reads.
KEEP_NODE_COLS = ["jet_id", "node_id", "flavor", "truth_vertex_idx", "orc_curvature",
                  "frc_curvature"]
KEEP_EDGE_COLS = ["jet_id", "src", "dst", "flavor", "orc_edge_curvature", "frc_edge_curvature"]
# Columns whose values always fit in int16.
SMALL_INT_COLS = ("node_id", "src", "dst", "flavor", "truth_vertex_idx")

# Inclusive (low, high) bins; a value on a shared edge goes to the first bin.
MULT_BINS = [(3, 5), (6, 9), (10, 14), (15, 19), (20, 30), (31, 40)]
SV_BINS = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 10), (11, 15), (16, 20), (21, 40)]
# With the first-match rule these become {0}, (0, 0.2], (0.2, 0.4], and so on.
SVFRAC_BINS = [(0.0, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
JOINT_COND = "n_x_n_sv"

# Empirical W1 is biased upward roughly like 1/sqrt(N), so a (flavor, bin)
# cell with fewer jets than this is left empty rather than inflating the score.
MIN_JETS_PER_BIN = 200

# A distribution enters a W1 only with at least this many values; 0 disables the check.
MIN_VALUES_PER_SIDE = 50

FLAVORS = [0, 4, 5, 15]
FLAVOR_NAMES = {0: "light", 4: "c", 5: "b", 15: "tau"}
FLAVORPAIR_LABELS = ["b_vs_light", "c_vs_light", "tau_vs_light"]
CURVATURES = ["orc", "frc"]
AGGREGATIONS = ["union", "per_jet_mean"]
NODE_COMPARISONS = ["PV_nodes", "SV_nodes"]
EDGE_COMPARISONS = ["PVPV_edges", "SVSV_edges", "PVSV_edges"]

# Within-flavor subsets compare the PV and SV tracks of one flavor, so their
# only population pair is (PV, SV).
WITHIN_COMPARISONS = {f"PVSV_within_{FLAVOR_NAMES[flavor]}": flavor for flavor in FLAVORS}
WITHIN_PAIR_LABEL = "PV_vs_SV"
ALL_PAIR_LABELS = FLAVORPAIR_LABELS + [WITHIN_PAIR_LABEL]

# Keyed by the sorted flavor tuple, so (0, 5) and (5, 0) are both "b_vs_light".
# Heavy-flavor pairs such as (4, 5): "b_vs_c" can be added here; PAIR_TEXT and
# PAIR_COLOR_IDX already cover them.
PAIR_LABEL = {
    (0, 5): "b_vs_light",
    (0, 4): "c_vs_light",
    (0, 15): "tau_vs_light",
}
PAIR_FLAVORS = {pair_name: pair_flavors for pair_flavors, pair_name in PAIR_LABEL.items()}

PAIR_TEXT = {
    "b_vs_light": r"$b$ vs light",
    "c_vs_light": r"$c$ vs light",
    "tau_vs_light": r"$\tau$ vs light",
    "b_vs_c": r"$b$ vs $c$",
    "tau_vs_c": r"$\tau$ vs $c$",
    "tau_vs_b": r"$\tau$ vs $b$",
    "PV_vs_SV": "PV vs SV",
}
# tab10 color index per pair, chosen so that no construction family uses it.
PAIR_COLOR_IDX = {
    "b_vs_light": 6,
    "c_vs_light": 8,
    "tau_vs_light": 9,
    "b_vs_c": 5,
    "tau_vs_c": 1,
    "tau_vs_b": 3,
    "PV_vs_SV": 7,
}

# (comparison, distribution key, curvature, aggregation). A key reads
# <subset tag>_<curvature>, with a _mean suffix for the per-jet mean.
NODE_SPECS = [
    ("PV_nodes", "pv_orc", "orc", "union"),
    ("PV_nodes", "pv_orc_mean", "orc", "per_jet_mean"),
    ("PV_nodes", "pv_frc", "frc", "union"),
    ("PV_nodes", "pv_frc_mean", "frc", "per_jet_mean"),
    ("SV_nodes", "sv_orc", "orc", "union"),
    ("SV_nodes", "sv_orc_mean", "orc", "per_jet_mean"),
    ("SV_nodes", "sv_frc", "frc", "union"),
    ("SV_nodes", "sv_frc_mean", "frc", "per_jet_mean"),
]
EDGE_SPECS = [
    ("PVPV_edges", "pvpv_orc", "orc", "union"),
    ("PVPV_edges", "pvpv_orc_mean", "orc", "per_jet_mean"),
    ("PVPV_edges", "pvpv_frc", "frc", "union"),
    ("PVPV_edges", "pvpv_frc_mean", "frc", "per_jet_mean"),
    ("SVSV_edges", "svsv_orc", "orc", "union"),
    ("SVSV_edges", "svsv_orc_mean", "orc", "per_jet_mean"),
    ("SVSV_edges", "svsv_frc", "frc", "union"),
    ("SVSV_edges", "svsv_frc_mean", "frc", "per_jet_mean"),
    ("PVSV_edges", "pvsv_orc", "orc", "union"),
    ("PVSV_edges", "pvsv_orc_mean", "orc", "per_jet_mean"),
    ("PVSV_edges", "pvsv_frc", "frc", "union"),
    ("PVSV_edges", "pvsv_frc_mean", "frc", "per_jet_mean"),
]
# All tracks ("all") and all edges ("alle") of a jet.
for _curvature in CURVATURES:
    for _suffix, _aggregation in (("", "union"), ("_mean", "per_jet_mean")):
        NODE_SPECS.append(("all_nodes", f"all_{_curvature}{_suffix}", _curvature, _aggregation))
        EDGE_SPECS.append(("all_edges", f"alle_{_curvature}{_suffix}", _curvature, _aggregation))

# The all-track subsets exist only in the conditioned passes; the
# unconditioned tables have PV/SV rows only.
COND_NODE_COMPARISONS = NODE_COMPARISONS + ["all_nodes"]
COND_EDGE_COMPARISONS = EDGE_COMPARISONS + ["all_edges"]
# Union arrays over all tracks roughly double the memory of the binned data.
# Set to False to keep only the per-jet means, which is what a jet-level
# classifier sees anyway.
ALL_UNION = True

# (comparison, curvature, aggregation) -> distribution key
SPEC_KEY = {(comparison, curvature, aggregation): key for comparison, key, curvature, aggregation in NODE_SPECS + EDGE_SPECS}
DIST_KEYS = [spec[1] for spec in NODE_SPECS + EDGE_SPECS]

# Edge class codes from classify_edges -> subset tag in the distribution keys.
EDGE_TAGS = {0: "pvpv", 1: "svsv", 2: "pvsv"}

# Defaults from configs/graph_configs.yaml for names that omit a parameter.
DEFAULT_RADIUS = 0.3
DEFAULT_QUANTILE = 0.5
DEFAULT_KMAX = 7
P_VALUES = {"neg1": -1, "pos1": 1, "zero": 0}
KT_PATTERN = (r"kt_threshold_(?:quantile|q(\d)p(\d+))(?:_kmax(\d+))?_p_(neg1|pos1|zero)")

FAMILY_ORDER = ["knn", "radius", "kt_threshold", "laman", "unique_k", "fully_connected", "other"]
# tab10 color index per family
FAMILY_COLOR_IDX = {
    "knn": 0,
    "radius": 2,
    "kt_threshold": 1,
    "laman": 3,
    "unique_k": 4,
    "fully_connected": 7,
    "other": 5,
}

COMPARISON_LABELS = {
    "PV_nodes": "PV nodes",
    "SV_nodes": "SV nodes",
    "PVPV_edges": "PV-PV edges",
    "SVSV_edges": "SV-SV edges",
    "PVSV_edges": "PV-SV edges",
    "all_nodes": "all nodes",
    "all_edges": "all edges",
}
COMPARISON_LABELS.update({name: f"PV vs SV within {FLAVOR_NAMES[flavor]}" for name, flavor in WITHIN_COMPARISONS.items()})
COMPARISON_ORDER = ["all_nodes", "all_edges", "PV_nodes", "SV_nodes", "PVPV_edges","SVSV_edges", "PVSV_edges"] + list(WITHIN_COMPARISONS)

AGG_LINESTYLES = {"union": "-", "per_jet_mean": "--"}

# tab10 color index per p: p=-1 orange, p=0 green, p=+1 blue
P_COLOR_IDX = {-1: 1, 0: 2, 1: 0}
# (families, x parameter, x label) for each column of the sweep figure
SWEEP_AXES = [
    (("knn",), "k", "$k$ ($k$NN)"),
    (("radius",), "r", "$r$ (radius)"),
    (("kt_threshold",), "q", "$q$ ($k_T$ threshold)"),
    (("laman", "unique_k"), "k", "$k$ (Unique-$k$; Laman at $k=2$)"),
]
KMAX_MARKERS = {5: "v", 10: "^"}

