import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# JetSet shards, laid out as <JETSET_SHARD_ROOT>/temp_<split>/<family>_<split>_*.h5
JETSET_SHARD_ROOT = "/path/to/JetSet/shards"

# Frozen jet lists and the column cache of the loaded split. Point this at
# the shard root to reuse a cache that an earlier version wrote there.
JETSET_CACHE_DIR = os.path.join(REPO_ROOT, "cache", "jetset")

# QM9 path to where it was downloaded
QM9_DOWNLAODED_DATA = "/path/to/qm9/"

# QM9 shards, as a glob pattern.
QM9_SHARD_GLOB = "/path/to/qm9/graphs/temp_train/all_train_*.h5"

# Tables and figures go under <RESULTS_ROOT>/<experiment>/<run name>
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
