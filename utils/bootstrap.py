import gc
import os
import time

import numpy as np
import pandas as pd

from utils.tables import SLICE_KEYS

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import numpy as np
import pandas as pd

from utils.tables import SLICE_KEYS, preserved_sep

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

QUANTITIES = ["S_sep_cond", "S_sep_kept", "share", "preserved_sep"]
TABLE_NAMES = ("per_pair_bins", "per_bin", "per_pair", "per_construction")

# Range of each quantity, used to clip its interval: (lower, upper), None = open.
BOUNDS = {"Sep": (0.0, None), "S_sep_cond": (0.0, None), "S_sep_kept": (0.0, None), "share": (None, 1.0), "preserved_sep": (0.0, None)}

# Caps the (replicates x values) temporaries of the numpy kernel.
NUMPY_CHUNK_ELEMENTS = 4000000


def draw_counts(strata, n_boot, seed):
    """
    How often each point cloud is drawn in each replicate. Draws are made with
    replacement within each stratum, so stratum sizes stay fixed.

    :param strata: group label of every point cloud (flavor for jets, a constant for QM9), in the order of the sorted point-cloud ids
    :return: uint8 array of shape (n_boot, number of point clouds)
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    counts = np.zeros((n_boot, strata.size), dtype=np.uint8)
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        size = members.size
        for replicate in range(n_boot):
            drawn = np.bincount(rng.integers(0, size, size=size), minlength=size)
            # uint8 halves memory; a cloud drawn more than 255 times is vanishingly rare.
            counts[replicate, members] = np.minimum(drawn, 255)
    return counts


def plain(value):
    """
    A numpy scalar as the matching Python scalar, anything else unchanged
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def cell_weight_replicates(population, bin_label, counts):
    """
    Number of point clouds in every (population, bin) cell, in the sample and
    in every replicate. It does not depend on the construction, so it is
    computed once per conditioning variable.

    :param population: population of every point cloud, aligned with counts
    :param bin_label: bin label of every point cloud, None outside every bin
    :param counts: output of draw_counts
    :return: {(population, bin label): (count in the sample, int64 array of the count in every replicate)}
    """
    frame = pd.DataFrame({
        "population": np.asarray(population),
        "bin": np.asarray(bin_label, dtype=object),
        "position": np.arange(len(population)),
    }).dropna(subset=["bin"])
    weights = {}
    for (group_population, group_bin), group in frame.groupby(["population", "bin"], sort=False):
        members = group["position"].to_numpy()
        replicates = counts[:, members].sum(axis=1, dtype=np.int64)
        weights[(plain(group_population), str(group_bin))] = (members.size, replicates)
    return weights


# Weighted W1 on presorted values.
#
# W1 between two 1D distributions is the area between their CDFs:
#     W1 = sum_i |F_A(x_i) - F_B(x_i)| * (x_{i+1} - x_i)
# over the pooled sorted values x_i. Resampling only changes the weight of
# each value, so a replicate is the same sum with count-weighted CDFs.

def w1_replicates_numpy(deltas, rows, is_a, counts):
    """
    W1 between side A and side B of one presorted term in every replicate.

    :param deltas: gaps between consecutive sorted values
    :param rows: column of counts that owns each sorted value
    :param is_a: True where the sorted value belongs to side A
    :param counts: output of draw_counts
    :return: float array of length n_boot, NaN where a side drew no weight
    """
    n_boot = counts.shape[0]
    result = np.full(n_boot, np.nan)
    step = max(1, NUMPY_CHUNK_ELEMENTS // max(rows.size, 1))
    for start in range(0, n_boot, step):
        weight = counts[start:start + step][:, rows].astype(np.float64)
        weight_a = np.where(is_a, weight, 0.0)
        weight_b = weight - weight_a
        cdf_a = np.cumsum(weight_a, axis=1)
        cdf_b = np.cumsum(weight_b, axis=1)
        total_a = cdf_a[:, -1]
        total_b = cdf_b[:, -1]
        valid = (total_a > 0) & (total_b > 0)
        total_a = np.where(valid, total_a, 1.0)
        total_b = np.where(valid, total_b, 1.0)
        gap = np.abs(cdf_a[:, :-1] / total_a[:, None] - cdf_b[:, :-1] / total_b[:, None])
        values = gap @ deltas
        values[~valid] = np.nan
        result[start:start + step] = values
    return result


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def w1_replicates_numba(deltas, rows, is_a, counts):
        """
        Same as w1_replicates_numpy, one thread per replicate and no temporaries
        """
        n_boot = counts.shape[0]
        n_values = rows.size
        result = np.empty(n_boot)
        for replicate in numba.prange(n_boot):
            row_counts = counts[replicate]
            total_a = 0.0
            total_b = 0.0
            for index in range(n_values):
                weight = row_counts[rows[index]]
                if is_a[index]:
                    total_a += weight
                else:
                    total_b += weight
            if total_a == 0.0 or total_b == 0.0:
                result[replicate] = np.nan
                continue
            running_a = 0.0
            running_b = 0.0
            distance = 0.0
            for index in range(n_values - 1):
                weight = row_counts[rows[index]]
                if is_a[index]:
                    running_a += weight
                else:
                    running_b += weight
                distance += abs(running_a / total_a - running_b / total_b) * deltas[index]
            result[replicate] = distance
        return result


def w1_replicates(deltas, rows, is_a, counts):
    if HAVE_NUMBA:
        return w1_replicates_numba(deltas, rows, is_a, counts)
    return w1_replicates_numpy(deltas, rows, is_a, counts)


def term_replicates(values_a, rows_a, values_b, rows_b, counts):
    """
    Point estimate and replicates of one W1 term.
    The values are sorted here, once, and never again per replicate.

    :param rows_a: column of counts for the point cloud of every value of A
    :param rows_b: column of counts for the point cloud of every value of B
    :return: tuple (point estimate, float array of length n_boot)
    """
    n_a = len(values_a)
    n_b = len(values_b)
    values = np.concatenate([values_a, values_b]).astype(np.float64)
    order = np.argsort(values, kind="stable")
    deltas = np.diff(values[order])
    rows = np.concatenate([rows_a, rows_b])[order].astype(np.int32)
    is_a = order < n_a

    cdf_a = np.cumsum(is_a) / n_a
    cdf_b = np.cumsum(~is_a) / n_b
    point = float(np.abs(cdf_a[:-1] - cdf_b[:-1]) @ deltas)
    return point, w1_replicates(deltas, rows, is_a, counts)


def ids_key(dist_key):
    """
    Key of the point-cloud ids stored next to a distribution:
    "pv_orc" -> "pv_ids", "pv_orc_mean" -> "pv_ids_mean", "H_nodes_frc" -> "H_nodes_ids".
    """
    suffix = ""
    stem = dist_key
    if stem.endswith("_mean"):
        suffix = "_mean"
        stem = stem[:-len("_mean")]
    stem = stem.rsplit("_", 1)[0]
    return f"{stem}_ids{suffix}"


def summarize(name, point, replicates, level, lower=None, upper=None):
    """
    Basic bootstrap interval, standard error and bias flag of one quantity.

    The basic (reflected) interval [2 * point - q_high, 2 * point - q_low]
    moves against the bias instead of with it, which matters for W1: its
    finite-sample floor pushes every replicate up. Plain percentiles are kept
    in the "pct" columns. bias_flag is True when the replicate mean is more
    than one standard error from the point estimate; read such an interval as
    a bound.

    :param name: column prefix, e.g. "Sep" or "share"
    :param level: confidence level in percent
    :param lower: smallest possible value of the quantity (0 for a W1)
    :param upper: largest possible value of the quantity (1 for the share)
    :return: dict of "<name> <statistic>" columns
    """
    replicates = np.asarray(replicates, dtype=float)
    finite = replicates[np.isfinite(replicates)]
    defaults = {"point": point, "CI low": np.nan, "CI high": np.nan, "SE": np.nan, "boot mean": np.nan, "bias_flag": False, "pct low": np.nan, "pct high": np.nan}
    columns = {f"{name} {statistic}": value for statistic, value in defaults.items()}
    if finite.size < 2 or not np.isfinite(point):
        return columns

    tail = (100.0 - level) / 2.0
    pct_low, pct_high = np.percentile(finite, [tail, 100.0 - tail])
    low = 2.0 * point - pct_high
    high = 2.0 * point - pct_low
    if lower is not None:
        low = max(low, lower)
    if upper is not None:
        high = min(high, upper)
    standard_error = float(finite.std(ddof=1))
    mean = float(finite.mean())
    columns[f"{name} CI low"] = float(low)
    columns[f"{name} CI high"] = float(high)
    columns[f"{name} SE"] = standard_error
    columns[f"{name} boot mean"] = mean
    columns[f"{name} bias_flag"] = bool(abs(mean - point) > standard_error)
    columns[f"{name} pct low"] = float(pct_low)
    columns[f"{name} pct high"] = float(pct_high)
    return columns


def row_indices(cloud_ids, ids):
    """
    Column of the count matrix for every id.

    :param cloud_ids: sorted ids, one per column of the count matrix
    :return: int32 array of columns; raises KeyError for an unknown id
    """
    rows = np.searchsorted(cloud_ids, ids)
    rows = np.minimum(rows, cloud_ids.size - 1)
    if not np.array_equal(cloud_ids[rows], ids):
        raise KeyError("a value belongs to a point cloud that has no column in the count matrix; build counts from the same point clouds as the cells")
    return rows.astype(np.int32)


# One (slice, pair)

def bin_weight(cell_weights, populations, label, n_boot):
    """
    Bin weight w_b of one pair: the point clouds of both populations in bin b.

    :return: tuple (weight in the sample, int64 array of the weight in every replicate)
    """
    in_sample = 0
    in_replicates = np.zeros(n_boot, dtype=np.int64)
    for population in populations:
        sample_count, replicate_counts = cell_weights.get(
            (population, label), (0, np.zeros(n_boot, dtype=np.int64)))
        in_sample += sample_count
        in_replicates = in_replicates + replicate_counts
    return in_sample, in_replicates


def weighted_cond_replicates(w1_reps, weight_reps):
    """
    S_sep_cond = sum_b w_b W1_b / sum_b w_b in every replicate.
     A bin whose W1
    is NaN in a replicate (one side drew no weight) is left out of that replicate's average.

    :param w1_reps: (bins, n_boot) W1 replicates
    :param weight_reps: (bins, n_boot) bin-weight replicates
    """
    usable = np.isfinite(w1_reps) & (weight_reps > 0)
    numerator = np.where(usable, w1_reps * weight_reps, 0.0).sum(axis=0)
    denominator = np.where(usable, weight_reps, 0.0).sum(axis=0)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1.0), np.nan)


def bootstrap_pair(cells, term, bin_labels, cell_weights, cloud_ids, counts, n_val):
    """
    Point estimates and replicates of one (slice, pair): the per-bin W1, the
    bin weights, S_sep_cond, S_sep_kept, the share and preserved_sep.

    :param cells: {population: {bin label: {key: array}}} of one construction,  with the id arrays of ids_key next to the value arrays
    :param term: dict with side_a and side_b, each a (population, key) tuple
    :param cell_weights: output of cell_weight_replicates
    :param cloud_ids: sorted ids, one per column of counts
    :param n_val: minimum number of values per side of a W1
    :return: dict with "bins", "w1_point", "w1_reps" and a (point, replicates) tuple per QUANTITIES, or None if no bin passes the thresholds
    """
    pop_a, key_a = term["side_a"]
    pop_b, key_b = term["side_b"]
    populations = [pop_a] if pop_b == pop_a else [pop_a, pop_b]
    n_boot = counts.shape[0]

    bins, w1_point, w1_reps, weight_point, weight_reps = [], [], [], [], []
    pooled_a, pooled_rows_a, pooled_b, pooled_rows_b = [], [], [], []
    for label in bin_labels:
        cell_a = cells[pop_a][label]
        cell_b = cells[pop_b][label]
        values_a = cell_a[key_a]
        values_b = cell_b[key_b]
        if len(values_a) < n_val or len(values_b) < n_val:
            continue
        n_clouds, n_clouds_reps = bin_weight(cell_weights, populations, label, n_boot)
        if n_clouds <= 0:
            continue

        rows_a = row_indices(cloud_ids, cell_a[ids_key(key_a)])
        rows_b = row_indices(cloud_ids, cell_b[ids_key(key_b)])
        point, replicates = term_replicates(values_a, rows_a, values_b, rows_b, counts)
        bins.append(label)
        w1_point.append(point)
        w1_reps.append(replicates)
        weight_point.append(n_clouds)
        weight_reps.append(n_clouds_reps)
        pooled_a.append(values_a)
        pooled_rows_a.append(rows_a)
        pooled_b.append(values_b)
        pooled_rows_b.append(rows_b)
    if not bins:
        return None

    w1_point = np.array(w1_point, dtype=float)
    w1_reps = np.vstack(w1_reps)
    weight_point = np.array(weight_point, dtype=float)
    weight_reps = np.vstack(weight_reps).astype(float)

    cond_point = float((w1_point * weight_point).sum() / weight_point.sum())
    cond_reps = weighted_cond_replicates(w1_reps, weight_reps)
    # The bins partition the point clouds, so pooling the kept bins gives the
    # unconditioned distributions restricted to the same clouds.
    kept_point, kept_reps = term_replicates(
        np.concatenate(pooled_a), np.concatenate(pooled_rows_a),
        np.concatenate(pooled_b), np.concatenate(pooled_rows_b), counts)

    return {
        "bins": bins,
        "w1_point": w1_point, "w1_reps": w1_reps,
        **quantities(cond_point, cond_reps, kept_point, kept_reps),
    }


def quantities(cond_point, cond_reps, kept_point, kept_reps):
    """
    The (point, replicates) tuples of every quantity in QUANTITIES.
    """
    preserved_point = float(preserved_sep(cond_point, kept_point))
    preserved_reps = preserved_sep(cond_reps, kept_reps)
    return {
        "S_sep_cond": (cond_point, cond_reps),
        "S_sep_kept": (kept_point, kept_reps),
        "share": (1.0 - preserved_point, 1.0 - preserved_reps),
        "preserved_sep": (preserved_point, preserved_reps),
    }


def replicate_arrays(result):
    """{quantity: replicate array} of a bootstrap_pair result or a reduced dict."""
    return {quantity: result[quantity][1] for quantity in QUANTITIES}


def quantity_row(keys, result, level):
    """
    One row with the interval columns of every quantity in QUANTITIES
    """
    row = dict(keys)
    for quantity in QUANTITIES:
        point, replicates = result[quantity]
        row.update(summarize(quantity, point, replicates, level, *BOUNDS[quantity]))
    return row


def sep_row(keys, point, replicates, level):
    """
    One row with the interval columns of a per-bin separation score
    """
    row = dict(keys)
    row.update(summarize("Sep", point, replicates, level, *BOUNDS["Sep"]))
    return row


# One construction

def mean_over_pairs(members):
    """
    Construction-level quantities of one slice: the mean over pairs of
    S_sep_cond and of S_sep_kept, and the share and preserved_sep of those
    two means, all reduced replicate by replicate.
    """
    cond_point = float(np.mean([member["S_sep_cond"][0] for member in members]))
    kept_point = float(np.mean([member["S_sep_kept"][0] for member in members]))
    cond_reps = np.mean([member["S_sep_cond"][1] for member in members], axis=0)
    kept_reps = np.mean([member["S_sep_kept"][1] for member in members], axis=0)
    return quantities(cond_point, cond_reps, kept_point, kept_reps)


def pooled_bin_rows(keys, members, bin_labels, level):
    """
    Per-bin score of one slice: the mean over the pairs that have a W1 in that bin
    """
    rows = []
    for label in bin_labels:
        points, replicates = [], []
        for member in members:
            if label in member["bins"]:
                position = member["bins"].index(label)
                points.append(member["w1_point"][position])
                replicates.append(member["w1_reps"][position])
        if points:
            rows.append(sep_row(dict(keys, Bin=label, n_terms_sep=len(points)), float(np.mean(points)), np.mean(replicates, axis=0), level))
    return rows


def bootstrap_construction(graph, cells, terms, bin_labels, cell_weights, cloud_ids, counts,
                           cond_var, n_val=50, level=95.0):
    """
    Bootstrap intervals of one construction for one conditioning variable, per pair and per slice. Terms with the same two sides are bootstrapped once.

    :param graph: construction name as it appears in the CSVs (upper case)
    :param cells: {population: {bin label: {key: array}}} of this construction
    :param terms: term dicts with Comparison, Curvature, Aggregation, FlavorPair, side_a and side_b
    :param cell_weights: output of cell_weight_replicates
    :param cloud_ids: sorted ids, one per column of counts
    :param cond_var: conditioning variable, copied into every row
    :param n_val: minimum number of values per side of a W1
    :param level: confidence level in percent
    :return: dict with a DataFrame per TABLE_NAMES and "replicates", a list of (row keys, {quantity: replicate array})
    """
    cloud_ids = np.asarray(cloud_ids)
    rows = {name: [] for name in TABLE_NAMES}
    replicate_rows = []
    results_by_slice = {}
    pair_cache = {}

    for term in terms:
        sides = (term["side_a"], term["side_b"])
        if sides not in pair_cache:
            pair_cache[sides] = bootstrap_pair(cells, term, bin_labels, cell_weights, cloud_ids, counts, n_val)
        result = pair_cache[sides]
        if result is None:
            continue
        slice_key = tuple(term[column] for column in SLICE_KEYS)
        pair_keys = {"Graph": graph, "CondVar": cond_var, **dict(zip(SLICE_KEYS, slice_key)), "FlavorPair": term["FlavorPair"]}

        for position, label in enumerate(result["bins"]):
            rows["per_pair_bins"].append(sep_row(dict(pair_keys, Bin=label), float(result["w1_point"][position]), result["w1_reps"][position], level))
        rows["per_pair"].append(quantity_row(dict(pair_keys, n_bins=len(result["bins"])), result, level))
        replicate_rows.append((pair_keys, replicate_arrays(result)))
        results_by_slice.setdefault(slice_key, []).append(result)

    for slice_key, members in results_by_slice.items():
        keys = {"Graph": graph, "CondVar": cond_var, **dict(zip(SLICE_KEYS, slice_key))}
        reduced = mean_over_pairs(members)
        rows["per_construction"].append(quantity_row(dict(keys, n_pairs=len(members)), reduced, level))
        replicate_rows.append((dict(keys, FlavorPair="ALL"), replicate_arrays(reduced)))
        rows["per_bin"].extend(pooled_bin_rows(keys, members, bin_labels, level))

    tables = {name: pd.DataFrame(rows[name]) for name in TABLE_NAMES}
    tables["replicates"] = replicate_rows
    return tables


# Writing and attaching

def write_outputs(results, save_dir, cond_var):
    """
    Write the interval tables of one conditioning variable, plus the raw
    replicates of S_sep_cond, S_sep_kept and the share. Intervals of
    differences between constructions (share differences, ranks, Kendall's
    tau) come from differences between replicate rows, so the rows are kept.

    :param results: list of bootstrap_construction outputs
    :param cond_var: conditioning variable, used in the file names
    :return: {table name: concatenated DataFrame}, non-empty tables only
    """
    os.makedirs(save_dir, exist_ok=True)
    tables = {}
    for name in TABLE_NAMES:
        frames = [result[name] for result in results if not result[name].empty]
        if not frames:
            continue
        tables[name] = pd.concat(frames, ignore_index=True)
        tables[name].to_csv(os.path.join(save_dir, f"boot_{cond_var}_{name}.csv"), index=False)

    index_rows = []
    arrays = {quantity: [] for quantity in QUANTITIES}
    for result in results:
        for keys, replicates in result["replicates"]:
            index_rows.append(keys)
            for quantity in QUANTITIES:
                arrays[quantity].append(replicates[quantity].astype(np.float32))
    if index_rows:
        pd.DataFrame(index_rows).to_csv(
            os.path.join(save_dir, f"boot_{cond_var}_replicates_index.csv"), index=False)
        stacked = {quantity: np.vstack(arrays[quantity]) for quantity in QUANTITIES}
        np.savez_compressed(os.path.join(save_dir, f"boot_{cond_var}_replicates.npz"), **stacked)
    return tables


def bins_as_text(table):
    """
    Bin as str, in place, so that CSV-read and in-memory labels merge
    """
    if "Bin" in table.columns:
        table["Bin"] = table["Bin"].astype(str)


def attach_columns(csv_path, boot_table, keys, columns):
    """
    Merge interval columns into a score CSV in place, replacing columns of
    the same name from an earlier run.

    :param boot_table: one of the tables returned by write_outputs
    :param keys: columns to merge on
    :param columns: {column in boot_table: column name in the CSV}, see interval_columns
    :return: the updated DataFrame
    """
    scores = pd.read_csv(csv_path)
    stale = [column for column in columns.values() if column in scores.columns]
    scores = scores.drop(columns=stale)
    addition = boot_table[list(keys) + list(columns)].rename(columns=columns)
    bins_as_text(scores)
    bins_as_text(addition)
    merged = scores.merge(addition, on=list(keys), how="left")
    merged.to_csv(csv_path, index=False)
    return merged


def interval_columns(boot_name, csv_name):
    """
    {column of the bootstrap table: column name in the score CSV}
    """
    return {f"{boot_name} CI low": f"{csv_name} CI low", f"{boot_name} CI high": f"{csv_name} CI high",  f"{boot_name} SE": f"{csv_name} SE", f"{boot_name} bias_flag": f"{csv_name} bias_flag"}


def report_agreement(csv_path, boot_table, keys, csv_col, boot_col):
    """
    Print how closely the bootstrap point estimates match the pipeline's.
    Both use the same arithmetic on the same data, so any gap beyond
    rounding means the distributions differ and the intervals are invalid.

    :param csv_col: score column in the CSV
    :param boot_col: point-estimate column in boot_table
    :return: True if the CSV exists
    """
    file_name = os.path.basename(csv_path)
    if not os.path.exists(csv_path):
        print(f"[check] {file_name}: not found, skipped", flush=True)
        return False
    scores = pd.read_csv(csv_path)
    boot = boot_table[list(keys) + [boot_col]].copy()
    bins_as_text(scores)
    bins_as_text(boot)
    merged = scores.merge(boot, on=list(keys), how="inner")
    both = merged[[csv_col, boot_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if both.empty:
        print(f"[check] {file_name}: no rows in common", flush=True)
        return True
    gap = float((both[csv_col] - both[boot_col]).abs().max())
    scale = float(both[csv_col].abs().max())
    verdict = "ok" if gap <= 1e-6 * max(scale, 1e-12) + 1e-9 else "MISMATCH, look before using"
    print(f"[check] {file_name}: max |pipeline - bootstrap| = {gap:.2e} over {len(both)} of {len(scores)} rows (largest value {scale:.3g})  {verdict}",flush=True)
    return True


def check_and_attach(out_dir, jobs, tables, attach):
    """
    Check the point estimates of one pass against the score CSVs and, if
    attach is set, add the interval columns to them. Missing CSVs are skipped.

    :param jobs: list of (file name, bootstrap table name, merge keys, score column in the CSV, point-estimate column in the table, output of interval_columns)
    :param tables: output of write_outputs
    """
    for file_name, table_name, keys, csv_col, boot_col, columns in jobs:
        if table_name not in tables:
            continue
        path = os.path.join(out_dir, file_name)
        found = report_agreement(path, tables[table_name], keys, csv_col, boot_col)
        if found and attach:
            attach_columns(path, tables[table_name], keys, columns)
            print(f"[attach] {file_name}: {sorted(set(columns.values()))[:2]} ...", flush=True)


# Running the passes

def configure_numba(n_threads):
    """Set the thread count of the numba kernel, or report that numpy is used."""
    if HAVE_NUMBA:
        numba.set_num_threads(min(n_threads, numba.config.NUMBA_NUM_THREADS))
        print(f"[cfg] numba kernel on {numba.get_num_threads()} threads", flush=True)
    else:
        print("[cfg] numba not installed: numpy kernel, about 5x slower per thread (pip install numba)", flush=True)


def select_graphs(graphs, wanted):
    """T
    he names in graphs whose lower-cased form is in wanted (any case)
    """
    wanted_lower = {name.lower() for name in wanted}
    return [graph for graph in graphs if graph.lower() in wanted_lower]


def bootstrap_pass(name, graphs, cells_of, terms, bin_labels, cell_weights, cloud_ids, counts,
                   n_val, level, boot_dir):
    """
    Bootstrap every construction of one pass and write its tables. Cells are
    built one construction at a time and freed right after, to bound memory.

    :param name: pass name used in the file names, "uncond" or a conditioning variable
    :param cells_of: function construction -> its cells dict
    :param terms: term dicts, as bootstrap_construction takes them
    :param cell_weights: output of cell_weight_replicates for the pass
    :param cloud_ids: sorted point-cloud ids, one per column of counts
    :return: output of write_outputs
    """
    pass_start = time.time()
    results = []
    for position, graph_type in enumerate(graphs, start=1):
        graph_start = time.time()
        cells = cells_of(graph_type)
        results.append(bootstrap_construction(
            graph_type.upper(), cells, terms, bin_labels, cell_weights, cloud_ids, counts,
            name, n_val=n_val, level=level))
        del cells
        gc.collect()
        elapsed = time.time() - pass_start
        left = elapsed / position * (len(graphs) - position)
        print(f"  [{position}/{len(graphs)}] {graph_type}: {(time.time() - graph_start) / 60:.1f} min  (pass {elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)", flush=True)

    tables = write_outputs(results, boot_dir, name)
    if "per_construction" in tables:
        table = tables["per_construction"]
        flag_col = "S_sep_kept bias_flag" if name == "uncond" else "share bias_flag"
        flagged = int(table[flag_col].sum())
        print(f"  wrote {len(table)} construction rows to {boot_dir}. {flagged} flagged as floor-dominated", flush=True)
    return tables
