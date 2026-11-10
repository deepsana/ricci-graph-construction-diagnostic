import argparse
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from experiments.qm9 import config
from experiments.qm9.distributions import cond_bin_labels, per_mol_bin
from experiments.qm9.loading import load_qm9
from utils.graph_descriptors import add_graph_descriptors
from utils.tables import SEP_COL

GRAPH_TYPES = config.GRAPH_TYPES
FC_GRAPH = config.FC_GRAPH
JOINT_COND = config.JOINT_COND
MIN_MOLS_PER_BIN = config.MIN_MOLS_PER_BIN
MIN_VALUES_PER_DIST = config.MIN_VALUES_PER_DIST
COND_BINS = {
    "n": config.MULT_BINS,
    "n_H": config.NH_BINS,
    "n_heavy": config.NHEAVY_BINS,
    "h_frac": config.HFRAC_BINS,
    JOINT_COND: (config.NHEAVY_BINS, config.NH_BINS),
}
DEFAULT_OUT = os.path.join(config.PATHS.RESULTS_ROOT, "qm9", "construction-null")


def load_construction(graph_type, descriptors, max_files):
    """
    Load the node table for one construction, adding deg / tri / clu if asked for
    """
    data = load_qm9([graph_type], config.SHARD_GLOB, config.N_WORKERS, max_files)

    needs_graph_descriptors = False
    for d in ("deg", "tri", "clu"):
        if d in descriptors:
            needs_graph_descriptors = True
    if needs_graph_descriptors:
        data = add_graph_descriptors(data, [graph_type], "mol_id")

    return data[graph_type]["nodes"]


AGGREGATIONS = ["union", "per_mol_mean"]
DESCRIPTOR_COL = "{}_curvature"
N_PERM = 200
FC_TOL = 1e-9


def molecule_arrays(nodes, cond, keep):
    """
    Turn the node table into plain numpy arrays (fast, and easy to send to a worker).
    Atoms whose molecule is not in any bin are dropped.
    """
    bins = COND_BINS[cond]
    labels = cond_bin_labels(cond, bins)
    mol_bin = per_mol_bin(nodes, cond, bins)

    code_of = {}
    for i, label in enumerate(labels):
        code_of[label] = i
    mol_code = mol_bin.map(code_of)

    mol_ids = nodes["mol_id"].to_numpy()
    atom_bin = mol_code.reindex(mol_ids).to_numpy()
    inside = ~pd.isna(atom_bin)

    is_h = nodes["is_hydrogen"].to_numpy().astype(bool)

    if keep is None:
        eligible = np.ones(len(nodes), dtype=bool)
    else:
        missing = np.setdiff1d(np.unique(mol_ids), keep.index.to_numpy())
        if missing.size:
            raise SystemExit(f"{missing.size} molecules are not in the keep table, e.g. {missing[:5]}")
        keep_h = keep["keep_h"].reindex(mol_ids).to_numpy().astype(bool)
        keep_x = keep["keep_x"].reindex(mol_ids).to_numpy().astype(bool)
        eligible = (is_h & keep_h) | (~is_h & keep_x)

    # count every molecule in a bin as the normal run does
    counts = mol_code.dropna().astype(int).value_counts()
    mols_in_bin = np.zeros(len(labels), dtype=np.int64)
    mols_in_bin[counts.index.to_numpy()] = counts.to_numpy()

    _, mol_index = np.unique(mol_ids[inside], return_inverse=True)

    return {
        "mol": mol_index.astype(np.int64),
        "is_h": is_h[inside],
        "eligible": eligible[inside],
        "bin": atom_bin[inside].astype(np.int64),
        "mols_in_bin": mols_in_bin,
        "n_mols_total": int(nodes["mol_id"].nunique()),
        "labels": labels,
        "inside": inside,
    }


def side_values(values, mol, mask, aggregation, n_mols):
    """
    Values of one side: one per atom (union) or one mean per molecule
    """
    if aggregation == "union":
        return values[mask]

    m = mol[mask]
    sums = np.bincount(m, weights=values[mask], minlength=n_mols)
    cnt = np.bincount(m, minlength=n_mols)
    has = cnt > 0
    return sums[has] / cnt[has]


def score(values, mol, is_h, eligible, atom_bin, mols_in_bin, aggregation, n_mols):
    """
    S_sep_cond and S_sep_kept for one set of labels.

    :return: (s_cond, s_kept, n_bins_used, n_mols_kept) nan when no bin is usable
    """
    w1s = []
    weights = []
    side_a = []
    side_b = []

    big_enough_bins = np.flatnonzero(mols_in_bin >= MIN_MOLS_PER_BIN)
    for b in big_enough_bins:
        in_bin = eligible & (atom_bin == b)
        a = side_values(values, mol, in_bin & is_h, aggregation, n_mols)
        c = side_values(values, mol, in_bin & ~is_h, aggregation, n_mols)

        if len(a) < MIN_VALUES_PER_DIST or len(c) < MIN_VALUES_PER_DIST:
            continue

        w1s.append(wasserstein_distance(a, c))
        weights.append(mols_in_bin[b])
        side_a.append(a)
        side_b.append(c)

    if not w1s:
        return np.nan, np.nan, 0, 0

    w = np.asarray(weights, dtype=float)
    s_cond = float((np.asarray(w1s) * w).sum() / w.sum())
    s_kept = float(wasserstein_distance(np.concatenate(side_a), np.concatenate(side_b)))

    return s_cond, s_kept, len(w1s), int(w.sum())


def shuffle_within_molecules(is_h, eligible, mol, rng):
    """
    Shuffle the hydrogen / heavy labels among the eligible atoms of each molecule.
    Every molecule keeps its number of each label.
    """
    idx = np.flatnonzero(eligible)
    m = mol[idx]

    by_mol = idx[np.argsort(m, kind="stable")]
    # Same molecule groups as by_mol but in random order inside each molecule.
    shuffled = idx[np.lexsort((rng.random(idx.size), m))]

    out = is_h.copy()
    out[shuffled] = is_h[by_mol]
    return out


def score_construction(task):
    """
    Real scores and the shuffle null for one construction, for every descriptor
    and aggregation. Runs in a worker process.

    :param task: (name, cond, arrays, {descriptor: values}, n_perm, seed)
    :return: list of row dicts
    """
    name, cond, arr, desc_values, n_perm, seed = task
    start = time.time()

    if arr["mol"].size:
        n_mols = int(arr["mol"].max()) + 1
    else:
        n_mols = 0

    mol = arr["mol"]
    is_h = arr["is_h"]
    eligible = arr["eligible"]
    atom_bin = arr["bin"]
    mols_in_bin = arr["mols_in_bin"]

    rows = []
    for descriptor, values in desc_values.items():
        for aggregation in AGGREGATIONS:
            s_cond, s_kept, n_bins, n_kept = score(values, mol, is_h, eligible, atom_bin,
                                                   mols_in_bin, aggregation, n_mols)

            rng_name = f"{name}-{descriptor}-{aggregation}"
            rng = np.random.default_rng([seed, zlib.crc32(rng_name.encode())])

            null_cond = np.full(n_perm, np.nan)
            null_kept = np.full(n_perm, np.nan)

            if np.isfinite(s_cond):
                for k in range(n_perm):
                    perm_h = shuffle_within_molecules(is_h, eligible, mol, rng)
                    result = score(values, mol, perm_h, eligible, atom_bin,
                                   mols_in_bin, aggregation, n_mols)
                    null_cond[k] = result[0]
                    null_kept[k] = result[1]

            finite = null_cond[np.isfinite(null_cond)]

            if s_kept and np.isfinite(s_kept) and s_kept > 0:
                preserved = s_cond / s_kept
            else:
                preserved = np.nan

            if finite.size:
                null_cond_mean = float(np.nanmean(null_cond))
                null_cond_q95 = float(np.nanpercentile(null_cond, 95))
                null_kept_mean = float(np.nanmean(null_kept))
                p_value = float((1 + (finite >= s_cond).sum()) / (1 + finite.size))
                excess = s_cond - float(np.nanmean(null_cond))
            else:
                null_cond_mean = np.nan
                null_cond_q95 = np.nan
                null_kept_mean = np.nan
                p_value = np.nan
                excess = np.nan

            if arr["n_mols_total"]:
                coverage = n_kept / arr["n_mols_total"]
            else:
                coverage = np.nan

            rows.append({
                "construction": name,
                "cond": cond,
                "descriptor": descriptor,
                "aggregation": aggregation,
                "S_sep_cond": s_cond,
                "S_sep_kept": s_kept,
                "preserved": preserved,
                "null_cond_mean": null_cond_mean,
                "null_cond_q95": null_cond_q95,
                "null_kept_mean": null_kept_mean,
                "p_value": p_value,
                "excess": excess,
                "n_perm": int(finite.size),
                "n_bins": n_bins,
                "n_mols_kept": n_kept,
                "coverage": coverage,
            })

    print(f"[done] {name}: {len(rows)} rows in {time.time() - start:.0f} s", flush=True)
    return rows


def check_fc(table):
    """
    On the complete graph, the score must equal its own null.
    """
    fc = table[table["construction"] == FC_GRAPH]
    if fc.empty:
        return
    for _, r in fc.iterrows():
        gap_cond = abs(r["S_sep_cond"] - r["null_cond_mean"])
        gap_kept = abs(r["S_sep_kept"] - r["null_kept_mean"])
        gap = max(gap_cond, gap_kept)
        if gap <= FC_TOL:
            verdict = "ok"
        else:
            verdict = "MISMATCH: the shuffle does not keep the molecules fixed"
        print(f"[check] complete graph, {r['descriptor']} {r['aggregation']}: S_sep_cond {r['S_sep_cond']:.6g}, "
              f"null {r['null_cond_mean']:.6g} ({verdict})", flush=True)


def check_share(table, share_dir, cond):
    """
    Compare our real scores with share_<cond>.csv from the normal run
    """
    path = os.path.join(share_dir, f"share_{cond}.csv")
    if not os.path.exists(path):
        print(f"[check-share] {path} not found; skipped")
        return

    share = pd.read_csv(path)
    share = share[share["Comparison"] == "H_vs_heavy"]

    ours = table.assign(Graph=table["construction"].str.lower(), Curvature=table["descriptor"].str.lower(), Aggregation=table["aggregation"].str.lower())
    share = share.assign(Graph=share["Graph"].str.lower(), Curvature=share["Curvature"].str.lower(), Aggregation=share["Aggregation"].str.lower())

    m = ours.merge(share, on=["Graph", "Curvature", "Aggregation"], suffixes=("", "_run"))
    if m.empty:
        raise SystemExit(f"[check-share] no rows in common with {path}: the names are spelled differently")

    d_cond = (m["S_sep_cond"] - m[SEP_COL]).abs().max()
    d_kept = (m["S_sep_kept"] - m["S_sep_kept_run"]).abs().max()
    if max(d_cond, d_kept) < 1e-9:
        verdict = "ok"
    else:
        verdict = "DIFFERENT: check the settings"
    print(f"[check-share] {len(m)} rows against {path}: max |diff| S_sep_cond {d_cond:.2e}, S_sep_kept {d_kept:.2e} ({verdict})")


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--constructions", nargs="*", default=None, help="default: every construction")
    parser.add_argument("--descriptors", nargs="*", default=["orc"], choices=["orc", "frc", "deg", "tri", "clu"])
    parser.add_argument("--cond", default="n", choices=sorted(COND_BINS))
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="scoring workers")
    parser.add_argument("--max-files", type=int, default=None, help="read only this many shard files (smoke test)")
    parser.add_argument("--keep-table", default=None, help="keep_table.csv of experiments.qm9.run_size_confound")
    parser.add_argument("--check-share", default=None, help="folder of the normal run, to compare the real scores")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="default: DEFAULT_OUT, or the keep table's folder with --keep-table")
    args = parser.parse_args()

    # Save next to the keep table where the confound run's summary looks for it
    if args.keep_table and args.out_dir == DEFAULT_OUT:
        args.out_dir = os.path.dirname(os.path.abspath(args.keep_table))

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"qm9_construction_null_{args.cond}.csv")

    keep = None
    if args.keep_table:
        keep = pd.read_csv(args.keep_table, index_col="mol_id")
        print(f"[cfg] size-confounded populations from {args.keep_table}")
    print(f"[cfg] cond {args.cond}, descriptors {args.descriptors}, {args.n_perm} shuffles, out {out_csv}", flush=True)

    if os.path.exists(out_csv):
        done = pd.read_csv(out_csv)
    else:
        done = pd.DataFrame(columns=["construction"])
    rows = done.to_dict("records")

    # construction is done only when every requested descriptor is already in file
    want = set(args.descriptors)
    have = {}
    old_graphs = done.get("construction", [])
    old_descriptors = done.get("descriptor", [])
    for g, d in zip(old_graphs, old_descriptors):
        if g not in have:
            have[g] = set()
        have[g].add(d)

    if args.constructions:
        names = args.constructions
    else:
        names = GRAPH_TYPES

    todo = []
    for g in names:
        already = have.get(g, set())
        if not want.issubset(already):
            todo.append(g)

    reference = None

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []

        for name in todo:
            load_start = time.time()
            nodes = load_construction(name, args.descriptors, args.max_files)
            nodes = nodes.sort_values(["mol_id", "node_id"], kind="stable").reset_index(drop=True)
            arr = molecule_arrays(nodes, args.cond, keep)

            values = {}
            for d in args.descriptors:
                all_values = nodes[DESCRIPTOR_COL.format(d)].to_numpy(dtype=np.float64)
                values[d] = all_values[arr["inside"]]

            # every construction must have the same atoms in the same molecules
            key = (arr["mol"].tobytes(), arr["is_h"].tobytes(), arr["bin"].tobytes())
            if reference is None:
                reference = key
            elif key != reference:
                raise SystemExit(f"{name} holds different atoms, labels or bins than the first construction")

            del nodes
            arr.pop("inside")

            print(f"[load] {name}: {arr['n_mols_total']} molecules in {time.time() - load_start:.0f} s", flush=True)
            task = (name, args.cond, arr, values, args.n_perm, args.seed)
            futures.append(pool.submit(score_construction, task))

        for fut in futures:
            new_rows = fut.result()
            if new_rows:
                name = new_rows[0]["construction"]
            else:
                name = None

            # drop this construction's old rows for the descriptors we just redid
            kept_rows = []
            for r in rows:
                if not (r["construction"] == name and r.get("descriptor") in want):
                    kept_rows.append(r)
            rows = kept_rows

            rows.extend(new_rows)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)

    check_fc(table)
    if args.check_share:
        check_share(table, args.check_share, args.cond)

    cols = ["construction", "descriptor", "aggregation", "S_sep_cond", "null_cond_mean", "null_cond_q95",
            "p_value", "S_sep_kept", "null_kept_mean", "preserved", "coverage"]
    sorted_table = table[cols].sort_values(["descriptor", "aggregation", "S_sep_cond"], ascending=[True, True, False])
    with pd.option_context("display.width", 240, "display.float_format", "{:.4f}".format):
        print(sorted_table.to_string(index=False))
    print(f"\nwritten {out_csv}")


if __name__ == "__main__":
    main()

