import argparse
import os
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import wasserstein_distance

from experiments.jetset import config
from experiments.jetset.cache import read_cached_split
from experiments.jetset.jets import get_kept_ids
from experiments.jetset.loading import load_split, prune_split, shrink_split
from utils.graph_descriptors import add_graph_descriptors
from utils.tables import SEP_COL

GRAPH_TYPES = list(config.GRAPH_TYPES)
MULT_BINS = config.MULT_BINS  # bins for n (number of tracks)
SV_BINS = config.SV_BINS   # bins for n_sv (number of SV tracks)
JOINT_COND = config.JOINT_COND  # name for "bin by both n and n_sv"
MIN_JETS_PER_BIN = config.MIN_JETS_PER_BIN
MIN_VALUES_PER_SIDE = config.MIN_VALUES_PER_SIDE
MAX_TRAIN_JETS = config.MAX_TRAIN_JETS
SUBSAMPLE_SEED = config.SUBSAMPLE_SEED
OUT_DIR = os.path.join(config.PATHS.RESULTS_ROOT, "jetset", "construction-null")


def load_construction(name, kept_ids, descriptors, n_workers):
    """
    Load node table for one graph construction
    It tries the cache first and reads the shard files only if neede
    If degree / triangle / clustering were asked for, those columns are added.

    :return: node DataFrame with jet_id, node_id, flavor, truth_vertex_idx and <descriptor>_curvature
    """
    data = read_cached_split(config.CACHE_DIR, [name], kept_ids, config.KEEP_NODE_COLS, config.KEEP_EDGE_COLS)

    if data is None:
        data = load_split("train", [name], kept_ids, n_workers,
                          node_cols=config.KEEP_NODE_COLS, edge_cols=config.KEEP_EDGE_COLS)
        data = prune_split(data)
        data = shrink_split(data)

    data = {name: data[name]}

    # Degree, triangles and clustering are not stored in the files so compute them
    needs_graph_descriptors = False
    for d in ("deg", "tri", "clu"):
        if d in descriptors:
            needs_graph_descriptors = True
    if needs_graph_descriptors:
        data = add_graph_descriptors(data, [name], "jet_id")

    return data[name]["nodes"]

FC_GRAPH = "fully_connected"
FLAVORS = [0, 4, 5, 15]               
PAIRS = [(5, 0), (4, 0), (15, 0)]    
PAIR_LABEL = {(5, 0): "b_vs_light", (4, 0): "c_vs_light", (15, 0): "tau_vs_light"}
COND_CHOICES = ["n", "n_sv", JOINT_COND]
DESCRIPTOR_COL = "{}_curvature"       
FC_TOL = 1e-9                    


# Putting jets into size bins
def first_bin(value, bins):
    """
    Return the index of the first (low, high) range that contains value.
    Return -1 if no range contains it. (Same rule as the scoring run.)
    """
    for k, (lo, hi) in enumerate(bins):
        if lo <= value <= hi:
            return k
    return -1


def bin_codes(n, n_sv, cond):
    """
    Give every jet a bin number for the chosen size variable.
    -1 means the jet does not fit in any bin.

    For JOINT_COND the bin number is: (n bin) * len(SV_BINS) + (n_sv bin).
    This turns a pair of bin numbers into one unique number.
    """
    # bin number for n of every jet
    m_list = []
    for v in n:
        m_list.append(first_bin(v, MULT_BINS))
    m = np.array(m_list)

    # Binbinnumber for n_sv of every jet
    s_list = []
    for v in n_sv:
        s_list.append(first_bin(v, SV_BINS))
    s = np.array(s_list)

    if cond == "n":
        return m
    if cond == "n_sv":
        return s

    # joint bin: only valid if both bins are valid, otherwise -1
    both_valid = (m >= 0) & (s >= 0)
    joint = m * len(SV_BINS) + s
    return np.where(both_valid, joint, -1)


def jet_arrays(nodes, descriptors, cond):
    """
    Turn the node table into plain numpy arrays, which are much faster to
    work with. Jets are sorted by (bin, jet_id), and the nodes are grouped
    by jet in that same order.

    :return: dict with per-jet arrays (jet_id, flavor, n, n_sv, bin, exact cell),
        per-node arrays (jet index, values), per-jet means, and the jet / node
        range of every bin
    """
    nodes = nodes.sort_values(["jet_id", "node_id"], kind="stable")
    jet_id_node = nodes["jet_id"].to_numpy()

    # jet_ids: each jet once
    # first:where each jet's first node is
    # n: how many nodes (tracks) each jet has
    jet_ids, first, n = np.unique(jet_id_node, return_index=True, return_counts=True)
    num_jets = len(jet_ids)

    tvi = nodes["truth_vertex_idx"].to_numpy()

    # For every node, which jet (0, 1, 2, ...) it belongs to
    node_jet = np.repeat(np.arange(num_jets), n)

    # n_sv = number of nodes in each jet with truth_vertex_idx > 0.
    # bincount with weights adds up the 1s (True) for each jet.
    n_sv = np.bincount(node_jet, weights=(tvi > 0), minlength=num_jets).astype(np.int64)

    # Every node of a jet has the same flavor, so take it from the first node.
    flavor = nodes["flavor"].to_numpy()[first].astype(np.int64)

    bins = bin_codes(n, n_sv, cond)

    # Order jets by bin -- "stable" keeps them in jet_id order inside a bin
    jet_order = np.argsort(bins, kind="stable")

    starts = np.r_[0, np.cumsum(n)[:-1]]

    # New node order: go through the jets in their new order,
    # and list each jet's nodes.
    node_pieces = []
    for j in jet_order:
        node_pieces.append(np.arange(starts[j], starts[j] + n[j]))
    node_order = np.concatenate(node_pieces)

    # rank[old jet index] = new jet index
    rank = np.empty(num_jets, dtype=np.int64)
    rank[jet_order] = np.arange(num_jets)

    out = {
        "jet_id": jet_ids[jet_order],
        "flavor": flavor[jet_order],
        "n": n[jet_order],
        "n_sv": n_sv[jet_order],
        "bin": bins[jet_order],
        "node_jet": rank[node_jet[node_order]],   # for every node, its jet's new index
        "values": {},
        "means": {},
    }

    # for each descriptor the per-node values and the per-jet mean
    for d in descriptors:
        v = nodes[DESCRIPTOR_COL.format(d)].to_numpy(dtype=np.float64)
        sum_per_jet = np.bincount(node_jet, weights=v, minlength=num_jets)
        mean_per_jet = sum_per_jet / n
        out["means"][d] = mean_per_jet[jet_order]
        out["values"][d] = v[node_order]

    # "Exact cell" = the exact (n, n_sv) of a jet, packed into one number --jets outside the bins get -1
    out["exact_cell"] = np.where(out["bin"] >= 0, out["n"] * 1000 + out["n_sv"], -1)

    # For every bin that has jets, find where its jets and nodes start and end.
    # Because everything is sorted by bin, each bin is one continuous block,
    # and searchsorted can find the edges of that block.
    jb = out["bin"]  # bin of every jet
    nb = jb[out["node_jet"]]  # bin of every node
    out["ranges"] = []
    for b in np.unique(jb[jb >= 0]):
        js, je = np.searchsorted(jb, [b, b + 1], side="left")  # jet start, jet end
        ns, ne = np.searchsorted(nb, [b, b + 1], side="left")  # node start, node end
        out["ranges"].append((int(b), int(js), int(je), int(ns), int(ne)))

    return out

def scores(arr, descriptor, aggregation, labels, counts, min_jets, min_values):
    """
    Compute size-controlled scores for one set of labels and one set of
    jet counts. counts is 1 for every jet for the real data and the shuffles;
    for the bootstrap it is how many times each jet was picked.

    :return: dict with S_sep_cond, S_sep_kept, preserved (means over pairs),
        n_jets_kept, n_jets_total, and "pairs" with the per-pair scores
    """
    union = (aggregation == "union")
    node_jet = arr["node_jet"]
    out = {"pairs": {}}

    cond_per_pair = []
    kept_per_pair = []
    jets_kept = 0
    jets_total = 0

    #count how many jets of each flavor are in each bin
    cell = {}
    for b, js, je, _, _ in arr["ranges"]:
        lab = labels[js:je]
        cnt = counts[js:je]
        for f in FLAVORS:
            cell[(f, b)] = int(cnt[lab == f].sum())

    # score each flavor pair (e.g. b vs light)
    for fa, fb in PAIRS:
        w_sum = 0.0   # weighted sum of W1 over bins
        total = 0.0    # sum of the weights
        a_values = []  # values of flavor fa from every used bin
        a_weights = []
        b_values = []   # values of flavor fb from every used bin
        b_weights = []
        used = 0  # number of bins used

        for b, js, je, ns, ne in arr["ranges"]:
            ca = cell[(fa, b)]
            cb = cell[(fb, b)]

            # skip bins that do not have enough jets of both flavors
            if ca < min_jets or cb < min_jets:
                continue

            if union:
                v = arr["values"][descriptor][ns:ne]
                jl = labels[node_jet[ns:ne]]    # label of each node's jet
                jc = counts[node_jet[ns:ne]]    # count of each node's jet
            else:
                v = arr["means"][descriptor][js:je]
                jl = labels[js:je]
                jc = counts[js:je]

            # pick values of each flavor (skip jets with count 0)
            ma = (jl == fa) & (jc > 0)
            mb = (jl == fb) & (jc > 0)
            va = v[ma]
            wa = jc[ma].astype(float)
            vb = v[mb]
            wb = jc[mb].astype(float)

            # skip bins with too few values on either side
            if wa.sum() < min_values or wb.sum() < min_values:
                continue

            weight = ca + cb
            w_sum += weight * wasserstein_distance(va, vb, wa, wb)
            total += weight
            a_values.append(va)
            a_weights.append(wa)
            b_values.append(vb)
            b_weights.append(wb)
            used += 1

        # no usable bins for this pair -- skip it
        if total == 0:
            continue

        # Score inside bins (weighted average over bins)
        cond = w_sum / total

        # Score on the same jets all bins thrown together
        kept = wasserstein_distance(np.concatenate(a_values), np.concatenate(b_values), np.concatenate(a_weights), np.concatenate(b_weights))

        cond_per_pair.append(cond)
        kept_per_pair.append(kept)
        jets_kept += int(total)
        jets_total += int(counts[(labels == fa) | (labels == fb)].sum())

        if kept > 0:
            pair_preserved = cond / kept
        else:
            pair_preserved = np.nan

        out["pairs"][PAIR_LABEL[(fa, fb)]] = {
            "S_sep_cond": cond,
            "S_sep_kept": kept,
            "preserved": pair_preserved,
            "n_bins": used,
            "n_jets_kept": int(total),
        }

    if not cond_per_pair:
        out["S_sep_cond"] = np.nan
        out["S_sep_kept"] = np.nan
        out["preserved"] = np.nan
        out["n_jets_kept"] = 0
        out["n_jets_total"] = 0
        return out

    # average over the pairs then take the ratio
    mean_cond = float(np.mean(cond_per_pair))
    mean_kept = float(np.mean(kept_per_pair))
    out["S_sep_cond"] = mean_cond
    out["S_sep_kept"] = mean_kept
    if mean_kept > 0:
        out["preserved"] = mean_cond / mean_kept
    else:
        out["preserved"] = np.nan
    out["n_jets_kept"] = jets_kept
    out["n_jets_total"] = jets_total
    return out


def shuffle_within(labels, cell_id, rng):
    """
    Shuffle the labels, but only between jets that share the same cell_id.
    Jets with cell_id -1 are not touched.
    Every cell keeps the same number of each label.
    """
    # jets that are inside some cell.
    idx = np.flatnonzero(cell_id >= 0)
    c = cell_id[idx]

    # jets sorted by cell (keeping their order inside a cell)
    by_cell = idx[np.argsort(c, kind="stable")]

    # jets sorted by cell but in random order inside each cell
    # lexsort sorts by the LAST key first (the cell) then by the random numbers
    shuffled = idx[np.lexsort((rng.random(idx.size), c))]

    # move the labels since both lists go cell by cell, each label
    # only moves to another jet in the same cell
    out = labels.copy()
    out[shuffled] = labels[by_cell]
    return out


def score_construction(task):
    """
    For one construction, and for every descriptor and aggregation, compute:
    the real scores, the bootstrap interval, and both shuffle nulls.
    This runs in a separate worker process.

    :param task: (name, cond, arrays, descriptors, aggregations, n_boot, n_perm, min_jets, min_values, seed)
    :return: (rows, pair_rows)
    """
    name, cond, arr, descriptors, aggregations, n_boot, n_perm, min_jets, min_values, seed = task
    start = time.time()

    flavor = arr["flavor"]
    ones = np.ones(len(flavor), dtype=np.int64)   # every jet counted once
    rows = []
    pair_rows = []

    for descriptor in descriptors:
        for aggregation in aggregations:

            obs = scores(arr, descriptor, aggregation, flavor, ones, min_jets, min_values)

            if obs["n_jets_total"]:
                coverage = obs["n_jets_kept"] / obs["n_jets_total"]
            else:
                coverage = np.nan

            row = {
                "construction": name,
                "cond": cond,
                "descriptor": descriptor,
                "aggregation": aggregation,
                "S_sep_cond": obs["S_sep_cond"],
                "S_sep_kept": obs["S_sep_kept"],
                "preserved": obs["preserved"],
                "n_jets": len(flavor),
                "n_jets_kept": obs["n_jets_kept"],
                "n_jets_total": obs["n_jets_total"],
                "coverage": coverage,
            }

            for pair, s in obs["pairs"].items():
                pair_row = {
                    "construction": name,
                    "cond": cond,
                    "descriptor": descriptor,
                    "aggregation": aggregation,
                    "pair": pair,
                }
                pair_row.update(s)
                pair_rows.append(pair_row)

            # Bootstrap
            # random generator is made fresh here with the same seed so
            # every descriptor and aggregation of a construction gets the same random draws.
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])

            # The jet indices of each flavor.
            jets_of = []
            for f in FLAVORS:
                jets_of.append(np.flatnonzero(flavor == f))

            boot = []
            for _ in range(n_boot):
                # counts[j] = how many times jet j was picked this round
                counts = np.zeros(len(flavor), dtype=np.int64)
                for jets in jets_of:
                    picked = rng.choice(jets, len(jets), replace=True)
                    np.add.at(counts, picked, 1)   # add 1 for every pick (repeats too)
                boot_scores = scores(arr, descriptor, aggregation, flavor, counts, min_jets, min_values)
                boot.append(boot_scores["preserved"])

            if boot:
                lo_hi = np.nanquantile(boot, [0.025, 0.975])
                row["preserved_lo"] = lo_hi[0]
                row["preserved_hi"] = lo_hi[1]
            else:
                row["preserved_lo"] = np.nan
                row["preserved_hi"] = np.nan

            # Shuffle nulls ("exact" first, then "bin")
            kept_real = obs["S_sep_kept"]
            null_types = [("exact", arr["exact_cell"]), ("bin", arr["bin"])]

            for tag, cell_id in null_types:
                null = []
                for _ in range(n_perm):
                    lab = shuffle_within(flavor, cell_id, rng)
                    s = scores(arr, descriptor, aggregation, lab, ones, min_jets, min_values)["S_sep_cond"]
                    # Divide by the REAL S_sep_kept not the shuffled one
                    if kept_real > 0:
                        null.append(s / kept_real)
                    else:
                        null.append(np.nan)

                # throw away nan / inf values
                null = np.asarray(null, dtype=float)
                null = null[np.isfinite(null)]

                real = obs["preserved"]
                if null.size and np.isfinite(real):
                    # p-values: what fraction of shuffles are at least as extreme as the real value
                    # +1 on top and bottom counts the real value itself so p is never 0
                    p_above = float((1 + (null >= real).sum()) / (1 + null.size))
                    p_below = float((1 + (null <= real).sum()) / (1 + null.size))
                    p_two_sided = min(1.0, 2 * min(p_above, p_below))

                    null_mean = float(null.mean())
                    q025 = np.quantile(null, 0.025)
                    q975 = np.quantile(null, 0.975)

                    row[f"null_{tag}_mean"] = null_mean
                    row[f"null_{tag}_q025"] = float(q025)
                    row[f"null_{tag}_q975"] = float(q975)
                    row[f"null_{tag}_p_above"] = p_above
                    row[f"null_{tag}_p_below"] = p_below
                    row[f"null_{tag}_p_two_sided"] = p_two_sided
                    row[f"excess_over_{tag}_null"] = real - null_mean
                    row[f"above_{tag}_null"] = bool(real > q975)
                    row[f"below_{tag}_null"] = bool(real < q025)

            rows.append(row)

    print(f"[done] {name}: {len(rows)} rows in {time.time() - start:.0f} s", flush=True)
    return rows, pair_rows


def check_fc(table):
    """
    On the complete grap "preserved" must equal its exact null
    """
    fc = table[table["construction"] == FC_GRAPH]
    for _, r in fc.iterrows():
        null_mean = r.get("null_exact_mean", np.nan)
        gap = abs(r["preserved"] - null_mean)
        if gap <= FC_TOL:
            verdict = "ok"
        else:
            verdict = "MISMATCH: the exact null does not keep the sizes fixed"
        print(f"[check] complete graph, {r['descriptor']} {r['aggregation']}: preserved {r['preserved']:.6g}, "
              f"exact null {null_mean:.6g} ({verdict})", flush=True)


def check_share(table, share_dir, cond):
    """
    Compare our real scores with share_<cond>.csv (all nodes) from the scoring run
    """
    path = os.path.join(share_dir, f"share_{cond}.csv")
    if not os.path.exists(path):
        print(f"[check-share] {path} not found; skipped")
        return

    share = pd.read_csv(path)
    share = share[share["Comparison"] == "all_nodes"]

    ours = table.assign(Graph=table["construction"].str.lower(), Curvature=table["descriptor"].str.lower(), Aggregation=table["aggregation"].str.lower())
    share = share.assign(Graph=share["Graph"].str.lower(), Curvature=share["Curvature"].str.lower(), Aggregation=share["Aggregation"].str.lower())

    m = ours.merge(share, on=["Graph", "Curvature", "Aggregation"], suffixes=("", "_run"))
    if m.empty:
        raise SystemExit(f"[check-share] no rows in common with {path}: the names are spelled differently")

    d_cond = float((m["S_sep_cond"] - m[SEP_COL]).abs().max())
    d_kept = float((m["S_sep_kept"] - m["S_sep_kept_run"]).abs().max())
    if max(d_cond, d_kept) < 1e-9:
        verdict = "ok"
    else:
        verdict = "DIFFERENT: check the settings"
    print(f"[check-share] {len(m)} rows against {path}: max |diff| S_sep_cond {d_cond:.2e}, S_sep_kept {d_kept:.2e} ({verdict})")

def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--constructions", nargs="*", default=None, help="default: every construction")
    parser.add_argument("--descriptors", nargs="*", default=["orc"], choices=["orc", "frc", "deg", "tri", "clu"])
    parser.add_argument("--aggregations", nargs="*", default=["union"], choices=["union", "per_jet_mean"])
    parser.add_argument("--cond", default=JOINT_COND, choices=COND_CHOICES)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--min-jets", type=int, default=MIN_JETS_PER_BIN, help="jets a (flavor, bin) cell needs")
    parser.add_argument("--workers", type=int, default=8, help="scoring workers; also shard-reading workers")
    parser.add_argument("--max-jets", type=int, default=MAX_TRAIN_JETS)
    parser.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    parser.add_argument("--check-share", default=None, help="folder of experiments.jetset.run_scores, to compare the real scores")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.constructions:
        names = args.constructions
    else:
        names = GRAPH_TYPES

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"construction_null_{args.cond}.csv")
    pairs_csv = os.path.join(args.out_dir, f"construction_null_{args.cond}_pairs.csv")

    # load results from an earlier run if there are any.
    if os.path.exists(out_csv):
        done = pd.read_csv(out_csv)
    else:
        done = pd.DataFrame(columns=["construction"])

    # every (descriptor, aggregation) combination we were asked for
    want = set()
    for d in args.descriptors:
        for a in args.aggregations:
            want.add((d, a))

    # which combinations each construction already has in the file
    have = {}
    old_graphs = done.get("construction", [])
    old_descriptors = done.get("descriptor", [])
    old_aggregations = done.get("aggregation", [])
    for g, d, a in zip(old_graphs, old_descriptors, old_aggregations):
        if g not in have:
            have[g] = set()
        have[g].add((d, a))

    # a construction is done only if it has every combination we want
    todo = []
    for g in names:
        already = have.get(g, set())
        if not want.issubset(already):
            todo.append(g)

    print(f"{len(names)} constructions requested, {len(todo)} to score; cond {args.cond}, descriptors "
          f"{args.descriptors}, aggregations {args.aggregations}, boot {args.n_boot}, perm {args.n_perm}", flush=True)

    rows = done.to_dict("records")
    if os.path.exists(pairs_csv):
        pair_rows = pd.read_csv(pairs_csv).to_dict("records")
    else:
        pair_rows = []

    if todo:
        kept_ids = get_kept_ids("train", args.max_jets, args.seed)
        reference = None   # the jets of the first construction, to compare the others with

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []

            # load each construction here then send it to a worker to score
            for name in todo:
                load_start = time.time()
                nodes = load_construction(name, kept_ids, args.descriptors, args.workers)
                arr = jet_arrays(nodes, args.descriptors, args.cond)
                del nodes   # free memory, we only need arr now

                # Every construction must have the same jets, sizes and flavors.
                key = (arr["jet_id"].tobytes(), arr["n"].tobytes(),
                       arr["n_sv"].tobytes(), arr["flavor"].tobytes())
                if reference is None:
                    reference = key
                elif key != reference:
                    raise SystemExit(f"{name} has a different jet set, sizes or flavors than the first construction")

                print(f"[load] {name}: {len(arr['flavor'])} jets in {time.time() - load_start:.0f} s", flush=True)

                task = (name, args.cond, arr, args.descriptors, args.aggregations, args.n_boot, args.n_perm,
                        args.min_jets, MIN_VALUES_PER_SIDE, args.seed)
                futures.append(pool.submit(score_construction, task))

            for fut in futures:
                new_rows, new_pairs = fut.result()
                if new_rows:
                    name = new_rows[0]["construction"]
                else:
                    name = None

                # Remove this construction's old rows for the combinations we
                # just redid so they are not in the file twice.
                kept_rows = []
                for r in rows:
                    combo = (r.get("descriptor"), r.get("aggregation"))
                    if not (r["construction"] == name and combo in want):
                        kept_rows.append(r)
                rows = kept_rows

                kept_pair_rows = []
                for r in pair_rows:
                    combo = (r.get("descriptor"), r.get("aggregation"))
                    if not (r["construction"] == name and combo in want):
                        kept_pair_rows.append(r)
                pair_rows = kept_pair_rows

                rows.extend(new_rows)
                pair_rows.extend(new_pairs)

                pd.DataFrame(rows).to_csv(out_csv, index=False)
                pd.DataFrame(pair_rows).to_csv(pairs_csv, index=False)

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("nothing scored")

    # how much more each construction keeps than the complete graph does
    fc_rows = table[table["construction"] == FC_GRAPH]
    fc = fc_rows.set_index(["descriptor", "aggregation"])["preserved"]
    excess = []
    for _, r in table.iterrows():
        fc_value = fc.get((r["descriptor"], r["aggregation"]), np.nan)
        excess.append(r["preserved"] - fc_value)
    table["excess_over_complete"] = excess

    table.to_csv(out_csv, index=False)

    check_fc(table)
    if args.check_share:
        check_share(table, args.check_share, args.cond)

    wanted_cols = ["construction", "descriptor", "aggregation", "preserved", "preserved_lo", "preserved_hi",
                   "null_exact_mean", "null_exact_q975", "null_exact_p_above", "null_exact_p_below",
                   "above_exact_null", "below_exact_null", "coverage"]
    cols = []
    for c in wanted_cols:
        if c in table.columns:
            cols.append(c)

    with pd.option_context("display.width", 260, "display.float_format", "{:.4f}".format):
        print(table[cols].to_string(index=False))
    print(f"wrote {out_csv} and {pairs_csv}", flush=True)


if __name__ == "__main__":
    main()

