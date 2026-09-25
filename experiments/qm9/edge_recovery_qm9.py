import os
import time
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem

from src.qm9_graphs import ALL_GRAPH_TYPES, ATOM_MASS, ATOM_Z, DROP_HYDROGENS, GRAPH_CONFIGS


from utils.local_paths import local_paths
PATHS = local_paths()


# this is where the data is 
INPUT_DIR = PATHS.QM9_DOWNLAODED_DATA
OUT_DIR = PATHS.QM9_SHARD_GLOB
SDF_PATH = os.path.join(PATHS.QM9_DOWNLAODED_DATA, "qm9", "gdb9.sdf")
VERIFIED_IDS_PATH = os.path.join(PATHS.QM9_DOWNLAODED_DATA, "qm9", "verified_mol_ids.txt")

RESULTS_DIR = PATHS.RESULTS_ROOT
OUT_PATH = os.path.join(RESULTS_DIR, "edge_recovery.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "edge_recovery_summary.csv")
FALSE_PAIRS_PATH = os.path.join(RESULTS_DIR, "edge_recovery_false_pairs.csv")
HFRAC_PATH = os.path.join(RESULTS_DIR, "edge_recovery_knn_hfrac.csv")
PR_FIGURE_PATH = os.path.join(RESULTS_DIR, "edge_recovery_pr.png")
HFRAC_FIGURE_PATH = os.path.join(RESULTS_DIR, "edge_recovery_knn_hfrac.png")

PAPER_GRAPH_TYPES = ["knn_3", "radius_bond", "fully_connected"]
SWEEP_GRAPH_TYPES = [
    "knn_2", "knn_3", "knn_4", "knn_5",
    "radius_1p5", "radius_bond", "radius_2p5", "radius_3A", "radius_4A",
    "unique_2_p_neg1", "unique_3_p_neg1", "unique_4_p_neg1",
    "unique_3_p_zero", "unique_4_p_zero",
    "fully_connected",
]
RADIUS_GRAPH_TYPE = "radius_bond"
HFRAC_GRAPH_TYPE = "knn_3"
H_FRAC_BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

MIN_ATOMS = 3
MAX_MOLS = None
CEILING_TOL = 1e-6


def load_verified_ids(path):
    if not os.path.exists(path):
        raise SystemExit(f"Error: Verified ID list not found at {path}.")
    with open(path) as f:
        ids = {int(line.strip()) for line in f if line.strip()}
    if not ids:
        raise SystemExit(f"Error: {path} is empty.")
    return ids


def sdf_entry_to_mol_dict(rd_mol):
    if rd_mol is None:
        return None

    conf = rd_mol.GetConformer()
    elements, coords, masses, atomic_numbers = [], [], [], []

    for atom in rd_mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ATOM_MASS:
            return None
        pos = conf.GetAtomPosition(atom.GetIdx())
        elements.append(sym)
        coords.append([pos.x, pos.y, pos.z])
        masses.append(ATOM_MASS[sym])
        atomic_numbers.append(ATOM_Z[sym])

    true_edges = {
        (min(b := bnd.GetBeginAtomIdx(), e := bnd.GetEndAtomIdx()), max(b, e))
        for bnd in rd_mol.GetBonds()
    }
    mol_id = int(rd_mol.GetProp("_Name").split("_")[1])

    mol = {
        "mol_id": mol_id,
        "num_atoms": len(elements),
        "smiles": "",
        "atoms": {
            "coords": np.array(coords),
            "element": elements,
            "Z": np.array(atomic_numbers, dtype=np.int32),
            "mass": np.array(masses),
            "mulliken": np.zeros(len(elements)),
        },
        "targets": {},
    }
    return mol, true_edges


def to_undirected_set(edges):
    return {(min(s := int(src), d := int(dst)), max(s, d)) for src, dst in edges}


def score_edges(edge_set, true_edges):
    n_hits = len(edge_set & true_edges)
    if not edge_set or not true_edges:
        return n_hits, np.nan, np.nan, np.nan

    precision = n_hits / len(edge_set)
    recall = n_hits / len(true_edges)
    if precision + recall == 0:
        return n_hits, precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return n_hits, precision, recall, f1


def edge_stats(edge_set, true_edges, elements, num_atoms, false_pairs):
    stats = {"n_edges_H": 0, "hits_H": 0, "n_edges_heavy": 0, "hits_heavy": 0}
    degree = np.zeros(num_atoms, dtype=int)

    for i, j in edge_set:
        degree[i] += 1
        degree[j] += 1
        has_h = elements[i] == "H" or elements[j] == "H"

        if has_h:
            stats["n_edges_H"] += 1
        else:
            stats["n_edges_heavy"] += 1

        if (i, j) in true_edges:
            if has_h:
                stats["hits_H"] += 1
            else:
                stats["hits_heavy"] += 1
        else:
            pair = f"{elements[i]}-{elements[j]}" if elements[i] <= elements[j] else f"{elements[j]}-{elements[i]}"
            false_pairs[pair] += 1

    stats["min_degree"] = int(degree.min())
    return stats


def knn_degree_ceiling(graph_type, params, num_atoms, n_true):
    if not graph_type.startswith("knn") or "k" not in params:
        return np.nan
    k_eff = min(params["k"], num_atoms - 1)
    if k_eff <= 0:
        return np.nan
    return min(1.0, 2 * n_true / (num_atoms * k_eff))


def family_of(graph_type):
    for fam in ["knn", "radius", "kt_threshold", "laman", "unique", "fully_connected"]:
        if graph_type.startswith(fam):
            return fam
    return "other"


def plot_precision_recall(summary, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    graphs = sorted(summary.index.tolist())
    cmap = plt.get_cmap("tab20")
    colors = {g: cmap(i % 20) for i, g in enumerate(graphs)}
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p"]
    marker_map = {g: markers[i % len(markers)] for i, g in enumerate(graphs)}

    curves = [("knn", "k"), ("radius", "radius")]
    for family, param_name in curves:
        points = []
        for g_type in summary.index:
            if family_of(g_type) == family:
                val = GRAPH_CONFIGS[g_type][1][param_name]
                points.append((val, summary.loc[g_type, "recall"], summary.loc[g_type, "precision"], g_type))
        points.sort()
        if not points:
            continue

        recalls = [p[1] for p in points]
        precisions = [p[2] for p in points]
        line_color = colors[points[0][3]]

        ax.plot(recalls, precisions, color=line_color, alpha=0.6, linewidth=1.5, zorder=1)
        for val, rec, prec, g in points:
            ax.scatter(rec, prec, facecolors=colors[g], edgecolors="black",
                       marker=marker_map[g], s=70, linewidths=1.1, zorder=3)
            ax.annotate(f"{val:g}", (rec, prec), textcoords="offset points", xytext=(4, 3), fontsize=7)

    for g_type in summary.index:
        if family_of(g_type) not in ["knn", "radius"]:
            rec = summary.loc[g_type, "recall"]
            prec = summary.loc[g_type, "precision"]
            ax.scatter(rec, prec, facecolors=colors[g_type], edgecolors="black",
                       marker=marker_map[g_type], s=70, linewidths=1.1, zorder=3)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Edge Recovery Precision vs. Recall", fontsize=11)

    handles = [Line2D([0], [0], marker=marker_map[g], color="w", markerfacecolor=colors[g],
                      markeredgecolor="black", markersize=8, label=g) for g in graphs]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=7, frameon=True, title="Graph Construction")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_knn_hfrac(hfrac_table, path):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    positions = np.arange(len(hfrac_table))
    tick_labels = [f"{max(iv.left, 0.0):.1f}–{iv.right:.1f}" for iv in hfrac_table.index]

    cmap = plt.get_cmap("tab20")
    ax.plot(positions, hfrac_table["precision"], marker="o", color=cmap(0), linewidth=1.5, markersize=6,
            label="Precision")
    ax.plot(positions, hfrac_table["degree_ceiling"], marker="s", color=cmap(2), linestyle="--", linewidth=1.5,
            markersize=6, label="Degree-floor bound")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Hydrogen fraction $n^H / n$", fontsize=9)
    ax.set_ylabel(f"Precision ({HFRAC_GRAPH_TYPE})", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=True)
    ax.set_title(f"Precision by Hydrogen Fraction ({HFRAC_GRAPH_TYPE})", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if DROP_HYDROGENS:
        raise SystemExit("Error: DROP_HYDROGENS must be False in src/qm9_graphs.py for this experiment.")

    graph_types = []
    skipped = []
    for name in PAPER_GRAPH_TYPES + list(ALL_GRAPH_TYPES) + SWEEP_GRAPH_TYPES:
        if name in graph_types or name in skipped:
            continue
        if name in GRAPH_CONFIGS:
            graph_types.append(name)
        else:
            skipped.append(name)

    for name in PAPER_GRAPH_TYPES:
        if name in skipped:
            raise SystemExit(f"Error: Required paper graph '{name}' missing from GRAPH_CONFIGS.")

    verified = load_verified_ids(VERIFIED_IDS_PATH)
    false_pairs = {gt: Counter() for gt in graph_types}
    identical_pairs = [(graph_types[i], graph_types[j]) for i in range(len(graph_types)) for j in
                       range(i + 1, len(graph_types))]
    hh_distances = []

    supplier = Chem.SDMolSupplier(SDF_PATH, removeHs=False, sanitize=False)
    rows = []
    n_scored, n_skipped = 0, 0
    start_time = time.time()

    for rd_mol in supplier:
        parsed = sdf_entry_to_mol_dict(rd_mol)
        if parsed is None:
            n_skipped += 1
            continue
        mol, true_edges = parsed
        if mol["mol_id"] not in verified or mol["num_atoms"] < MIN_ATOMS:
            continue

        elements = mol["atoms"]["element"]
        coords = mol["atoms"]["coords"]
        num_atoms = mol["num_atoms"]
        n_true = len(true_edges)
        n_true_h = sum(1 for i, j in true_edges if elements[i] == "H" or elements[j] == "H")

        edge_sets = {}
        for gt in graph_types:
            build_graph, params = GRAPH_CONFIGS[gt]
            edges = build_graph(mol, **params)
            edge_set = to_undirected_set(edges)
            edge_sets[gt] = edge_set

            n_hits, precision, recall, f1 = score_edges(edge_set, true_edges)
            count_ceiling = min(1.0, n_true / len(edge_set)) if edge_set else np.nan

            if gt == RADIUS_GRAPH_TYPE:
                for i, j in edge_set:
                    if elements[i] == "H" and elements[j] == "H":
                        hh_distances.append(float(np.linalg.norm(coords[i] - coords[j])))

            row = {
                "mol_id": mol["mol_id"],
                "num_atoms": num_atoms,
                "n_H": elements.count("H"),
                "graph_type": gt,
                "n_edges": len(edge_set),
                "n_edges_raw": len(edges),
                "n_true": n_true,
                "n_true_H": n_true_h,
                "n_hits": n_hits,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "count_ceiling": count_ceiling,
                "degree_ceiling": knn_degree_ceiling(gt, params, num_atoms, n_true),
            }
            row.update(edge_stats(edge_set, true_edges, elements, num_atoms, false_pairs[gt]))
            rows.append(row)

        identical_pairs = [(f, s) for f, s in identical_pairs if edge_sets[f] == edge_sets[s]]
        n_scored += 1

        if MAX_MOLS and n_scored >= MAX_MOLS:
            break

    if not rows:
        raise SystemExit("Error: No molecules were successfully scored.")

    results = pd.DataFrame(rows)
    grouped = results.groupby("graph_type", sort=False)
    totals = grouped[
        ["n_hits", "n_edges", "n_true", "n_true_H", "n_edges_H", "hits_H", "n_edges_heavy", "hits_heavy"]].sum()
    n_true_heavy = totals["n_true"] - totals["n_true_H"]

    summary = pd.DataFrame({
        "precision": grouped["precision"].mean(),
        "recall": grouped["recall"].mean(),
        "f1": grouped["f1"].mean(),
        "count_ceiling": grouped["count_ceiling"].mean(),
        "degree_ceiling": grouped["degree_ceiling"].mean(),
        "min_degree": grouped["min_degree"].min(),
        "precision_pooled": totals["n_hits"] / totals["n_edges"],
        "recall_pooled": totals["n_hits"] / totals["n_true"],
        "prec_H_endpoint": totals["hits_H"] / totals["n_edges_H"],
        "prec_heavy_heavy": totals["hits_heavy"] / totals["n_edges_heavy"],
        "recall_H_bonds": totals["hits_H"] / totals["n_true_H"],
        "recall_heavy_bonds": totals["hits_heavy"] / n_true_heavy,
        "frac_edges_H": totals["n_edges_H"] / totals["n_edges"],
        "n_mols": grouped["mol_id"].nunique(),
    }).reindex(graph_types)
    summary.index.name = "graph_type"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(OUT_PATH, index=False, float_format="%.6f")
    summary.round(6).to_csv(SUMMARY_PATH)

    false_rows = [
        {"graph_type": gt, "pair": p, "count": c, "fraction": c / sum(false_pairs[gt].values())}
        for gt in graph_types for p, c in false_pairs[gt].most_common()
    ]
    pd.DataFrame(false_rows).to_csv(FALSE_PAIRS_PATH, index=False, float_format="%.6f")

    knn = results[results["graph_type"] == HFRAC_GRAPH_TYPE].copy()
    knn["h_frac"] = knn["n_H"] / knn["num_atoms"]
    knn["n_rings"] = knn["n_true"] - knn["num_atoms"] + 1
    knn["h_bin"] = pd.cut(knn["h_frac"], H_FRAC_BINS, include_lowest=True)

    hfrac_table = pd.DataFrame({
        "precision": knn.groupby("h_bin", observed=True)["precision"].mean(),
        "degree_ceiling": knn.groupby("h_bin", observed=True)["degree_ceiling"].mean(),
        "mean_rings": knn.groupby("h_bin", observed=True)["n_rings"].mean(),
        "n_mols": knn.groupby("h_bin", observed=True).size(),
    })
    hfrac_table.round(6).to_csv(HFRAC_PATH)

    plot_precision_recall(summary, PR_FIGURE_PATH)
    plot_knn_hfrac(hfrac_table, HFRAC_FIGURE_PATH)
    print(f"Finished processing. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
