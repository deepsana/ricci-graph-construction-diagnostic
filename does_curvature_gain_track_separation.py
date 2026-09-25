import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import spearmanr

DR_CSV = "diminishing_returns-2026-09-07_default_300k_50k.csv"
SEP_DIR = "separation/results-09-20-separation"
CONDS = ["n", "n_x_n_sv"]
COMPARISON = "all_nodes"
CURV = "orc"
DEEP_MODELS = {"transformer", "GNN"}
MODELS = ["transformer", "GNN"]


plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})
TAB10 = plt.get_cmap("tab10")
FAMILY_COLOR = {"knn": TAB10(0), "radius": TAB10(2), "kt_threshold": TAB10(1),"laman": TAB10(3), "unique_k": TAB10(4), "fully_connected": TAB10(7)}
P_MARKER = {-1: "v", 0: "s", 1: "^"}
P_TEXT = {"neg1": -1, "zero": 0, "pos1": 1}
PAIR_FILL = {"b_vs_light": "full", "c_vs_light": "none", "tau_vs_light": "bottom"}
PAIR_TEXT = {"b_vs_light": r"$b$ vs light", "c_vs_light": r"$c$ vs light", "tau_vs_light": r"$\tau$ vs light"}
FAMILY_ORDER = ["knn", "radius", "kt_threshold", "laman", "unique_k", "fully_connected"]


plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})
TAB10 = plt.get_cmap("tab10")
FAMILY_COLOR = {"knn": TAB10(0), "radius": TAB10(2), "kt_threshold": TAB10(1),
                "laman": TAB10(3), "unique_k": TAB10(4), "fully_connected": TAB10(7)}
P_MARKER = {-1: "v", 0: "s", 1: "^"}
P_TEXT = {"neg1": -1, "zero": 0, "pos1": 1}
PAIR_FILL = {"b_vs_light": "full", "c_vs_light": "none", "tau_vs_light": "bottom"}
PAIR_TEXT = {"b_vs_light": r"$b$ vs light", "c_vs_light": r"$c$ vs light",
             "tau_vs_light": r"$\tau$ vs light"}
FAMILY_ORDER = ["knn", "radius", "kt_threshold", "laman", "unique_k", "fully_connected"]


def parse(name):
    n = name.lower()
    if n == "fully_connected":
        return "fully_connected", "o", "Fully connected"
    if m := re.fullmatch(r"knn_(\d+)", n):
        return "knn", "o", rf"$k$NN ($k={m.group(1)}$)"
    if re.fullmatch(r"radius_based(?:_r\dp\d+)?", n):
        return "radius", "D", r"$\epsilon$-based ($\epsilon=0.3$)"
    if m := re.search(r"kt_threshold.*_p_(neg1|pos1|zero)", n):
        p = P_TEXT[m.group(1)]
        return "kt_threshold", P_MARKER[p], rf"$k_T$ thr. ($q=0.5,\,p={p:+d}$)"
    if m := re.fullmatch(r"laman_p_(neg1|pos1|zero)", n):
        p = P_TEXT[m.group(1)]
        return "laman", P_MARKER[p], rf"Laman ($p={p:+d}$)"
    if m := re.fullmatch(r"unique_(\d+)_p_(neg1|pos1|zero)", n):
        p = P_TEXT[m.group(2)]
        return "unique_k", P_MARKER[p], rf"Unique-{m.group(1)} ($p={p:+d}$)"
    return "other", "x", name


#  baseline and gain per (target, construction, model), over seeds
dr = pd.read_csv(DR_CSV)
piv = dr.pivot_table(index=["dataset", "graph_type", "model", "model_seed"],
                     columns="setting", values="score").reset_index()
piv["gain"] = piv["+ curvature"] - piv["baseline"]
per = (piv.groupby(["dataset", "graph_type", "model"])
          .agg(baseline=("baseline", "mean"), baseline_std=("baseline", "std"),
               gain=("gain", "mean"), gain_std=("gain", "std"), n_seeds=("gain", "count"))
          .reset_index())
per["Graph"] = per["graph_type"].str.upper()
per["FlavorPair"] = per["dataset"].str.replace("jets_", "", regex=False)
per["Aggregation"] = np.where(per["model"].isin(DEEP_MODELS), "union", "per_jet_mean")
per = per[per["Graph"] != "FULLY_CONNECTED"]   # FC is the control, not a candidate

#conditioned separation per (target, construction): cond x curvature
CURVS = ["orc", "frc"]
df = per
for cond in CONDS:
    s = pd.read_csv(f"{SEP_DIR}/scores_conditioned_{cond}_per_pair_matched.csv")
    s = s[(s["Comparison"] == COMPARISON) & s["Curvature"].isin(CURVS)]
    w = s.pivot_table(index=["Graph", "FlavorPair", "Aggregation"], columns="Curvature", values="S_sep_cond").reset_index()
    w = w.rename(columns={c: f"sep_{cond}_{c}" for c in CURVS})
    w[f"sep_{cond}_max"] = w[[f"sep_{cond}_{c}" for c in CURVS]].max(axis=1)
    df = df.merge(w, on=["Graph", "FlavorPair", "Aggregation"], how="inner")
XS = [f"sep_{cond}_{c}" for cond in CONDS for c in CURVS + ["max"]]

# within pair correlations: per pair, and pooled on within-pair ranks
# Pooled Spearman across pairs measures task difficulty (harder pair -> lower separation and lower AUROC)
rows = []
for model in MODELS:
    d = df[df["model"] == model].copy()
    ys = ("gain",) if model == "transformer" else ("baseline", "gain")
    for x in XS:
        for y in ys:
            for pair, g in d.groupby("FlavorPair"):
                rho, p = spearmanr(g[x], g[y])
                rows.append({"model": model, "x": x, "y": y, "scope": pair, "spearman": rho, "p": p, "n_points": len(g)})
            rx = d.groupby("FlavorPair")[x].rank()
            ry = d.groupby("FlavorPair")[y].rank()
            rho, p = spearmanr(rx, ry)
            rows.append({"model": model, "x": x, "y": y, "scope": "within-pair pooled", "spearman": rho, "p": p, "n_points": len(d)})
summary = pd.DataFrame(rows)
pooled = summary[summary["scope"] == "within-pair pooled"]
print(pooled.pivot_table(index=["model", "y"], columns="x", values="spearman").to_string(float_format=lambda v: f"{v:+.2f}"))
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
summary.to_csv("rcurvature_gain_tracking_conditioned_summary.csv", index=False)

# epsilon against the rest, within pair, for each curvature at fixed n
eps = df["graph_type"].str.lower().str.startswith("radius")
for model in MODELS:
    d = df[df["model"] == model]
    for pair, g in d.groupby("FlavorPair"):
        e, o = g[eps[g.index]], g[~eps[g.index]]
        line = (f"{model:12s} {pair:13s} eps gain {e['gain'].iloc[0]:+.4f} ± {e['gain_std'].iloc[0]:.4f}"
                f"   others {o['gain'].min():+.4f} to {o['gain'].max():+.4f}")
        for c in CURVS:
            x = f"sep_n_{c}"
            line += f"   | {c} sep eps {e[x].iloc[0]:.4f}, others {o[x].min():.4f}-{o[x].max():.4f}"
        print(line)



COND_LABEL = {"n": r"fixed $n$", "n_x_n_sv": r"fixed $(n,\,n_{\mathrm{SV}})$"}
CURV_LABEL = {"orc": "ORC", "frc": "FRC", "max": "max(ORC, FRC)"}


def x_label(x):
    cond, curv = x[len("sep_"):].rsplit("_", 1)
    return f"{CURV_LABEL[curv]} separation at {COND_LABEL[cond]}"


def pooled_rho(model, x, y):
    q = pooled[(pooled["model"] == model) & (pooled["x"] == x) & (pooled["y"] == y)]
    return q["spearman"].iloc[0] if len(q) else None


COND =  "n_x_n_sv" #"n" #"n_x_n_sv"
assert COND in CONDS, f"{COND} not in CONDS={CONDS}"
ROW_CURVS = ["orc", "frc"]
PANELS = [("transformer", "gain"), ("GNN", "baseline"), ("GNN", "gain")]
COL_TEXT = {("transformer", "gain"): "Transformer: AUROC gain",
            ("GNN", "baseline"): "GNN: AUROC baseline",
            ("GNN", "gain"): "GNN: AUROC gain"}

graphs = sorted(df["graph_type"].unique(), key=lambda g: (FAMILY_ORDER.index(parse(g)[0]), g))
graph_handles = [Line2D([], [], marker=parse(g)[1], color=FAMILY_COLOR[parse(g)[0]], ls="none", ms=6, label=parse(g)[2]) for g in graphs]
pair_handles = [Line2D([], [], marker="o", color="0.3", ls="none", ms=6, fillstyle=PAIR_FILL[p], markerfacecoloralt="white", label=PAIR_TEXT[p]) for p in PAIR_FILL]
leg1_rows = (len(graphs) + 2) // 3

def within_pair_rho(d, x, y):
    rx = d.groupby("FlavorPair")[x].rank()
    ry = d.groupby("FlavorPair")[y].rank()
    return spearmanr(rx, ry)[0]

def perm_p(d, x, y, n=5000, seed=0):
    rng, obs, d = np.random.default_rng(seed), within_pair_rho(d, x, y), d.copy()
    groups = [g.index for _, g in d.groupby("FlavorPair")]
    vals, hits = d[y].to_numpy().copy(), 0
    for _ in range(n):
        for idx in groups:
            pos = d.index.get_indexer(idx)
            vals[pos] = rng.permutation(vals[pos])
        hits += abs(spearmanr(d.groupby("FlavorPair")[x].rank(), pd.Series(vals, index=d.index).groupby(d["FlavorPair"]).rank())[0]) >= abs(obs)
    return obs, (hits + 1) / (n + 1)

eps_mask = df["graph_type"].str.lower().str.startswith("radius")
for label, sub in [("all", df), ("no eps", df[~eps_mask])]:
    for model, y in [("transformer", "gain"), ("GNN", "baseline"), ("GNN", "gain")]:
        for cond in CONDS:
            for c in CURVS:
                rho, p = perm_p(sub[sub["model"] == model], f"sep_{cond}_{c}", y)
                print(f"{label:6s} {model:11s} {y:8s} {cond:9s} {c}  rho={rho:+.2f}  p_perm={p:.3f}")

def plot_tracking(cond):
    fig, axes = plt.subplots(len(ROW_CURVS), len(PANELS), figsize=(7.0, 4.8),
                             sharex="row", sharey="col", squeeze=False)
    for i, curv in enumerate(ROW_CURVS):
        x = f"sep_{cond}_{curv}"
        for j, (model, y) in enumerate(PANELS):
            ax = axes[i][j]
            d = df[df["model"] == model]
            for _, r in d.iterrows():
                fam, mk, _ = parse(r["graph_type"])
                eb = ax.errorbar(r[x], r[y], yerr=r[f"{y}_std"], fmt=mk, ms=5, lw=0.8,
                                 color=FAMILY_COLOR[fam], fillstyle=PAIR_FILL[r["FlavorPair"]],
                                 markeredgewidth=1.1, capsize=0)
                eb[0].set_markerfacecoloralt("white")
            if curv == "frc":
                if (d[x] <= 0).any():
                    raise ValueError(f"non-positive {x}; log axis impossible")
                ax.set_xscale("log")
            if y == "gain":
                ax.axhline(0, color="0.5", lw=0.6)

            rho = pooled_rho(model, x, y)
            tag = rf"$\rho_s={rho:+.2f}$" if rho is not None else ""
            head = f"{COL_TEXT[(model, y)]}\n" if i == 0 else ""
            ax.set_title(head + tag, fontsize=9, loc="left")
            if j == 1:
                ax.set_xlabel(x_label(x), fontsize=9)
            ax.tick_params(labelsize=8)

    leg1 = fig.legend(handles=graph_handles, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    leg2 = fig.legend(handles=pair_handles, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01 - 0.045 * leg1_rows))
    fig.tight_layout()
    fig.savefig(f"sep_vs_auroc_{cond}.pdf", bbox_inches="tight",
                bbox_extra_artists=(leg1, leg2))
    plt.close(fig)


plot_tracking(COND)
plot_tracking("n")
