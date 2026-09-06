import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

_CACHE_KEY = "_true_labels"

_SDF_BONDS = None
_SDF_PATH_LOADED = None

TRUTH_COLUMNS = ["in_ring", "true_degree", "heavy_degree", "is_hydrogen",
                 "is_true_bond", "labels_ok", "smiles_match",
                 "n_true_bonds", "n_tp_bonds", "bond_precision",
                 "bond_recall", "bond_f1"]


def connectivity_smiles(mol):
    '''
    Canonical SMILES describing CONNECTIVITY ONLY.

    :param mol: an RDKit mol (or None)
    :return: canonical SMILES string, or None
    '''
    if mol is None:
        return None
    try:
        mol = Chem.RWMol(mol)

        h_indices = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                h_indices.append(atom.GetIdx())
        for idx in sorted(h_indices, reverse=True):
            mol.RemoveAtom(idx)

        for atom in mol.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.SetIsAromatic(False)

        for bond in mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)

        return Chem.MolToSmiles(mol.GetMol(), isomericSmiles=False, canonical=True)
    except Exception:
        return None


def load_sdf_bonds(sdf_path):
    '''
    Read gdb9.sdf once and build the truth table.

    :param sdf_path: path to gdb9.sdf
    :return: the truth table dict
    '''
    global _SDF_BONDS, _SDF_PATH_LOADED
    if _SDF_BONDS is not None and _SDF_PATH_LOADED == sdf_path:
        return _SDF_BONDS

    print(f"Loading true bonds from {sdf_path} ...", flush=True)
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)

    table = {}
    unreadable = 0
    for record_index, rdmol in enumerate(supplier):
        mol_id = record_index + 1
        if rdmol is None:
            unreadable += 1
            continue

        if rdmol.HasProp("_Name"):
            name = rdmol.GetProp("_Name").strip()
            if name.startswith("gdb_") and int(name.split("_")[-1]) != mol_id:
                raise SystemExit(f"FATAL: SDF record {mol_id} is named '{name}' — file order mismatch.")

        element_list = []
        for a in rdmol.GetAtoms():
            element_list.append(a.GetSymbol())
        elements = tuple(element_list)

        bond_list = []
        for b in rdmol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            if i > j:
                i, j = j, i
            bond_list.append((i, j))
        bond_list.sort()
        bonds = tuple(bond_list)

        table[mol_id] = (elements, bonds)

    print(f"Loaded bonds for {len(table)} molecules ({unreadable} unreadable records)", flush=True)
    _SDF_BONDS = table
    _SDF_PATH_LOADED = sdf_path
    return table


def _rwmol_from_bonds(elements, bonds):
    '''
    Build a minimal RDKit mol from elements and bond pairs.

    :param elements: element symbols in atom order
    :param bonds: (i, j) index pairs
    :return: an RDKit RWMol
    '''
    m = Chem.RWMol()
    for e in elements:
        m.AddAtom(Chem.Atom(e))
    for i, j in bonds:
        m.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    return m


def _compute_smiles_match(elements, bonds, mol):
    '''
    Per-molecule receipt comparing SDF connectivity to SMILES.

    :param elements: element symbols from SDF
    :param bonds: bond pairs from SDF
    :param mol: the mol dict
    :return: True, False, or None
    '''
    try:
        ours = connectivity_smiles(_rwmol_from_bonds(elements, bonds))
        if ours is None:
            return None
        relaxed = mol.get("smiles_relaxed", mol["smiles"])
        rel_conn = connectivity_smiles(Chem.MolFromSmiles(relaxed, sanitize=False))
        gdb_conn = connectivity_smiles(Chem.MolFromSmiles(mol["smiles"], sanitize=False))
        if rel_conn is None and gdb_conn is None:
            return None
        return (ours == rel_conn) or (ours == gdb_conn)
    except Exception:
        return None


def assign_true_bonds(mol, charge=0):
    '''
    Look up molecule bonds in the preloaded SDF table and derive strata.

    :param mol: the mol dict
    :param charge: accepted for signature compatibility, unused
    :return: dict with ok, bonds, in_ring, true_degree, heavy_degree, smiles_match, error
    '''
    n = mol["num_atoms"]
    Z = np.asarray(mol["atoms"]["Z"])
    out = {"ok": False,
           "bonds": [],
           "in_ring": np.zeros(n, dtype=bool),
           "true_degree": np.zeros(n, dtype=np.int32),
           "heavy_degree": np.full(n, -1, dtype=np.int32),
           "smiles_match": None,
           "error": None}

    if _SDF_BONDS is None:
        raise RuntimeError("SDF bond table not loaded. Call load_sdf_bonds first.")

    if (Z == 1).sum() == 0 and n > 1:
        out["error"] = "no hydrogens present so true-bond labels require them."
        return out

    entry = _SDF_BONDS.get(mol["mol_id"])
    if entry is None:
        out["error"] = f"mol_id {mol['mol_id']} not found in gdb9.sdf"
        return out
    elements_sdf, bonds = entry

    elements_xyz = tuple(mol["atoms"]["element"])
    if elements_sdf != elements_xyz:
        out["error"] = "element sequence mismatch between gdb9.sdf and xyz file"
        return out

    bond_pairs = [tuple(b) for b in bonds]
    out["bonds"] = bond_pairs
    out["ok"] = True

    Gchem = nx.Graph()
    Gchem.add_nodes_from(range(n))
    Gchem.add_edges_from(out["bonds"])

    for comp in nx.biconnected_components(Gchem):
        if len(comp) >= 3:
            for i in comp:
                out["in_ring"][i] = True

    out["true_degree"] = np.array([Gchem.degree(i) for i in range(n)], dtype=np.int32)

    for i in range(n):
        if Z[i] > 1:
            n_heavy = sum(1 for j in Gchem.neighbors(i) if Z[j] > 1)
            out["heavy_degree"][i] = n_heavy

    out["smiles_match"] = _compute_smiles_match(elements_sdf, bonds, mol)
    return out


def get_true_labels(mol, charge=0):
    '''
    Memoized accessor for true labels.

    :param mol: the mol dict
    :param charge: passed through to assign_true_bonds
    :return: the label dict
    '''
    if _CACHE_KEY not in mol:
        mol[_CACHE_KEY] = assign_true_bonds(mol, charge=charge)
    return mol[_CACHE_KEY]


def attach_true_labels(G, labels):
    '''
    Write serialization-friendly scalars onto a constructed graph.

    :param G: the constructed graph
    :param labels: the dict from get_true_labels
    :return: the same graph, modified in place
    '''
    T = {tuple(b) for b in labels["bonds"]}

    n_edges = G.number_of_edges()
    tp = 0
    for e in G.edges:
        flag = tuple(sorted(e)) in T
        G.edges[e]["is_true_bond"] = bool(flag)
        tp += int(flag)

    precision = tp / n_edges if n_edges else 0.0
    recall = tp / len(T) if T else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    sm = labels["smiles_match"]
    G.graph["labels_ok"] = int(labels["ok"])
    G.graph["smiles_match"] = -1 if sm is None else int(sm)
    G.graph["n_true_bonds"] = len(T)
    G.graph["n_tp_bonds"] = tp
    G.graph["bond_precision"] = precision
    G.graph["bond_recall"] = recall
    G.graph["bond_f1"] = f1

    for i in G.nodes:
        G.nodes[i]["in_ring"] = bool(labels["in_ring"][i])
        G.nodes[i]["true_degree"] = int(labels["true_degree"][i])
        G.nodes[i]["heavy_degree"] = int(labels["heavy_degree"][i])
        G.nodes[i]["is_hydrogen"] = (G.nodes[i]["Z"] == 1)
    return G


def edge_recovery_scores(constructed_edges, true_bonds):
    '''
     P/R/F1 calculation

    :param constructed_edges: edges from a construction
    :param true_bonds: the true bond pairs
    :return: dict with precision, recall, f1, n_constructed, n_true
    '''
    C = {tuple(sorted(e)) for e in constructed_edges}
    T = {tuple(sorted(e)) for e in true_bonds}

    if len(C) == 0 or len(T) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_constructed": len(C), "n_true": len(T)}

    tp = len(C & T)
    p = tp / len(C)
    r = tp / len(T)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out = {"precision": p,
            "recall": r,
            "f1": f1,
            "n_constructed": len(C),
            "n_true": len(T)}
    return out