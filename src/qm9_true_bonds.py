import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Key under which get_true_labels caches its result in the molecule dict.
CACHE_KEY = "_true_labels"

# gdb9.sdf is read once per process and reused by every later call.
SDF_BONDS = None
SDF_PATH_LOADED = None

TRUTH_COLUMNS = ["in_ring", "true_degree", "heavy_degree", "is_hydrogen",
                 "is_true_bond", "labels_ok", "smiles_match",
                 "n_true_bonds", "n_tp_bonds", "bond_precision",
                 "bond_recall", "bond_f1"]


def connectivity_smiles(mol):
    """
    Build a canonical SMILES that describes connectivity only. Hydrogens
    are removed, charges, radicals, aromaticity and stereo are cleared and
    every bond is made single, so two molecules match exactly when their
    heavy-atom bond graphs are the same.

    :param mol: RDKit molecule, or None
    :return: canonical SMILES string, or None if mol is None or RDKit fails
    """
    if mol is None:
        return None
    try:
        editable = Chem.RWMol(mol)

        hydrogen_indices = []
        for atom in editable.GetAtoms():
            if atom.GetAtomicNum() == 1:
                hydrogen_indices.append(atom.GetIdx())
        # Removing from the highest index down keeps the other indices valid.
        for index in sorted(hydrogen_indices, reverse=True):
            editable.RemoveAtom(index)

        for atom in editable.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.SetIsAromatic(False)

        for bond in editable.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
            bond.SetStereo(Chem.BondStereo.STEREONONE)

        return Chem.MolToSmiles(editable.GetMol(), isomericSmiles=False,
                                canonical=True)
    except Exception:
        # RDKit raises several different exception types on bad molecules.
        return None


def load_sdf_bonds(sdf_path):
    """
    Read the atoms and bonds of every molecule in gdb9.sdf. The table is
    kept in memory, so later calls with the same path return it directly.

    :param sdf_path: path to gdb9.sdf
    :return: {mol_id: (elements, bonds)}, where elements is a tuple of
        element symbols and bonds is a sorted tuple of (i, j) pairs, i < j
    """
    global SDF_BONDS, SDF_PATH_LOADED
    if SDF_BONDS is not None and SDF_PATH_LOADED == sdf_path:
        return SDF_BONDS

    print(f"Loading true bonds from {sdf_path} ...", flush=True)
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)

    table = {}
    n_unreadable = 0
    for record_index, rd_mol in enumerate(supplier):
        mol_id = record_index + 1
        if rd_mol is None:
            n_unreadable += 1
            continue

        # Molecules are identified by record position, so the "gdb_<id>"
        # name has to agree with it.
        if rd_mol.HasProp("_Name"):
            name = rd_mol.GetProp("_Name").strip()
            if name.startswith("gdb_"):
                name_id = int(name.split("_")[-1])
                if name_id != mol_id:
                    raise SystemExit(f"FATAL: SDF record {mol_id} is named '{name}' -- file order mismatch.")

        elements = []
        for atom in rd_mol.GetAtoms():
            elements.append(atom.GetSymbol())

        bonds = []
        for bond in rd_mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bonds.append((min(begin, end), max(begin, end)))
        bonds.sort()

        table[mol_id] = (tuple(elements), tuple(bonds))

    print(f"Loaded bonds for {len(table)} molecules ({n_unreadable} unreadable records)", flush=True)
    SDF_BONDS = table
    SDF_PATH_LOADED = sdf_path
    return table


def mol_from_bonds(elements, bonds):
    """
    Build a bare RDKit molecule from element symbols and bond pairs, with
    every bond single.

    :param elements: element symbols in atom order
    :param bonds: (i, j) atom index pairs
    :return: RDKit RWMol
    """
    mol = Chem.RWMol()
    for element in elements:
        mol.AddAtom(Chem.Atom(element))
    for begin, end in bonds:
        mol.AddBond(int(begin), int(end), Chem.BondType.SINGLE)
    return mol


def compute_smiles_match(elements, bonds, mol):
    """
    Check whether the SDF connectivity of a molecule matches the
    connectivity of its relaxed or its GDB-17 SMILES.

    :param elements: element symbols from gdb9.sdf
    :param bonds: bond pairs from gdb9.sdf
    :param mol: molecule dict with "smiles" and optionally "smiles_relaxed"
    :return: True or False, or None if the comparison cannot be made
    """
    try:
        sdf_connectivity = connectivity_smiles(mol_from_bonds(elements, bonds))
        if sdf_connectivity is None:
            return None
        relaxed_smiles = mol.get("smiles_relaxed", mol["smiles"])
        relaxed_connectivity = connectivity_smiles(
            Chem.MolFromSmiles(relaxed_smiles, sanitize=False))
        gdb17_connectivity = connectivity_smiles(
            Chem.MolFromSmiles(mol["smiles"], sanitize=False))
        if relaxed_connectivity is None and gdb17_connectivity is None:
            return None
        return (sdf_connectivity == relaxed_connectivity
                or sdf_connectivity == gdb17_connectivity)
    except Exception:
        return None


def assign_true_bonds(mol, charge=0):
    """
    Look up the true bonds of a molecule in the loaded SDF table and derive
    the per-atom labels. An atom is in a ring when it belongs to a
    biconnected component of at least three atoms, and heavy_degree counts
    the heavy neighbors of each heavy atom (-1 for hydrogens).

    :param mol: molecule dict with mol_id, num_atoms and atoms
    :param charge: unused, kept so existing callers still work
    :return: dict with ok, bonds, in_ring, true_degree, heavy_degree,
        smiles_match and error
    """
    n_atoms = mol["num_atoms"]
    atomic_numbers = np.asarray(mol["atoms"]["Z"])
    labels = {"ok": False,
              "bonds": [],
              "in_ring": np.zeros(n_atoms, dtype=bool),
              "true_degree": np.zeros(n_atoms, dtype=np.int32),
              "heavy_degree": np.full(n_atoms, -1, dtype=np.int32),
              "smiles_match": None,
              "error": None}

    if SDF_BONDS is None:
        raise RuntimeError("SDF bond table not loaded. Call load_sdf_bonds first.")

    if (atomic_numbers == 1).sum() == 0 and n_atoms > 1:
        labels["error"] = "no hydrogens present, but true-bond labels require them"
        return labels

    entry = SDF_BONDS.get(mol["mol_id"])
    if entry is None:
        labels["error"] = f"mol_id {mol['mol_id']} not found in gdb9.sdf"
        return labels
    sdf_elements, bonds = entry

    if sdf_elements != tuple(mol["atoms"]["element"]):
        labels["error"] = "element sequence mismatch between gdb9.sdf and xyz file"
        return labels

    labels["bonds"] = list(bonds)
    labels["ok"] = True

    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(range(n_atoms))
    bond_graph.add_edges_from(labels["bonds"])

    for component in nx.biconnected_components(bond_graph):
        if len(component) >= 3:
            for atom_index in component:
                labels["in_ring"][atom_index] = True

    for atom_index in range(n_atoms):
        labels["true_degree"][atom_index] = bond_graph.degree(atom_index)
        if atomic_numbers[atom_index] > 1:
            n_heavy = 0
            for neighbor in bond_graph.neighbors(atom_index):
                if atomic_numbers[neighbor] > 1:
                    n_heavy += 1
            labels["heavy_degree"][atom_index] = n_heavy

    labels["smiles_match"] = compute_smiles_match(sdf_elements, bonds, mol)
    return labels


def get_true_labels(mol, charge=0):
    """
    Return the true labels of a molecule, computing them only on the first
    call and caching them in the molecule dict.

    :param mol: molecule dict
    :param charge: passed through to assign_true_bonds
    :return: label dict from assign_true_bonds
    """
    if CACHE_KEY not in mol:
        mol[CACHE_KEY] = assign_true_bonds(mol, charge=charge)
    return mol[CACHE_KEY]


def f1_score(precision, recall):
    """
    Compute the harmonic mean of precision and recall.

    :param precision: precision value
    :param recall: recall value
    :return: F1 score, or 0.0 when both are 0
    """
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def attach_true_labels(graph, labels):
    """
    Write the true labels onto a constructed graph as plain scalars that
    serialize cleanly: an is_true_bond flag per edge, the atom labels per
    node, and the labeling status and bond recovery scores on the graph.

    :param graph: constructed networkx graph
    :param labels: label dict from get_true_labels
    :return: the same graph, modified in place
    """
    true_bonds = set()
    for bond in labels["bonds"]:
        true_bonds.add(tuple(bond))

    n_edges = graph.number_of_edges()
    n_true_positive = 0
    for edge in graph.edges:
        is_bond = (min(edge), max(edge)) in true_bonds
        graph.edges[edge]["is_true_bond"] = bool(is_bond)
        if is_bond:
            n_true_positive += 1

    precision = 0.0
    if n_edges:
        precision = n_true_positive / n_edges
    recall = 0.0
    if true_bonds:
        recall = n_true_positive / len(true_bonds)

    smiles_match = -1
    if labels["smiles_match"] is not None:
        smiles_match = int(labels["smiles_match"])

    graph.graph["labels_ok"] = int(labels["ok"])
    graph.graph["smiles_match"] = smiles_match
    graph.graph["n_true_bonds"] = len(true_bonds)
    graph.graph["n_tp_bonds"] = n_true_positive
    graph.graph["bond_precision"] = precision
    graph.graph["bond_recall"] = recall
    graph.graph["bond_f1"] = f1_score(precision, recall)

    for node in graph.nodes:
        attributes = graph.nodes[node]
        attributes["in_ring"] = bool(labels["in_ring"][node])
        attributes["true_degree"] = int(labels["true_degree"][node])
        attributes["heavy_degree"] = int(labels["heavy_degree"][node])
        attributes["is_hydrogen"] = attributes["Z"] == 1
    return graph


def edge_recovery_scores(constructed_edges, true_bonds):
    """
    Score a constructed edge set against the true bonds with precision,
    recall and F1.

    :param constructed_edges: (i, j) edges from a construction
    :param true_bonds: (i, j) true bond pairs
    :return: dict with precision, recall, f1, n_constructed and n_true.
             the scores are 0.0 if either set is empty
    """
    constructed_set = set()
    for begin, end in constructed_edges:
        constructed_set.add((min(begin, end), max(begin, end)))
    true_set = set()
    for begin, end in true_bonds:
        true_set.add((min(begin, end), max(begin, end)))

    if not constructed_set or not true_set:
        return {"precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "n_constructed": len(constructed_set),
                "n_true": len(true_set)}

    n_true_positive = len(constructed_set & true_set)
    precision = n_true_positive / len(constructed_set)
    recall = n_true_positive / len(true_set)
    return {"precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "n_constructed": len(constructed_set),
            "n_true": len(true_set)}