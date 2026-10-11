"""Read backbone coordinates out of a structure file, with no heavy deps.

Handles the two formats this project actually produces: the flat mmCIF that
Boltz writes, and ordinary PDB (the MD topology / frame dumps). Only the four
backbone atoms are kept -- N, CA, C, O -- plus CB where present, which is all
the torsions and the contact map need.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field

import numpy as np

from .residues import is_canonical

BACKBONE = ("N", "CA", "C", "O", "CB")


def _open(path):
    """Open a structure file, transparently handling gzip (PDB serves .gz)."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


@dataclass
class Residue:
    name: str                       # three-letter code, e.g. "ARG"
    seq_id: int                     # author sequence number
    atoms: dict = field(default_factory=dict)   # atom name -> xyz array
    bfactor: float = 0.0            # pLDDT for Boltz output, 0 for MD

    def coord(self, atom: str):
        return self.atoms.get(atom)

    @property
    def complete_backbone(self) -> bool:
        return all(a in self.atoms for a in ("N", "CA", "C"))


def _finish(residues, min_length, allow_noncanonical=False):
    """Group a flat residue list by chain and keep the longest peptide.

    ``allow_noncanonical`` keeps modified or non-standard residues as long as
    their backbone is complete; they land in the one-hot's "other" slot. Short
    peptide structures from the PDB often contain a D-amino acid or a modified
    residue, and dropping the whole chain for one of them wastes most of them.
    """
    def keep(r):
        return r.complete_backbone and (allow_noncanonical or is_canonical(r.name))

    chains = {}
    for chain_id, res in residues:
        chains.setdefault(chain_id, []).append(res)

    best, best_n = None, 0
    for chain_id, res_list in chains.items():
        n = sum(1 for r in res_list if keep(r))
        if n > best_n:
            best, best_n = res_list, n

    if best is None or best_n < min_length:
        return None
    return [r for r in best if keep(r)]


def read_cif(path, min_length: int = 3, model: int = 1,
             allow_noncanonical: bool = False):
    """Parse the _atom_site loop of a flat mmCIF. Returns a residue list."""
    columns, rows, in_loop = [], [], False
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("_atom_site."):
                columns.append(line.split(".", 1)[1].strip())
                in_loop = True
                continue
            if in_loop:
                if line.startswith(("ATOM", "HETATM")):
                    rows.append(line.split())
                elif line.strip() in ("#", "") or line.startswith("loop_"):
                    if rows:
                        break
                    in_loop = False
                    columns = []
    if not rows:
        return None

    col = {name: i for i, name in enumerate(columns)}
    need = ("label_atom_id", "auth_comp_id", "auth_seq_id", "auth_asym_id",
            "Cartn_x", "Cartn_y", "Cartn_z")
    # Boltz writes auth_* and label_*; fall back to label_* if auth is absent.
    alt = {"auth_comp_id": "label_comp_id", "auth_seq_id": "label_seq_id",
           "auth_asym_id": "label_asym_id"}
    for key in need:
        if key not in col and key in alt and alt[key] in col:
            col[key] = col[alt[key]]
    if any(k not in col for k in need):
        return None

    model_col = col.get("pdbx_PDB_model_num")
    bf_col = col.get("B_iso_or_equiv")

    residues, current, current_key = [], None, None
    for row in rows:
        if model_col is not None and int(row[model_col]) != model:
            continue
        atom = row[col["label_atom_id"]].strip('"')
        if atom not in BACKBONE:
            continue
        chain = row[col["auth_asym_id"]]
        seq_id = int(row[col["auth_seq_id"]])
        key = (chain, seq_id)
        if key != current_key:
            current = Residue(name=row[col["auth_comp_id"]].upper(), seq_id=seq_id)
            if bf_col is not None:
                try:
                    current.bfactor = float(row[bf_col])
                except ValueError:
                    pass
            residues.append((chain, current))
            current_key = key
        current.atoms[atom] = np.array(
            [float(row[col["Cartn_x"]]), float(row[col["Cartn_y"]]),
             float(row[col["Cartn_z"]])], dtype=np.float64)

    return _finish(residues, min_length, allow_noncanonical)


def read_pdb(path, min_length: int = 3, allow_noncanonical: bool = False):
    """Parse the first model of a PDB file by fixed columns."""
    residues, current, current_key = [], None, None
    with _open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom not in BACKBONE:
                continue
            chain = line[21]
            seq_id = int(line[22:26])
            key = (chain, seq_id)
            if key != current_key:
                current = Residue(name=line[17:20].strip().upper(), seq_id=seq_id)
                try:
                    current.bfactor = float(line[60:66])
                except ValueError:
                    pass
                residues.append((chain, current))
                current_key = key
            current.atoms[atom] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64)

    return _finish(residues, min_length, allow_noncanonical)


def read_structure(path, min_length: int = 3, allow_noncanonical: bool = False):
    """Dispatch on extension. Returns a residue list, or None if unusable."""
    path = str(path)
    stem = path[:-3] if path.endswith(".gz") else path
    if stem.endswith((".cif", ".mmcif")):
        return read_cif(path, min_length, allow_noncanonical=allow_noncanonical)
    if stem.endswith((".pdb", ".ent")):
        return read_pdb(path, min_length, allow_noncanonical=allow_noncanonical)
    raise ValueError(f"unsupported structure format: {path}")
