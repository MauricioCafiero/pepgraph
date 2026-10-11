"""Per-residue property tables for peptide graph node features.

The values are literature standards, rounded: residue mass is the mass the
residue contributes to a chain (free amino acid minus water), hydropathy is
Kyte-Doolittle, volume is the Zamyatnin residue volume, and the H-bond counts
are side-chain donor/acceptor *groups* counted by hand. The H-bond counts and
the 0.1 charge on histidine are approximations; they are there to give the
model a physicochemical prior on small data, not to be exact.
"""

from __future__ import annotations

# Canonical order; index in this list is the one-hot position.
AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA3_INDEX = {name: i for i, name in enumerate(AA3)}
UNKNOWN_INDEX = len(AA3)  # 20: one extra slot for anything non-canonical
N_IDENTITY = len(AA3) + 1  # 21

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# name: (charge at pH 7, residue mass Da, aromatic, Kyte-Doolittle,
#         side-chain volume A^3, sc H-bond donors, sc H-bond acceptors)
PROPERTIES = {
    "ALA": (0.0,  71.08, 0, 1.8,  88.6, 0, 0),
    "ARG": (1.0, 156.19, 0, -4.5, 173.4, 3, 0),
    "ASN": (0.0, 114.10, 0, -3.5, 114.1, 1, 1),
    "ASP": (-1.0, 115.09, 0, -3.5, 111.1, 0, 2),
    "CYS": (0.0, 103.14, 0, 2.5, 108.5, 1, 0),
    "GLN": (0.0, 128.13, 0, -3.5, 143.8, 1, 1),
    "GLU": (-1.0, 129.12, 0, -3.5, 138.4, 0, 2),
    "GLY": (0.0,  57.05, 0, -0.4,  60.1, 0, 0),
    "HIS": (0.1, 137.14, 1, -3.2, 153.2, 1, 1),
    "ILE": (0.0, 113.16, 0, 4.5, 166.7, 0, 0),
    "LEU": (0.0, 113.16, 0, 3.8, 166.7, 0, 0),
    "LYS": (1.0, 128.17, 0, -3.9, 168.6, 1, 0),
    "MET": (0.0, 131.19, 0, 1.9, 162.9, 0, 1),
    "PHE": (0.0, 147.18, 1, 2.8, 189.9, 0, 0),
    "PRO": (0.0,  97.12, 0, -1.6, 112.7, 0, 0),
    "SER": (0.0,  87.08, 0, -0.8,  89.0, 1, 1),
    "THR": (0.0, 101.10, 0, -0.7, 116.1, 1, 1),
    "TRP": (0.0, 186.21, 1, -0.9, 227.8, 1, 0),
    "TYR": (0.0, 163.18, 1, -1.3, 193.6, 1, 1),
    "VAL": (0.0,  99.13, 0, 4.2, 140.0, 0, 0),
}

# Mean of the table, used for a non-canonical residue so its features are
# neutral rather than zero (zero would read as glycine-like and uncharged).
_MEAN = tuple(
    sum(p[i] for p in PROPERTIES.values()) / len(PROPERTIES) for i in range(7)
)

N_PHYSCHEM = 7
N_TERMINUS = 3  # N-terminal / middle / C-terminal


def physchem(resname: str) -> tuple:
    """The 7 physicochemical features, scaled to roughly unit range."""
    charge, mass, aromatic, kd, vol, don, acc = PROPERTIES.get(resname, _MEAN)
    return (charge, mass / 100.0, float(aromatic), kd / 4.5, vol / 100.0,
            don / 3.0, acc / 2.0)


def identity_index(resname: str) -> int:
    return AA3_INDEX.get(resname, UNKNOWN_INDEX)


def is_canonical(resname: str) -> bool:
    return resname in AA3_INDEX
