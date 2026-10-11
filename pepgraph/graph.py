"""Residue-level peptide graphs: nodes are residues, edges are peptide bonds
plus spatial contacts.

Node features (32):
    [0:21]  one-hot residue identity (20 canonical + 1 "other")
    [21:24] one-hot position: N-terminal / middle / C-terminal
    [24:31] charge, mass, aromatic, hydropathy, volume, H-bond donors/acceptors
    [31]    pLDDT / 100 (0 when the source has no confidence, e.g. MD frames)

Edge features (10), on each directed edge:
    [0:2]  sin, cos of psi   -- the three torsions spanning the peptide bond
    [2:4]  sin, cos of omega     from residue i to residue i+1, so the link
    [4:6]  sin, cos of phi       itself is described rather than the residues
    [6]    is a backbone edge
    [7]    is a contact edge
    [8]    direction: +1 N->C, -1 C->N, 0 for a contact
    [9]    CA-CA distance / 10

An angle that cannot be computed is stored as sin = cos = 0, which no real
angle produces (sin^2 + cos^2 = 1), so it needs no separate mask bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import residues as R
from .structure import read_structure

NODE_DIM = R.N_IDENTITY + R.N_TERMINUS + R.N_PHYSCHEM + 1   # 32
EDGE_DIM = 10
CONTACT_CUTOFF = 8.0     # angstrom, CA-CA
CONTACT_MIN_SEP = 3      # |i-j| must exceed this to count as a contact


@dataclass
class PeptideGraph:
    name: str
    sequence: str                # one-letter, len == n_residues
    x: np.ndarray                # [n_res, NODE_DIM]      float32
    edge_index: np.ndarray       # [2, n_edges]           int64
    edge_attr: np.ndarray        # [n_edges, EDGE_DIM]    float32
    phi: np.ndarray              # [n_res] degrees, nan where undefined
    psi: np.ndarray              # [n_res]
    omega: np.ndarray            # [n_res]
    y: float = float("nan")      # label slot, unused for now

    @property
    def n_residues(self) -> int:
        return self.x.shape[0]


def dihedral(p0, p1, p2, p3) -> float:
    """Signed dihedral in degrees, IUPAC convention."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    # project b0 and b2 onto the plane perpendicular to the central bond
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def backbone_torsions(res_list):
    """phi, psi, omega per residue in degrees; nan where undefined.

    phi_i   = C(i-1) - N(i)  - CA(i) - C(i)
    psi_i   = N(i)   - CA(i) - C(i)  - N(i+1)
    omega_i = CA(i)  - C(i)  - N(i+1) - CA(i+1)

    Residues that are not actually bonded (a chain break, or a gap in the
    numbering) give nan rather than a meaningless angle.
    """
    n = len(res_list)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    omega = np.full(n, np.nan)

    def bonded(i, j):
        """True if residue j follows i through a real peptide bond."""
        c, nxt = res_list[i].coord("C"), res_list[j].coord("N")
        if c is None or nxt is None:
            return False
        return np.linalg.norm(nxt - c) < 2.0   # C-N is ~1.33 A

    for i in range(n):
        r = res_list[i]
        if i > 0 and bonded(i - 1, i):
            phi[i] = dihedral(res_list[i - 1].coord("C"), r.coord("N"),
                              r.coord("CA"), r.coord("C"))
        if i + 1 < n and bonded(i, i + 1):
            nxt = res_list[i + 1]
            psi[i] = dihedral(r.coord("N"), r.coord("CA"),
                              r.coord("C"), nxt.coord("N"))
            omega[i] = dihedral(r.coord("CA"), r.coord("C"),
                                nxt.coord("N"), nxt.coord("CA"))
    return phi, psi, omega


def node_features(res_list, use_plddt: bool = True) -> np.ndarray:
    n = len(res_list)
    x = np.zeros((n, NODE_DIM), dtype=np.float32)
    off_term = R.N_IDENTITY
    off_phys = off_term + R.N_TERMINUS
    for i, r in enumerate(res_list):
        x[i, R.identity_index(r.name)] = 1.0
        x[i, off_term + (0 if i == 0 else 2 if i == n - 1 else 1)] = 1.0
        x[i, off_phys:off_phys + R.N_PHYSCHEM] = R.physchem(r.name)
        if use_plddt:
            x[i, -1] = r.bfactor / 100.0
    return x


def _angle_pair(deg):
    """(sin, cos) of an angle in degrees; (0, 0) when undefined."""
    if np.isnan(deg):
        return 0.0, 0.0
    rad = np.radians(deg)
    return float(np.sin(rad)), float(np.cos(rad))


def build_edges(res_list, phi, psi, omega,
                contact_cutoff: float = CONTACT_CUTOFF):
    """Backbone edges (both directions) plus symmetric contact edges."""
    n = len(res_list)
    ca = np.array([r.coord("CA") for r in res_list])
    src, dst, attr = [], [], []

    def add(i, j, feats):
        src.append(i)
        dst.append(j)
        attr.append(feats)

    # backbone: the torsions of the i -> i+1 link are psi_i, omega_i, phi_{i+1}
    for i in range(n - 1):
        if np.isnan(psi[i]) and np.isnan(omega[i]):
            continue    # not actually bonded; a contact edge may still appear
        sp, cp = _angle_pair(psi[i])
        so, co = _angle_pair(omega[i])
        sf, cf = _angle_pair(phi[i + 1])
        d = float(np.linalg.norm(ca[i + 1] - ca[i])) / 10.0
        add(i, i + 1, [sp, cp, so, co, sf, cf, 1.0, 0.0, 1.0, d])
        add(i + 1, i, [sp, cp, so, co, sf, cf, 1.0, 0.0, -1.0, d])

    # contacts
    for i in range(n):
        for j in range(i + CONTACT_MIN_SEP + 1, n):
            d = float(np.linalg.norm(ca[j] - ca[i]))
            if d <= contact_cutoff:
                feats = [0.0] * 6 + [0.0, 1.0, 0.0, d / 10.0]
                add(i, j, feats)
                add(j, i, list(feats))

    if not src:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, EDGE_DIM), dtype=np.float32))
    return (np.array([src, dst], dtype=np.int64),
            np.array(attr, dtype=np.float32))


def graph_from_structure(path, name: str = None, min_length: int = 3,
                         contact_cutoff: float = CONTACT_CUTOFF,
                         use_plddt: bool = True):
    """Structure file -> PeptideGraph. Returns None if no usable peptide."""
    res_list = read_structure(path, min_length=min_length)
    if not res_list or len(res_list) < min_length:
        return None
    return graph_from_residues(
        res_list, name or str(path), contact_cutoff, use_plddt)


def graph_from_residues(res_list, name: str,
                        contact_cutoff: float = CONTACT_CUTOFF,
                        use_plddt: bool = True) -> PeptideGraph:
    phi, psi, omega = backbone_torsions(res_list)
    edge_index, edge_attr = build_edges(
        res_list, phi, psi, omega, contact_cutoff)
    return PeptideGraph(
        name=name,
        sequence="".join(R.THREE_TO_ONE.get(r.name, "X") for r in res_list),
        x=node_features(res_list, use_plddt),
        edge_index=edge_index,
        edge_attr=edge_attr,
        phi=phi, psi=psi, omega=omega,
    )
