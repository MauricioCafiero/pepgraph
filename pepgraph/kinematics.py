"""Forward kinematics: backbone torsions -> N/CA/C coordinates, and back.

Needed because corrupting torsions is only half of a denoising task. The other
half is recomputing everything the graph derives from coordinates -- above all
the CA-CA contact distances, which otherwise still describe the *clean*
structure and hand the model the answer it is supposed to predict.

The backbone is treated as one chain of atoms, N(0), CA(0), C(0), N(1), ... and
each atom after the third is placed from the previous three by NeRF (natural
extension of reference frame). Along that chain the torsion for

    N(r)  is psi(r-1)      C(r-1) is the pivot
    CA(r) is omega(r-1)
    C(r)  is phi(r)

so the per-residue phi/psi/omega arrays map onto the atom chain exactly.
"""

from __future__ import annotations

import numpy as np

# Engh & Huber ideal protein backbone geometry.
BOND_N_CA = 1.458     # within a residue
BOND_CA_C = 1.525
BOND_C_N = 1.329      # the peptide bond
ANGLE_N_CA_C = 111.2  # tau
ANGLE_CA_C_N = 116.2
ANGLE_C_N_CA = 121.7

# per atom-in-residue (N, CA, C): the bond and angle used to place it
IDEAL_BONDS = (BOND_C_N, BOND_N_CA, BOND_CA_C)
IDEAL_ANGLES = (ANGLE_CA_C_N, ANGLE_C_N_CA, ANGLE_N_CA_C)


def place_atom(a, b, c, bond, angle_deg, torsion_deg):
    """Place D given A-B-C, |CD|, the B-C-D angle and the A-B-C-D torsion."""
    theta, chi = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-8:          # A, B, C collinear: any perpendicular will do
        n = np.cross(bc, np.array([1.0, 0.0, 0.0]))
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            n = np.cross(bc, np.array([0.0, 1.0, 0.0]))
            norm_n = np.linalg.norm(n)
    n /= norm_n
    m = np.stack([bc, np.cross(n, bc), n], axis=1)      # 3x3 frame, columns
    d = np.array([-bond * np.cos(theta),
                  bond * np.sin(theta) * np.cos(chi),
                  bond * np.sin(theta) * np.sin(chi)])
    return c + m @ d


def internal_coords(coords):
    """Measured bond lengths, bond angles and torsions along the atom chain.

    ``coords`` is [n_residues, 3, 3] of N/CA/C. Returns three arrays of length
    3*n_residues, with the first three entries unused (nan) since the first
    three atoms define the frame rather than being placed in it.
    """
    flat = coords.reshape(-1, 3)
    n = len(flat)
    bonds = np.full(n, np.nan)
    angles = np.full(n, np.nan)
    torsions = np.full(n, np.nan)
    for k in range(1, n):
        bonds[k] = np.linalg.norm(flat[k] - flat[k - 1])
    for k in range(2, n):
        v1 = flat[k - 2] - flat[k - 1]
        v2 = flat[k] - flat[k - 1]
        cosa = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angles[k] = np.degrees(np.arccos(np.clip(cosa, -1.0, 1.0)))
    from .graph import dihedral
    for k in range(3, n):
        torsions[k] = dihedral(flat[k - 3], flat[k - 2], flat[k - 1], flat[k])
    return bonds, angles, torsions


def torsion_chain(phi, psi, omega):
    """Per-residue phi/psi/omega -> the atom-chain torsion array.

    Entry 3r+j is the torsion that places atom j of residue r.
    """
    n_res = len(phi)
    chain = np.full(3 * n_res, np.nan)
    for r in range(1, n_res):
        chain[3 * r] = psi[r - 1]       # places N(r)
        chain[3 * r + 1] = omega[r - 1]  # places CA(r)
    for r in range(1, n_res):
        chain[3 * r + 2] = phi[r]        # places C(r)
    return chain


def build_backbone(torsions, bonds=None, angles=None, n_residues=None):
    """Atom-chain torsions -> coordinates [n_residues, 3, 3].

    ``bonds`` and ``angles`` default to ideal geometry; pass the measured
    arrays from :func:`internal_coords` to reproduce a structure exactly.
    The first three atoms are seeded in a canonical frame, so the result is
    correct up to a rigid-body transform and should be compared after
    superposition.
    """
    n_atoms = len(torsions)
    if n_residues is None:
        n_residues = n_atoms // 3
    flat = np.zeros((n_atoms, 3))

    def bond_for(k):
        return IDEAL_BONDS[k % 3] if bonds is None else bonds[k]

    def angle_for(k):
        return IDEAL_ANGLES[k % 3] if angles is None else angles[k]

    # seed: N at the origin, CA along +x, C in the xy plane
    b0, b1 = bond_for(1), bond_for(2)
    a1 = angle_for(2)
    flat[0] = (0.0, 0.0, 0.0)
    flat[1] = (b0, 0.0, 0.0)
    flat[2] = flat[1] + b1 * np.array(
        [-np.cos(np.radians(a1)), np.sin(np.radians(a1)), 0.0])

    for k in range(3, n_atoms):
        t = torsions[k]
        if np.isnan(t):
            t = 180.0   # a missing torsion falls back to extended
        flat[k] = place_atom(flat[k - 3], flat[k - 2], flat[k - 1],
                             bond_for(k), angle_for(k), t)

    return flat.reshape(n_residues, 3, 3)


def kabsch_rmsd(p, q):
    """RMSD between two point sets after optimal rigid superposition."""
    p = p.reshape(-1, 3).astype(np.float64)
    q = q.reshape(-1, 3).astype(np.float64)
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    u, s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((p @ rot) - q) ** 2).sum(axis=1).mean()))


def perturb_torsions(phi, psi, sigma_deg, rng, omega=None):
    """Add wrapped Gaussian noise of ``sigma_deg`` to phi and psi.

    Omega is left alone: it is held near planarity by a high barrier, so noising
    it would mostly teach the model to undo something that does not happen.
    """
    def noisy(a):
        out = a + rng.normal(0.0, sigma_deg, size=a.shape)
        return (out + 180.0) % 360.0 - 180.0
    return (noisy(phi), noisy(psi),
            omega if omega is None else omega.copy())
