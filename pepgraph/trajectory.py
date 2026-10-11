"""Build peptide graphs from MD trajectories.

The MD runs under ``<runs>/*/md/*/`` store a small solute topology in
``traj_wrapped.pdb`` and the frames themselves in ``traj_wrapped.xtc`` -- around
20k frames each, so trajectories are read in chunks with ``mdtraj.iterload``
rather than loaded whole.

mdtraj is imported lazily: everything else in this package runs without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph import CONTACT_CUTOFF, graph_from_residues
from .residues import is_canonical
from .structure import BACKBONE, Residue

NM_TO_ANGSTROM = 10.0


def _peptide_backbone_index(topology):
    """[(resname, seq_id, {atom_name: atom_index})] for the longest peptide chain.

    Picks the chain with the most canonical residues, which drops the ligand
    chain and any ions without needing to name them.
    """
    best, best_n = None, 0
    for chain in topology.chains:
        n = sum(1 for r in chain.residues if is_canonical(r.name.upper()))
        if n > best_n:
            best, best_n = chain, n
    if best is None:
        return []

    out = []
    for res in best.residues:
        name = res.name.upper()
        if not is_canonical(name):
            continue
        atoms = {a.name: a.index for a in res.atoms if a.name in BACKBONE}
        if all(a in atoms for a in ("N", "CA", "C")):
            out.append((name, res.resSeq, atoms))
    return out


def _residues_for_frame(index, xyz):
    """Residue objects for one frame. ``xyz`` is [n_atoms, 3] in angstrom."""
    return [
        Residue(name=name, seq_id=seq_id,
                atoms={a: xyz[i] for a, i in atoms.items()})
        for name, seq_id, atoms in index
    ]


def iter_trajectory_graphs(xtc, top=None, stride: int = 1, chunk: int = 500,
                           max_frames: int = None, name: str = None,
                           contact_cutoff: float = CONTACT_CUTOFF):
    """Yield (frame_number, PeptideGraph) for each frame of a trajectory.

    ``stride`` subsamples frames; consecutive MD frames are highly correlated,
    so a stride of 10-100 loses little and costs proportionally less. pLDDT is
    0 for every node, since MD frames carry no confidence.
    """
    import mdtraj as md

    xtc = Path(xtc)
    top = Path(top) if top else xtc.with_suffix(".pdb")
    label = name or xtc.parent.name

    index = _peptide_backbone_index(md.load_topology(str(top)))
    if not index:
        return

    emitted, frame_no = 0, 0
    for batch in md.iterload(str(xtc), top=str(top), chunk=chunk, stride=stride):
        coords = batch.xyz * NM_TO_ANGSTROM
        for f in range(batch.n_frames):
            res_list = _residues_for_frame(index, coords[f])
            yield frame_no, graph_from_residues(
                res_list, f"{label}:f{frame_no}", contact_cutoff,
                use_plddt=False)
            frame_no += stride
            emitted += 1
            if max_frames and emitted >= max_frames:
                return


def find_trajectories(runs_dir, pattern: str = "*/md/*/*/traj_wrapped.xtc"):
    """Trajectories with a matching topology beside them, as (xtc, pdb) pairs."""
    pairs = []
    for xtc in sorted(Path(runs_dir).glob(pattern)):
        top = xtc.with_suffix(".pdb")
        if top.exists():
            pairs.append((xtc, top))
    return pairs


def torsion_arrays(xtc, top=None, stride: int = 1, chunk: int = 1000,
                   max_frames: int = None):
    """phi/psi/omega for every frame as [n_frames, n_residues] arrays, degrees.

    A fast path for looking at the angle distribution across an ensemble without
    building a graph per frame. Column i holds residue i, nan where undefined,
    matching the per-residue convention in :mod:`pepgraph.graph`.
    """
    import mdtraj as md

    xtc = Path(xtc)
    top = Path(top) if top else xtc.with_suffix(".pdb")
    index = _peptide_backbone_index(md.load_topology(str(top)))
    if not index:
        return None

    from .graph import backbone_torsions

    phis, psis, omegas = [], [], []
    emitted = 0
    for batch in md.iterload(str(xtc), top=str(top), chunk=chunk, stride=stride):
        coords = batch.xyz * NM_TO_ANGSTROM
        for f in range(batch.n_frames):
            phi, psi, omega = backbone_torsions(_residues_for_frame(index, coords[f]))
            phis.append(phi)
            psis.append(psi)
            omegas.append(omega)
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
        if max_frames and emitted >= max_frames:
            break

    return (np.array(phis), np.array(psis), np.array(omegas))
