"""Select, extract and reload MD frames.

What gets written to disk is backbone **coordinates**, not finished graphs.
Node features are the same in every frame of a sequence (identity, terminus,
and pLDDT is 0 for MD), so storing graphs per frame would duplicate hundreds of
megabytes of unchanging data. Coordinates are also the more reusable artifact:
the contact cutoff stays a parameter you can change on reload instead of being
baked into the stored edges.

Selection applies two rules established by inspecting the runs:

* **Deduplicate by run.** ``first_5ns`` is an exact prefix of ``first_10ns`` is
  an exact prefix of ``first_15ns`` is an exact prefix of ``prod_L1_modal``, and
  ``prod_20ns``/``seg_20_25ns`` sit inside ``prod_25ns``. Only the
  longest-spanning stage of each run carries unique frames.
* **Normalise the save interval.** The local ``prod_*`` lane saves every 1 ps
  and the Modal lane every 10 ps, so frames are strided to a common spacing
  before being pooled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph import CONTACT_CUTOFF, graph_from_residues
from .residues import THREE_TO_ONE
from .structure import Residue

BACKBONE_3 = ("N", "CA", "C")


@dataclass
class TrajectorySelection:
    xtc: Path
    top: Path
    run: str
    stage: str
    ligand: str
    sequence: str
    n_frames: int
    dt_ps: float
    stride: int          # stride to reach the target spacing
    n_selected: int      # frames after stride and cap

    @property
    def span_ns(self) -> float:
        return self.n_frames * self.dt_ps / 1000.0


def describe_trajectories(runs_dir, pattern="*/md/*/*/traj_wrapped.xtc"):
    """Read frame count, save interval and sequence for every trajectory."""
    import mdtraj as md

    from .trajectory import _peptide_backbone_index

    out = []
    for xtc in sorted(Path(runs_dir).glob(pattern)):
        top = xtc.with_suffix(".pdb")
        if not top.exists():
            continue
        with md.formats.XTCTrajectoryFile(str(xtc)) as fh:
            n = len(fh)
        if n < 2:
            continue
        head = md.load(str(xtc), top=str(top))[:2]
        dt = float(head.time[1] - head.time[0])
        index = _peptide_backbone_index(md.load_topology(str(top)))
        if not index:
            continue
        seq = "".join(THREE_TO_ONE.get(name, "X") for name, _, _ in index)
        out.append({
            "xtc": xtc, "top": top, "n_frames": n, "dt_ps": dt,
            "sequence": seq,
            "run": xtc.parents[1].name,
            "stage": xtc.parent.name,
            "ligand": xtc.parents[3].name,
        })
    return out


def select_trajectories(described, target_dt_ps: float = 10.0,
                        per_sequence_cap: int = None):
    """Apply the dedup and stride rules, then cap frames per sequence.

    The cap is spread evenly over the trajectories of a sequence rather than
    taken from the first ones, so a sequence with one 20 ns run and one 5 ns run
    contributes from both.
    """
    # keep the longest-spanning stage per run
    best = {}
    for d in described:
        span = d["n_frames"] * d["dt_ps"]
        prev = best.get(d["run"])
        if prev is None or span > prev["n_frames"] * prev["dt_ps"]:
            best[d["run"]] = d

    selections = []
    for d in best.values():
        stride = max(1, int(round(target_dt_ps / d["dt_ps"])))
        selections.append(TrajectorySelection(
            xtc=d["xtc"], top=d["top"], run=d["run"], stage=d["stage"],
            ligand=d["ligand"], sequence=d["sequence"],
            n_frames=d["n_frames"], dt_ps=d["dt_ps"], stride=stride,
            n_selected=len(range(0, d["n_frames"], stride)),
        ))

    if per_sequence_cap:
        by_seq = {}
        for s in selections:
            by_seq.setdefault(s.sequence, []).append(s)
        for group in by_seq.values():
            total = sum(s.n_selected for s in group)
            if total <= per_sequence_cap:
                continue
            # even share, with the remainder going to the longest trajectories
            share = per_sequence_cap // len(group)
            for s in group:
                s.n_selected = min(s.n_selected, share)
            left = per_sequence_cap - sum(s.n_selected for s in group)
            for s in sorted(group, key=lambda s: -s.n_frames):
                if left <= 0:
                    break
                room = len(range(0, s.n_frames, s.stride)) - s.n_selected
                add = min(room, left)
                s.n_selected += add
                left -= add

    return sorted(selections, key=lambda s: (s.ligand, s.run))


def extract_backbone(sel: TrajectorySelection, chunk: int = 500):
    """[n_selected, n_residues, 3, 3] float32 of N/CA/C, in angstrom."""
    import mdtraj as md

    from .trajectory import NM_TO_ANGSTROM, _peptide_backbone_index

    index = _peptide_backbone_index(md.load_topology(str(sel.top)))
    atom_idx = np.array([[atoms[a] for a in BACKBONE_3]
                         for _, _, atoms in index])          # [n_res, 3]

    frames, numbers, taken = [], [], 0
    for batch in md.iterload(str(sel.xtc), top=str(sel.top), chunk=chunk,
                             stride=sel.stride):
        xyz = batch.xyz * NM_TO_ANGSTROM
        for f in range(batch.n_frames):
            frames.append(xyz[f][atom_idx])
            numbers.append(taken * sel.stride)
            taken += 1
            if taken >= sel.n_selected:
                break
        if taken >= sel.n_selected:
            break

    return (np.asarray(frames, dtype=np.float32),
            np.asarray(numbers, dtype=np.int32))


def write_shard(path, sel: TrajectorySelection, coords, frame_numbers):
    np.savez_compressed(
        path, coords=coords, frame_numbers=frame_numbers,
        meta=json.dumps({
            "sequence": sel.sequence, "run": sel.run, "stage": sel.stage,
            "ligand": sel.ligand, "dt_ps": sel.dt_ps * sel.stride,
            "source_frames": sel.n_frames, "stride": sel.stride,
            "atoms": list(BACKBONE_3),
        }))


def load_shard(path, contact_cutoff: float = CONTACT_CUTOFF, limit: int = None):
    """Reload a shard as (meta, [PeptideGraph]).

    Graphs are rebuilt from the stored coordinates, so ``contact_cutoff`` can
    differ from whatever was used when the shard was written.
    """
    with np.load(path, allow_pickle=False) as z:
        coords = z["coords"]
        numbers = z["frame_numbers"]
        meta = json.loads(str(z["meta"]))

    seq = meta["sequence"]
    three = {v: k for k, v in THREE_TO_ONE.items()}
    names = [three.get(c, "UNK") for c in seq]

    graphs = []
    for i in range(len(coords) if limit is None else min(limit, len(coords))):
        res_list = [
            Residue(name=names[r], seq_id=r + 1,
                    atoms={a: coords[i, r, j] for j, a in enumerate(BACKBONE_3)})
            for r in range(coords.shape[1])
        ]
        graphs.append(graph_from_residues(
            res_list, f"{meta['run']}:{meta['stage']}:f{int(numbers[i])}",
            contact_cutoff, use_plddt=False))
    return meta, graphs


def iter_shards(shard_dir, **kwargs):
    """Yield (meta, graphs) for every shard in a directory."""
    for path in sorted(Path(shard_dir).glob("*.npz")):
        yield load_shard(path, **kwargs)
