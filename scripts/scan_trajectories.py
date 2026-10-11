"""Inventory the MD trajectories: how many frames, how many distinct sequences,
and how fast graphs come out of them.

Sequence count is the number that matters. Frames are plentiful and correlated;
distinct sequences are the scarce axis, and this says how scarce.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import mdtraj as md

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph import find_trajectories, iter_trajectory_graphs  # noqa: E402
from pepgraph.residues import THREE_TO_ONE  # noqa: E402
from pepgraph.trajectory import _peptide_backbone_index  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--bench", type=int, default=200,
                    help="frames to featurise for the rate benchmark")
    args = ap.parse_args()

    pairs = find_trajectories(args.runs)
    print(f"{len(pairs)} trajectories with a topology beside them\n")

    total_frames = 0
    by_sequence = defaultdict(list)
    by_ligand = Counter()
    rows = []

    for xtc, top in pairs:
        index = _peptide_backbone_index(md.load_topology(str(top)))
        seq = "".join(THREE_TO_ONE.get(n, "X") for n, _, _ in index)
        n = int(md.load(str(xtc), top=str(top), stride=1000).n_frames)
        # stride=1000 gives a cheap lower bound; get the true count from the file
        with md.formats.XTCTrajectoryFile(str(xtc)) as fh:
            n = len(fh)
        total_frames += n
        # ligand = the run's parent-of-md directory
        ligand = xtc.parents[2].name if xtc.parents[2].name != "md" else xtc.parents[3].name
        by_ligand[ligand] += n
        by_sequence[seq].append((xtc.parent.parent.name, xtc.parent.name, n))
        rows.append((ligand, xtc.parent.parent.name, xtc.parent.name, len(seq), n))

    print(f"total frames: {total_frames:,}")
    print(f"distinct peptide sequences: {len(by_sequence)}")
    for lig, n in by_ligand.most_common():
        print(f"  {lig:14s} {n:>10,} frames")

    print("\nframes per distinct sequence:")
    for seq, entries in sorted(by_sequence.items(), key=lambda kv: -sum(e[2] for e in kv[1])):
        tot = sum(e[2] for e in entries)
        print(f"  {seq[:36]:36s} len={len(seq):2d}  {tot:>9,} frames  "
              f"in {len(entries)} run(s)")

    print("\nframes per stage (first few runs):")
    for lig, run, stage, ln, n in rows[:12]:
        print(f"  {lig:12s} {run:24s} {stage:16s} len={ln:2d} {n:>8,}")

    # featurisation rate
    xtc, top = pairs[0]
    t0 = time.time()
    count = 0
    for _, g in iter_trajectory_graphs(xtc, top, max_frames=args.bench):
        count += 1
        last = g
    dt = time.time() - t0
    print(f"\nfeaturised {count} frames in {dt:.1f}s = {count / dt:.0f} frames/s")
    print(f"  per graph: {last.n_residues} nodes, {last.edge_index.shape[1]} edges, "
          f"x {last.x.shape}, edge_attr {last.edge_attr.shape}")
    print(f"  all {total_frames:,} frames would take "
          f"{total_frames / (count / dt) / 3600:.1f} h at this rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
