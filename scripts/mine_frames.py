"""Mine MD frames into coordinate shards under outputs/frames/.

Reads only -- nothing outside this repo is written to. Run with --dry-run first
to see what would be selected and how large it would be.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.mining import (describe_trajectories, extract_backbone,  # noqa: E402
                             select_trajectories, write_shard)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--out", type=Path, default=Path("outputs/frames"))
    ap.add_argument("--target-dt", type=float, default=10.0,
                    help="frame spacing in ps to normalise every run to")
    ap.add_argument("--cap", type=int, default=4000,
                    help="max frames per distinct sequence (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("reading trajectory metadata...")
    described = describe_trajectories(args.runs)
    print(f"  {len(described)} trajectories, "
          f"{sum(d['n_frames'] for d in described):,} frames on disk, "
          f"{len({d['sequence'] for d in described})} distinct sequences")

    sels = select_trajectories(described, target_dt_ps=args.target_dt,
                               per_sequence_cap=args.cap or None)
    dropped = len(described) - len(sels)
    total = sum(s.n_selected for s in sels)

    print(f"\nafter dedup: {len(sels)} trajectories kept, {dropped} dropped as "
          f"duplicate stages of the same run")
    print(f"after stride to {args.target_dt:g} ps"
          + (f" and cap of {args.cap:,}/sequence" if args.cap else "")
          + f": {total:,} frames")

    by_seq = Counter()
    for s in sels:
        by_seq[s.sequence] += s.n_selected
    print(f"\n{len(by_seq)} sequences, "
          f"{min(by_seq.values()):,}-{max(by_seq.values()):,} frames each:")
    for seq, n in by_seq.most_common():
        print(f"  {seq[:36]:36s} {n:>6,}")

    n_res = {len(s.sequence) for s in sels}
    approx_mb = total * max(n_res) * 3 * 3 * 4 / 1e6
    print(f"\nestimated shard size: {approx_mb:.0f} MB uncompressed "
          f"(backbone N/CA/C only)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    manifest, t0 = [], time.time()
    for i, s in enumerate(sels, 1):
        coords, numbers = extract_backbone(s)
        name = f"{s.ligand}__{s.run}__{s.stage}.npz"
        write_shard(args.out / name, s, coords, numbers)
        manifest.append({
            "shard": name, "ligand": s.ligand, "run": s.run, "stage": s.stage,
            "sequence": s.sequence, "n_residues": len(s.sequence),
            "n_frames": coords.shape[0], "stride": s.stride,
            "dt_ps": s.dt_ps * s.stride, "source_frames": s.n_frames,
            "source_span_ns": s.span_ns,
        })
        print(f"  [{i}/{len(sels)}] {name}  {coords.shape[0]:,} frames  "
              f"{(args.out / name).stat().st_size / 1e6:.1f} MB")

    with open(args.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    size = sum(p.stat().st_size for p in args.out.glob("*.npz")) / 1e6
    print(f"\nwrote {len(manifest)} shards, {total:,} frames, {size:.0f} MB "
          f"in {time.time() - t0:.0f}s -> {args.out}")
    print(f"manifest: {args.out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
