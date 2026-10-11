"""Ramachandran of the MD ensemble, beside the Boltz co-folds.

Two things are being checked. That the trajectory path produces the same kind
of geometry as the CIF path -- a units or atom-order mistake in the mdtraj
reader would show up as a displaced or smeared distribution. And how much the
ensemble actually differs from the single predicted structures, which is the
question of whether the frames carry information the co-folds do not.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph import find_trajectories, graph_from_structure, torsion_arrays  # noqa: E402

BASINS = {
    "alpha-R": (-160, -20, -120, 50),
    "beta/PPII": (-180, -20, 50, 180),
    "alpha-L": (20, 125, -20, 90),
}


def classify(phi, psi):
    for name, (p0, p1, s0, s1) in BASINS.items():
        if p0 <= phi <= p1 and s0 <= psi <= s1:
            return name
    return "outlier"


def basin_table(phi, psi, label):
    ok = ~np.isnan(phi) & ~np.isnan(psi)
    counts = Counter(classify(a, b) for a, b in zip(phi[ok], psi[ok]))
    n = int(ok.sum())
    print(f"\n{label} ({n:,} residue observations)")
    for name in list(BASINS) + ["outlier"]:
        c = counts.get(name, 0)
        print(f"  {name:11s} {c:8,}  {100 * c / n:5.1f}%")
    return phi[ok], psi[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--stride", type=int, default=200,
                    help="frame stride; MD frames are highly correlated")
    ap.add_argument("--out", type=Path, default=Path("outputs/rama_md_vs_boltz.png"))
    args = ap.parse_args()

    # --- Boltz co-folds
    b_phi, b_psi = [], []
    for p in sorted(args.runs.glob("*/boltz/boltz_results_*/predictions/*/*_model_0.cif")):
        g = graph_from_structure(p, name=p.parent.name)
        if g:
            b_phi.append(g.phi)
            b_psi.append(g.psi)
    b_phi, b_psi = np.concatenate(b_phi), np.concatenate(b_psi)

    # --- MD frames
    pairs = find_trajectories(args.runs)
    m_phi, m_psi, m_om, n_frames = [], [], [], 0
    for i, (xtc, top) in enumerate(pairs, 1):
        res = torsion_arrays(xtc, top, stride=args.stride)
        if res is None:
            continue
        phi, psi, om = res
        n_frames += phi.shape[0]
        m_phi.append(phi.ravel())
        m_psi.append(psi.ravel())
        m_om.append(om.ravel())
        if i % 20 == 0:
            print(f"  ...{i}/{len(pairs)} trajectories, {n_frames:,} frames")
    m_phi = np.concatenate(m_phi)
    m_psi = np.concatenate(m_psi)
    m_om = np.concatenate(m_om)
    print(f"\nsampled {n_frames:,} frames at stride {args.stride}")

    bp, bs = basin_table(b_phi, b_psi, "Boltz co-folds")
    mp, ms = basin_table(m_phi, m_psi, "MD ensemble")

    om = m_om[~np.isnan(m_om)]
    print(f"\nMD omega: trans {100 * (np.abs(om) > 150).mean():.1f}%, "
          f"cis {int((np.abs(om) < 30).sum()):,} "
          f"({100 * (np.abs(om) < 30).mean():.2f}%), "
          f"mean |omega| {np.abs(om).mean():.1f} deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, (x, y, title) in zip(axes, [
        (bp, bs, f"Boltz co-folds  ({len(bp):,} residues)"),
        (mp, ms, f"MD ensemble  ({len(mp):,} observations)"),
    ]):
        ax.hexbin(x, y, gridsize=72, bins="log", extent=(-180, 180, -180, 180),
                  cmap="viridis", mincnt=1)
        ax.set(xlim=(-180, 180), ylim=(-180, 180), title=title,
               xlabel="phi (deg)")
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-180, 181, 90))
        for p0, p1, s0, s1 in BASINS.values():
            ax.add_patch(plt.Rectangle((p0, s0), p1 - p0, s1 - s0, fill=False,
                                       ec="white", lw=0.9, ls="--", alpha=0.8))
    axes[0].set_ylabel("psi (deg)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
