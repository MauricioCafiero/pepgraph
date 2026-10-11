"""Featurise every co-fold prediction under --runs and check the
torsions against the Ramachandran plot.

The point is not the picture: if phi/psi come out in the canonical basins the
backbone atoms are being read in the right order, and if they do not, the
featuriser is wrong in a way that would be invisible after training.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph import graph_from_structure  # noqa: E402


# Rough boundaries, only used to report what fraction lands where.
BASINS = {
    "alpha-R":  (-160, -20, -120, 50),
    "beta/PPII": (-180, -20, 50, 180),
    "alpha-L":  (20, 125, -20, 90),
}


def classify(phi, psi):
    for name, (p0, p1, s0, s1) in BASINS.items():
        if p0 <= phi <= p1 and s0 <= psi <= s1:
            return name
    return "outlier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*/boltz/boltz_results_*/predictions/*/*_model_0.cif")
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--out", type=Path, default=Path("outputs/rama_boltz.png"))
    args = ap.parse_args()

    paths = sorted(args.runs.glob(args.glob))
    print(f"found {len(paths)} structures under {args.runs}")

    graphs, failed = [], []
    for p in paths:
        try:
            g = graph_from_structure(p, name=p.parent.name)
        except Exception as exc:                      # noqa: BLE001
            failed.append((p.name, repr(exc)))
            continue
        if g is None:
            failed.append((p.name, "no usable peptide chain"))
        else:
            graphs.append(g)

    print(f"featurised {len(graphs)}, failed {len(failed)}")
    for name, why in failed[:5]:
        print(f"  FAIL {name}: {why}")

    if not graphs:
        return 1

    phi = np.concatenate([g.phi for g in graphs])
    psi = np.concatenate([g.psi for g in graphs])
    omega = np.concatenate([g.omega for g in graphs])
    ok = ~np.isnan(phi) & ~np.isnan(psi)

    lengths = [g.n_residues for g in graphs]
    print(f"\nresidues: {sum(lengths)} total, "
          f"length {min(lengths)}-{max(lengths)} (median {int(np.median(lengths))})")
    print(f"phi/psi defined for {ok.sum()} of {len(phi)} residues "
          f"(the rest are chain termini, as expected)")

    counts = Counter(classify(a, b) for a, b in zip(phi[ok], psi[ok]))
    print("\nRamachandran basins:")
    for name in list(BASINS) + ["outlier"]:
        n = counts.get(name, 0)
        print(f"  {name:11s} {n:6d}  {100 * n / ok.sum():5.1f}%")

    om = omega[~np.isnan(omega)]
    trans = int((np.abs(om) > 150).sum())
    cis = int((np.abs(om) < 30).sum())
    print(f"\nomega: {len(om)} defined, trans {100 * trans / len(om):.1f}%, "
          f"cis {cis} ({100 * cis / len(om):.1f}%), "
          f"mean |omega| {np.abs(om).mean():.1f} deg")

    e_dims = {g.edge_attr.shape[1] for g in graphs}
    n_contact = sum(int(g.edge_attr[:, 7].sum()) for g in graphs)
    n_bb = sum(int(g.edge_attr[:, 6].sum()) for g in graphs)
    print(f"\nedges: {n_bb} backbone + {n_contact} contact, feature dims {e_dims}")
    print(f"nodes: feature dim {{{graphs[0].x.shape[1]}}}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[1.15, 1])
    ax.scatter(phi[ok], psi[ok], s=3, alpha=0.25, edgecolors="none", c="#2b6cb0")
    ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel="phi (deg)",
           ylabel="psi (deg)", title=f"Boltz co-folds, {ok.sum()} residues")
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    ax.axhline(0, lw=0.5, c="0.7")
    ax.axvline(0, lw=0.5, c="0.7")
    for name, (p0, p1, s0, s1) in BASINS.items():
        ax.add_patch(plt.Rectangle((p0, s0), p1 - p0, s1 - s0, fill=False,
                                   ec="#dd6b20", lw=1, ls="--"))
        ax.text(p0 + 4, s1 - 14, name, color="#dd6b20", fontsize=8)

    ax2.hist(om, bins=90, color="#2b6cb0")
    ax2.set(xlabel="omega (deg)", ylabel="count", title="peptide bond planarity")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
