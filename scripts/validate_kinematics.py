"""Does rebuilding a backbone from its own torsions give the backbone back?

Two separate tests, because they answer different questions.

1. **Exact.** Rebuild from the *measured* bond lengths, bond angles and
   torsions. This should reproduce the structure to floating-point precision.
   It tests the kinematics alone -- if this fails, the frame construction or the
   torsion-to-atom mapping is wrong and nothing downstream is trustworthy.

2. **Ideal geometry.** Rebuild from measured *torsions* but ideal bond lengths
   and angles. This cannot be exact, and the error it leaves is the price of
   using ideal geometry. That number matters directly: it is the noise floor of
   a denoising task, so any corruption smaller than it is invisible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph import backbone_torsions, graph_from_structure  # noqa: E402
from pepgraph.kinematics import (build_backbone, internal_coords,  # noqa: E402
                                 kabsch_rmsd, perturb_torsions, torsion_chain)
from pepgraph.mining import load_shard  # noqa: E402
from pepgraph.structure import read_structure  # noqa: E402

def coords_of(res_list):
    return np.array([[r.coord("N"), r.coord("CA"), r.coord("C")]
                     for r in res_list])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--shards", type=Path, default=Path("outputs/frames"))
    ap.add_argument("--n-cofold", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=40)
    args = ap.parse_args()

    cases = []
    for p in sorted(args.runs.glob(
            "*/boltz/boltz_results_*/predictions/*/*_model_0.cif"))[:args.n_cofold]:
        res_list = read_structure(p)
        if res_list:
            cases.append(("cofold", coords_of(res_list)))

    shard = sorted(args.shards.glob("*.npz"))
    if shard:
        _, graphs = load_shard(shard[0], limit=args.n_frames)
        with np.load(shard[0], allow_pickle=False) as z:
            for c in z["coords"][:args.n_frames]:
                cases.append(("md", c.astype(np.float64)))

    print(f"{len(cases)} structures "
          f"({sum(1 for k, _ in cases if k == 'cofold')} co-folds, "
          f"{sum(1 for k, _ in cases if k == 'md')} MD frames)\n")

    exact, ideal = [], []
    for _, coords in cases:
        bonds, angles, torsions = internal_coords(coords)

        rebuilt = build_backbone(torsions, bonds, angles,
                                 n_residues=coords.shape[0])
        exact.append(kabsch_rmsd(rebuilt, coords))

        # ideal geometry, measured torsions. Go through phi/psi/omega rather
        # than the measured chain, so this follows the path denoising will use.
        res_list = [type("R", (), {
            "coord": (lambda self, a, _c=c: dict(zip(("N", "CA", "C"), _c))[a])
        })() for c in coords]
        phi, psi, omega = backbone_torsions(res_list)
        rebuilt_ideal = build_backbone(torsion_chain(phi, psi, omega),
                                       n_residues=coords.shape[0])
        ideal.append(kabsch_rmsd(rebuilt_ideal, coords))

    exact, ideal = np.array(exact), np.array(ideal)
    print("1. measured bonds + angles + torsions  (tests the kinematics)")
    print(f"   RMSD  max {exact.max():.2e} A   mean {exact.mean():.2e} A")
    verdict = "PASS" if exact.max() < 1e-6 else "FAIL"
    print(f"   -> {verdict}: "
          + ("exact to floating point, the kinematics are right"
             if verdict == "PASS" else
             "NOT exact; frame construction or torsion mapping is wrong"))

    print("\n2. ideal bonds + angles, measured torsions  (the denoising path)")
    print(f"   RMSD  max {ideal.max():.2f} A   mean {ideal.mean():.2f} A   "
          f"median {np.median(ideal):.2f} A")
    print("   -> this is the noise floor: corruption below it is invisible")

    # how much backbone movement a given torsion perturbation actually causes,
    # which is what sets a sensible noise schedule
    print("\n3. RMSD caused by perturbing phi/psi (ideal geometry, 40 samples)")
    rng = np.random.default_rng(0)
    coords = cases[0][1]
    res_list = [type("R", (), {
        "coord": (lambda self, a, _c=c: dict(zip(("N", "CA", "C"), _c))[a])
    })() for c in coords]
    phi, psi, omega = backbone_torsions(res_list)
    base = build_backbone(torsion_chain(phi, psi, omega),
                          n_residues=coords.shape[0])
    print(f"   {'sigma (deg)':>12} {'RMSD vs clean rebuild (A)':>28}")
    for sigma in (1, 2, 5, 10, 20, 45):
        vals = []
        for _ in range(40):
            p2, s2, o2 = perturb_torsions(phi, psi, sigma, rng, omega)
            pert = build_backbone(torsion_chain(p2, s2, o2),
                                  n_residues=coords.shape[0])
            vals.append(kabsch_rmsd(pert, base))
        print(f"   {sigma:>12} {np.mean(vals):>21.2f} +- {np.std(vals):.2f}")
    return 0 if exact.max() < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
