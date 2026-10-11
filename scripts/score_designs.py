"""Score the designed peptides with the PDB-trained inverse-folding model.

Asks, for a backbone and the sequence it was given: how much does this
geometry *want* that sequence? Mean log-probability per residue, all positions
masked at once, so nothing of the sequence is visible to the model.

Three things are measured, in this order of importance:

1. **The shuffle control.** A shuffled sequence on the same backbone must score
   clearly worse than the real sequence. If it does not, the score is
   measuring nothing and the lane comparison below is meaningless. This runs
   first and its verdict gates the rest.
2. **Lane comparison.** The same sequence across fold lanes -- Boltz co-fold,
   OpenFold3, RosettaFold3, ESMFold -- asks which lane produces geometry that
   wants the sequence it was given.
3. **A PDB reference band.** Held-out experimental peptides scored the same
   way, so "good" and "bad" have a scale. Without it a log-probability is just
   a number.

Fold-lane structures are scored, **not MD frames**: the Boltz co-folds match
experimental Ramachandran statistics closely (9.0% vs 8.4% outliers) while
300 K MD frames do not (16.6%), so scoring MD would confound fold quality with
thermal broadening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.inverse import InverseMPNN, sequence_log_prob  # noqa: E402
from pepgraph.residues import THREE_TO_ONE  # noqa: E402
from pepgraph.structure import read_structure  # noqa: E402

def coords_and_seq(path):
    res_list = read_structure(path, min_length=8, allow_noncanonical=True)
    if not res_list:
        return None
    seq = "".join(THREE_TO_ONE.get(r.name, "X") for r in res_list)
    coords = np.array([[r.coord("N"), r.coord("CA"), r.coord("C")]
                       for r in res_list], dtype=np.float64)
    return coords, seq


def score(model, coords, seq, cutoff, device="cpu"):
    mean_lp, per_res = sequence_log_prob(model, coords, seq,
                                         contact_cutoff=cutoff, device=device)
    return mean_lp, per_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/inverse/inverse.pt"))
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--npz", type=Path,
                    default=Path("outputs/pdb_peptides/peptides.npz"))
    ap.add_argument("--clusters", type=Path,
                    default=Path("outputs/pdb_peptides/clusters.json"))
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--n-reference", type=int, default=120)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "mps", "cuda"],
                    help="cpu by default; see train_inverse.py --device")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/inverse/design_scores.json"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["config"]
    model = InverseMPNN(hidden=int(cfg["hidden"]), n_layers=int(cfg["layers"]))
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    print(f"loaded {args.ckpt}")

    # ---- the designs, from every Boltz co-fold prediction
    designs = []
    for p in sorted(args.runs.glob(
            "*/boltz/boltz_results_*/predictions/*/*_model_0.cif")):
        got = coords_and_seq(p)
        if got:
            designs.append((p.parent.name, p.parents[4].name, *got))
    print(f"{len(designs)} Boltz co-fold structures")

    rng = np.random.default_rng(0)
    rows = []
    for name, lig, coords, seq in designs:
        real, _ = score(model, coords, seq, args.cutoff, args.device)
        shuf = []
        for _ in range(args.n_shuffles):
            s = list(seq)
            rng.shuffle(s)
            shuf.append(score(model, coords, "".join(s), args.cutoff, args.device)[0])
        rows.append({"name": name, "ligand": lig, "sequence": seq,
                     "n_res": len(seq), "real": real,
                     "shuffled_mean": float(np.mean(shuf)),
                     "shuffled_sd": float(np.std(shuf)),
                     "z": float((real - np.mean(shuf)) / (np.std(shuf) + 1e-9))})

    real = np.array([r["real"] for r in rows])
    sh = np.array([r["shuffled_mean"] for r in rows])
    wins = int((real > sh).sum())
    print("\n" + "=" * 70)
    print("1. SHUFFLE CONTROL  (gates everything below)")
    print("=" * 70)
    print(f"  real sequence      mean log P/res  {real.mean():+.3f}")
    print(f"  shuffled sequence  mean log P/res  {sh.mean():+.3f}")
    print(f"  real beats shuffled on {wins}/{len(rows)} structures "
          f"({100 * wins / len(rows):.0f}%)")
    print(f"  mean z of real vs its own shuffles: "
          f"{np.mean([r['z'] for r in rows]):+.2f}")
    passed = wins / len(rows) > 0.75 and real.mean() > sh.mean()
    print(f"  -> {'PASS' if passed else 'FAIL'}: "
          + ("the score distinguishes real sequence from shuffled, "
             "so lane comparison is meaningful"
             if passed else
             "the score does NOT distinguish real from shuffled. "
             "Everything below is meaningless; stop here."))

    # ---- PDB reference band, held-out clusters only
    z = np.load(args.npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    groups = json.loads(args.clusters.read_text())["groups"]
    test_groups = set(ck["test_groups"])
    ref = []
    for gi, g in enumerate(groups):
        if gi not in test_groups:
            continue
        for member in g["members"]:
            m = meta[member]
            c = z[f"c{m['index']}"][0]
            ref.append(score(model, c.astype(np.float64), m["sequence"],
                             args.cutoff, args.device)[0])
            if len(ref) >= args.n_reference:
                break
        if len(ref) >= args.n_reference:
            break
    ref = np.array(ref)
    print("\n" + "=" * 70)
    print("2. REFERENCE BAND  (held-out experimental peptides)")
    print("=" * 70)
    print(f"  {len(ref)} PDB peptides: mean {ref.mean():+.3f}, "
          f"sd {ref.std():.3f}, "
          f"10th-90th pct [{np.percentile(ref, 10):+.3f}, "
          f"{np.percentile(ref, 90):+.3f}]")
    print(f"  designs: mean {real.mean():+.3f} "
          f"-> {100 * (real > np.percentile(ref, 10)).mean():.0f}% of designs "
          f"score above the PDB 10th percentile")

    # ---- designs ranked
    print("\n" + "=" * 70)
    print("3. DESIGNS RANKED  (real vs own shuffles)")
    print("=" * 70)
    print(f"  {'structure':<30}{'real':>8}{'shuf':>8}{'z':>7}")
    for r in sorted(rows, key=lambda r: -r["z"])[:10]:
        print(f"  {r['name'][:29]:<30}{r['real']:+8.3f}"
              f"{r['shuffled_mean']:+8.3f}{r['z']:+7.1f}")
    print("  ...")
    for r in sorted(rows, key=lambda r: -r["z"])[-5:]:
        print(f"  {r['name'][:29]:<30}{r['real']:+8.3f}"
              f"{r['shuffled_mean']:+8.3f}{r['z']:+7.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"shuffle_control": {
                   "real_mean": float(real.mean()),
                   "shuffled_mean": float(sh.mean()),
                   "wins": wins, "n": len(rows), "passed": bool(passed)},
               "reference": {"n": len(ref), "mean": float(ref.mean()),
                             "sd": float(ref.std())},
               "designs": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
