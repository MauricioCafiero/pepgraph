"""Score every peptide on every fold lane, and compare the lanes.

Reports **perplexity** (exp of the mean negative log-probability per residue --
the effective number of residue types the model is choosing between, where 21
is no information) and **z** (the same backbone with its sequence shuffled 20
times, so z isolates whether the backbone cares about residue *order* rather
than composition).

The headline comparison is **paired**: only sequences present in every lane are
used, so lane differences cannot come from one lane holding easier peptides.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.graph import CONTACT_MIN_SEP, graph_from_residues  # noqa: E402
from pepgraph.inverse import InverseMPNN, sequence_log_prob  # noqa: E402
from pepgraph.residues import THREE_TO_ONE  # noqa: E402
from pepgraph.structure import read_structure  # noqa: E402

LANES = ["esmfold", "of3apo", "openfold3", "rf3", "rf3apo"]
UNIFORM = 21.0


def geometry(res_list):
    g = graph_from_residues(res_list, "x", use_plddt=False)
    ca = np.array([r.coord("CA") for r in res_list])
    d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    sep = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])
    ok = ~np.isnan(g.phi) & ~np.isnan(g.psi)
    aR = float(((g.phi[ok] >= -160) & (g.phi[ok] <= -20)
                & (g.psi[ok] >= -120) & (g.psi[ok] <= 50)).mean()) if ok.any() else 0.0
    return {"rg": float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())),
            "contacts": int(((d < 8) & (sep > CONTACT_MIN_SEP)).sum() // 2),
            "end_end": float(np.linalg.norm(ca[-1] - ca[0])),
            "frac_aR": aR}


def load(path):
    res_list = read_structure(path, min_length=8, allow_noncanonical=True)
    if not res_list:
        return None
    seq = "".join(THREE_TO_ONE.get(r.name, "X") for r in res_list)
    coords = np.array([[r.coord("N"), r.coord("CA"), r.coord("C")]
                       for r in res_list], dtype=np.float64)
    return coords, seq, res_list


def collect(runs):
    """(lane, design_name, ligand, path) for every fold-lane structure."""
    found = []
    # each immediate subdirectory of --runs is one target/system
    systems = sorted(d.name for d in runs.iterdir()
                     if d.is_dir() and (d / "folds").is_dir() or
                     (d.is_dir() and (d / "boltz").is_dir()))
    for lig in systems:
        for lane in LANES:
            d = runs / lig / "folds" / lane
            if not d.is_dir():
                continue
            for p in sorted(list(d.glob("*.pdb")) + list(d.glob("*.cif"))):
                found.append((lane, p.stem, lig, p))
        for p in sorted((runs / lig / "boltz").glob(
                "boltz_results_*/predictions/*/*_model_0.cif")):
            found.append(("boltz", p.parent.name, lig, p))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/inverse/inverse.pt"))
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/inverse/lane_comparison.json"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["config"]
    model = InverseMPNN(hidden=int(cfg["hidden"]), n_layers=int(cfg["layers"]))
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()

    band = None
    ref = Path("outputs/inverse/design_scores.json")
    if ref.exists():
        r = json.loads(ref.read_text()).get("reference")
        if r:
            band = float(np.exp(-r["mean"]))

    found = collect(args.runs)
    print(f"{len(found)} structures across {len(set(l for l, *_ in found))} lanes")

    rng = np.random.default_rng(0)
    rows = []
    for i, (lane, name, lig, path) in enumerate(found, 1):
        got = load(path)
        if got is None:
            print(f"  skip {lane}/{name}: no usable chain")
            continue
        coords, seq, res_list = got
        lp, _ = sequence_log_prob(model, coords, seq,
                                  contact_cutoff=args.cutoff, device=args.device)
        shuf = []
        for _ in range(args.n_shuffles):
            s = list(seq)
            rng.shuffle(s)
            shuf.append(sequence_log_prob(model, coords, "".join(s),
                                          contact_cutoff=args.cutoff,
                                          device=args.device)[0])
        rows.append({"lane": lane, "design": name, "ligand": lig,
                     "sequence": seq, "n_res": len(seq),
                     "perplexity": float(np.exp(-lp)),
                     "shuffled_perplexity": float(np.exp(-np.mean(shuf))),
                     "z": float((lp - np.mean(shuf)) / (np.std(shuf) + 1e-9)),
                     **geometry(res_list)})
        if i % 25 == 0:
            print(f"  {i}/{len(found)}")

    # ---------- per lane, all structures
    by_lane = defaultdict(list)
    for r in rows:
        by_lane[r["lane"]].append(r)
    print("\n" + "=" * 84)
    print("ALL STRUCTURES BY LANE")
    print("=" * 84)
    print(f"{'lane':<12}{'n':>4}{'perplexity':>19}{'z':>14}"
          f"{'contacts':>10}{'frac aR':>9}{'z<=0':>7}")
    print(f"{'':<12}{'':>4}{'median (mean)':>19}{'median (mean)':>14}")
    print("-" * 84)
    for lane in sorted(by_lane, key=lambda k: np.median(
            [r["perplexity"] for r in by_lane[k]])):
        v = by_lane[lane]
        px = np.array([r["perplexity"] for r in v])
        z = np.array([r["z"] for r in v])
        print(f"{lane:<12}{len(v):>4}{np.median(px):>10.1f} ({px.mean():5.1f})"
              f"{np.median(z):>8.1f} ({z.mean():+4.1f})"
              f"{np.median([r['contacts'] for r in v]):>10.0f}"
              f"{np.median([r['frac_aR'] for r in v]):>9.2f}"
              f"{sum(1 for x in z if x <= 0):>7}")
    print("-" * 84)
    print(f"{'uniform':<12}{'':>4}{UNIFORM:>10.1f}{'':>8}"
          f"  (21 = no information)")
    if band:
        print(f"{'PDB held-out':<12}{'':>4}{band:>10.1f}")

    # ---------- paired: sequences present in every fold lane
    seq_lanes = defaultdict(dict)
    for r in rows:
        seq_lanes[r["sequence"]].setdefault(r["lane"], []).append(r)
    complete = [s for s, d in seq_lanes.items() if all(l in d for l in LANES)]
    print("\n" + "=" * 84)
    print(f"PAIRED COMPARISON -- {len(complete)} sequences present in all "
          f"{len(LANES)} fold lanes")
    print("=" * 84)
    if complete:
        print(f"{'lane':<12}{'median px':>11}{'mean px':>10}{'mean z':>9}"
              f"{'best-of-lane wins':>19}")
        print("-" * 84)
        wins = defaultdict(int)
        for s in complete:
            best = min(LANES, key=lambda l: np.mean(
                [r["perplexity"] for r in seq_lanes[s][l]]))
            wins[best] += 1
        stats = []
        for lane in LANES:
            px = np.array([np.mean([r["perplexity"] for r in seq_lanes[s][lane]])
                           for s in complete])
            z = np.array([np.mean([r["z"] for r in seq_lanes[s][lane]])
                          for s in complete])
            stats.append((np.median(px), lane, px, z))
        for med, lane, px, z in sorted(stats):
            print(f"{lane:<12}{med:>11.1f}{px.mean():>10.1f}{z.mean():>+9.1f}"
                  f"{wins[lane]:>19}")

        # matched apo vs holo pairs
        print("\nmatched apo vs ligand-bound pairs (paired, same sequences):")
        for holo, apo in (("openfold3", "of3apo"), ("rf3", "rf3apo")):
            a = np.array([np.mean([r["perplexity"] for r in seq_lanes[s][holo]])
                          for s in complete])
            b = np.array([np.mean([r["perplexity"] for r in seq_lanes[s][apo]])
                          for s in complete])
            d = b - a                       # positive = holo better
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"  {holo:<10} {a.mean():5.1f}  vs  {apo:<8} {b.mean():5.1f}"
                  f"   diff {d.mean():+5.1f} +- {se:.1f}"
                  f"  ({d.mean() / se:+.1f} sigma, "
                  f"{int((d > 0).sum())}/{len(d)} sequences favour {holo})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"rows": rows, "complete_sequences": complete,
         "uniform_perplexity": UNIFORM, "pdb_perplexity": band}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
