"""Score one peptide structure: how much does this backbone want its sequence?

    score_structure.py STRUCTURE [STRUCTURE ...] [--sequence SEQ] [--suggest]

Input  : any .cif/.pdb/.cif.gz holding an 8-50 residue peptide chain. Only
         backbone N/CA/C is read, so side chains are not needed and a
         backbone-only model works. Optionally a replacement --sequence to
         score on that same backbone.
Output : mean log-probability per residue, where it falls against the
         experimental PDB band, a shuffled-sequence control for this very
         backbone, and with --suggest the model's preferred residue at each
         position.

What the number is: sequence-structure **self-consistency**. It is not binding
affinity, and it is not proof the fold is right -- scrambled sequences that are
folded produce equally self-consistent backbones (see docs/inverse-folding.md).
Use it to compare sequences on one backbone, or one sequence across fold lanes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.inverse import (InverseMPNN, make_example,  # noqa: E402
                              sequence_log_prob)
from pepgraph.residues import AA3, THREE_TO_ONE  # noqa: E402
from pepgraph.structure import read_structure  # noqa: E402

LETTERS = [THREE_TO_ONE[a] for a in AA3] + ["X"]


def load(path):
    res_list = read_structure(path, min_length=8, allow_noncanonical=True)
    if not res_list:
        return None
    seq = "".join(THREE_TO_ONE.get(r.name, "X") for r in res_list)
    coords = np.array([[r.coord("N"), r.coord("CA"), r.coord("C")]
                       for r in res_list], dtype=np.float64)
    return coords, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structures", nargs="+", type=Path)
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/inverse/inverse.pt"))
    ap.add_argument("--sequence", default=None,
                    help="score this sequence on the backbone instead of the "
                         "one in the file (same length required)")
    ap.add_argument("--suggest", action="store_true",
                    help="show the model's preferred residue per position")
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["config"]
    model = InverseMPNN(hidden=int(cfg["hidden"]), n_layers=int(cfg["layers"]))
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()

    # the reference band, if the earlier batch run left it behind
    band = None
    ref_path = Path("outputs/inverse/design_scores.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text()).get("reference")
        if ref:
            band = (ref["mean"], ref["sd"])

    rng = np.random.default_rng(0)
    out = []
    for path in args.structures:
        got = load(path)
        if got is None:
            print(f"{path}: no usable 8-50 residue peptide chain")
            continue
        coords, file_seq = got
        seq = args.sequence or file_seq
        if len(seq) != len(coords):
            print(f"{path}: --sequence is {len(seq)} residues but the backbone "
                  f"has {len(coords)}")
            continue

        mean_lp, per_res = sequence_log_prob(model, coords, seq,
                                             contact_cutoff=args.cutoff,
                                             device=args.device)
        shuf = []
        for _ in range(args.n_shuffles):
            s = list(seq)
            rng.shuffle(s)
            shuf.append(sequence_log_prob(model, coords, "".join(s),
                                          contact_cutoff=args.cutoff,
                                          device=args.device)[0])
        z = (mean_lp - np.mean(shuf)) / (np.std(shuf) + 1e-9)

        print(f"\n{path.name}")
        print(f"  sequence        {seq}")
        if args.sequence and args.sequence != file_seq:
            print(f"  (file sequence  {file_seq})")
        print(f"  residues        {len(seq)}")
        print(f"  mean log P/res  {mean_lp:+.3f}")
        print(f"  shuffled        {np.mean(shuf):+.3f} +- {np.std(shuf):.3f}"
              f"   -> z {z:+.1f}")
        if band:
            print(f"  PDB band        {band[0]:+.3f} +- {band[1]:.3f}"
                  f"   -> {(mean_lp - band[0]) / band[1]:+.1f} sd from "
                  f"experimental mean")
        verdict = ("consistent with experimental peptides" if band and
                   mean_lp > band[0] - 2 * band[1] else
                   "below the experimental band")
        print(f"  verdict         {verdict}; z>2 vs shuffles means the backbone "
              f"discriminates its sequence")

        if args.suggest:
            x, ei, ea, y, mask, sc = make_example(
                coords, seq, np.random.default_rng(0), rate=1.0,
                contact_cutoff=args.cutoff)
            with torch.no_grad():
                lp = F.log_softmax(
                    model(torch.from_numpy(x).to(args.device),
                          torch.from_numpy(ei).to(args.device),
                          torch.from_numpy(ea).to(args.device)), dim=-1).cpu()
            print(f"\n  {'pos':>4} {'is':>3} {'logP':>7}  top-3 preferred")
            for i in range(len(seq)):
                top = torch.topk(lp[i], 3)
                alts = " ".join(f"{LETTERS[j]}{float(v):+.1f}"
                                for v, j in zip(top.values, top.indices))
                flag = "" if LETTERS[int(lp[i].argmax())] == seq[i] else "  <-"
                print(f"  {i + 1:>4} {seq[i]:>3} {per_res[i]:+7.2f}  {alts}{flag}")
            agree = sum(1 for i in range(len(seq))
                        if LETTERS[int(lp[i].argmax())] == seq[i])
            print(f"\n  model's top choice matches the actual residue at "
                  f"{agree}/{len(seq)} positions ({100 * agree / len(seq):.0f}%)")

        out.append({"path": str(path), "sequence": seq, "n_res": len(seq),
                    "mean_log_p": mean_lp, "shuffled_mean": float(np.mean(shuf)),
                    "shuffled_sd": float(np.std(shuf)), "z": float(z),
                    "per_residue": per_res.tolist()})

    if args.json and out:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
