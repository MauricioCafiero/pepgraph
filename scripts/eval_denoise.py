"""Re-evaluate a saved denoiser against the baselines, without retraining.

Exists mainly to vary the shrinkage prior's resolution. At 5 deg bins the
estimator returns bin centres, which puts a floor of about 1.25 deg on its
error and makes it look *worse* than the identity baseline at low noise -- an
artifact of the discretisation, not a property of the estimator. Finer bins
remove it, so the comparison at low sigma means something.
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

from pepgraph.denoise import (DenoiseMPNN, ShrinkagePrior,  # noqa: E402
                              angular_error, collate, frames_from_shard,
                              identity_baseline, make_example,
                              shrinkage_baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("outputs/denoise_metrics.pt"))
    ap.add_argument("--shards", type=Path, default=Path("outputs/frames"))
    ap.add_argument("--prior-bins", type=int, nargs="+", default=[72, 360])
    ap.add_argument("--eval-frames", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("outputs/denoise_eval.json"))
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    cfg = ckpt["config"]
    print(f"loaded {args.ckpt}: hidden {cfg['hidden']}, {cfg['layers']} layers")

    rng = np.random.default_rng(cfg["seed"])
    by_seq = defaultdict(list)
    for path in sorted(args.shards.glob("*.npz")):
        for f in frames_from_shard(path, limit=cfg["per_shard"]):
            by_seq[f.sequence].append(f)

    # reproduce the original split exactly
    srng = np.random.default_rng(cfg["seed"])
    order = list(by_seq)
    srng.shuffle(order)
    n_val, n_test = cfg["n_val"], cfg["n_test"]
    tr_seq = order[n_val + n_test:]
    te_seq = order[n_val:n_val + n_test]

    def pool(seqs, cap):
        frames = [f for s in seqs for f in by_seq[s]]
        idx = rng.permutation(len(frames))[:cap]
        return [frames[i] for i in idx]

    pool(tr_seq, cfg["train_frames"])          # consume rng as training did
    pool(order[:n_val], args.eval_frames)
    test = pool(te_seq, args.eval_frames)
    print(f"test: {len(te_seq)} sequences, {len(test)} frames")

    model = DenoiseMPNN(hidden=cfg["hidden"], n_layers=cfg["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sigmas = cfg["sigmas"]
    train_frames = [f for s in tr_seq for f in by_seq[s]]
    priors = {}
    for n_bins in args.prior_bins:
        blur = 2.0 * n_bins / 72.0      # keep the physical smoothing constant
        print(f"fitting shrinkage prior: {n_bins} bins "
              f"({360 / n_bins:.2f} deg), blur {blur:.1f} bins")
        priors[n_bins] = ShrinkagePrior(n_bins=n_bins, blur_bins=blur).fit(
            train_frames)

    results = {}
    for sigma in sigmas:
        examples = [make_example(f, sigma, rng, cfg["cutoff"]) for f in test]
        ident = identity_baseline(examples)
        errs = []
        with torch.no_grad():
            for i in range(0, len(examples), cfg["batch_size"]):
                x, ei, ea, t, m = collate(examples[i:i + cfg["batch_size"]])
                errs.append((angular_error(model(x, ei, ea), t, m), int(m.sum())))
        tot = sum(e[1] for e in errs)
        row = {"model_phi": sum(e[0][0] * e[1] for e in errs) / tot,
               "model_psi": sum(e[0][1] * e[1] for e in errs) / tot,
               "identity_phi": ident[0], "identity_psi": ident[1]}
        seqs = [f.sequence for f in test]
        for n_bins, prior in priors.items():
            sp, ss = shrinkage_baseline(examples, seqs, prior, sigma)
            row[f"shrink{n_bins}_phi"] = sp
            row[f"shrink{n_bins}_psi"] = ss
        results[sigma] = row
        print(f"  sigma {sigma:5.1f} done")

    bins = args.prior_bins
    head = (f"{'sigma':>6} {'identity':>9} "
            + " ".join(f"{'shrink/' + str(b):>11}" for b in bins)
            + f" {'model':>9} {'graph gain':>11}")
    print("\n" + "=" * len(head))
    print("PHI" + " " * (len(head) - 3))
    print(head)
    print("=" * len(head))
    for metric in ("phi", "psi"):
        if metric == "psi":
            print("\n" + "=" * len(head))
            print("PSI")
            print(head)
            print("=" * len(head))
        for sigma, r in results.items():
            best = min(r[f"shrink{b}_{metric}"] for b in bins)
            gain = 100 * (best - r[f"model_{metric}"]) / best
            print(f"{sigma:>6.1f} {r[f'identity_{metric}']:>9.2f} "
                  + " ".join(f"{r[f'shrink{b}_{metric}']:>11.2f}" for b in bins)
                  + f" {r[f'model_{metric}']:>9.2f} {gain:>10.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"results": {str(k): v for k, v in results.items()},
                   "prior_bins": bins, "test_sequences": te_seq},
                  fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\n'graph gain' is the model's improvement over the BEST shrinkage "
          "prior,\ni.e. what message passing adds once the Ramachandran prior "
          "is already free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
