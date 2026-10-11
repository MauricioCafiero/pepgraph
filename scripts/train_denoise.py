"""Train the angle denoiser, against the baselines that matter.

The split is by **sequence**, not by frame. Frames of one peptide are
correlated to the point of near-duplication, so a frame-level split would put
near-copies of the training data in validation and report a score that means
nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.denoise import (DenoiseMPNN, ShrinkagePrior,  # noqa: E402
                              angular_error, collate, fit_type_means,
                              frames_from_shard, identity_baseline,
                              make_example, masked_mse, shrinkage_baseline,
                              type_mean_baseline)


def load_frames(shard_dir, per_shard):
    by_seq = defaultdict(list)
    for path in sorted(Path(shard_dir).glob("*.npz")):
        for f in frames_from_shard(path, limit=per_shard):
            by_seq[f.sequence].append(f)
    return by_seq


def split_sequences(sequences, n_val, n_test, seed):
    rng = np.random.default_rng(seed)
    order = list(sequences)
    rng.shuffle(order)
    return (order[n_val + n_test:], order[:n_val], order[n_val:n_val + n_test])


def build_epoch(frames, sigmas, rng, cutoff):
    out = []
    for f in frames:
        sigma = float(rng.choice(sigmas))
        out.append(make_example(f, sigma, rng, cutoff))
    return out


def evaluate(model, frames, sigmas, rng, cutoff, batch_size, prior=None):
    """Per-sigma angular error, so performance is visible across the schedule.

    When ``prior`` is given, the geometry-free shrinkage baseline is scored on
    the *same* examples, with the same noise draws, so the comparison is exact
    rather than statistical.
    """
    model.eval()
    rows = {}
    for sigma in sigmas:
        examples = [make_example(f, sigma, rng, cutoff) for f in frames]
        errs, ident = [], identity_baseline(examples)
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                x, ei, ea, t, m = collate(examples[i:i + batch_size])
                pred = model(x, ei, ea)
                errs.append((angular_error(pred, t, m), int(m.sum())))
        wp = sum(e[0][0] * e[1] for e in errs) / sum(e[1] for e in errs)
        ws = sum(e[0][1] * e[1] for e in errs) / sum(e[1] for e in errs)
        rows[sigma] = {"model_phi": wp, "model_psi": ws,
                       "identity_phi": ident[0], "identity_psi": ident[1]}
        if prior is not None:
            sp, ss = shrinkage_baseline(
                examples, [f.sequence for f in frames], prior, sigma)
            rows[sigma]["shrinkage_phi"] = sp
            rows[sigma]["shrinkage_psi"] = ss
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("outputs/frames"))
    ap.add_argument("--per-shard", type=int, default=120,
                    help="frames taken from each shard")
    ap.add_argument("--train-frames", type=int, default=2500)
    ap.add_argument("--eval-frames", type=int, default=250)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[1.0, 2.0, 5.0, 10.0, 15.0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/denoise_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("loading frames...")
    by_seq = load_frames(args.shards, args.per_shard)
    print(f"  {len(by_seq)} sequences, "
          f"{sum(len(v) for v in by_seq.values()):,} frames")

    tr_seq, va_seq, te_seq = split_sequences(by_seq, args.n_val, args.n_test,
                                             args.seed)
    print(f"\nsplit by sequence: {len(tr_seq)} train / {len(va_seq)} val / "
          f"{len(te_seq)} test")
    for label, group in (("val", va_seq), ("test", te_seq)):
        for s in group:
            print(f"  {label:4s} {s}")

    def pool(seqs, cap):
        frames = [f for s in seqs for f in by_seq[s]]
        idx = rng.permutation(len(frames))[:cap]
        return [frames[i] for i in idx]

    train = pool(tr_seq, args.train_frames)
    val = pool(va_seq, args.eval_frames)
    test = pool(te_seq, args.eval_frames)
    print(f"\nframes: {len(train)} train, {len(val)} val, {len(test)} test")

    # --- baseline that uses no geometry at all
    means = fit_type_means([f for s in tr_seq for f in by_seq[s]])
    tm_val = type_mean_baseline(val, means)
    tm_test = type_mean_baseline(test, means)
    print(f"\nper-residue-type mean baseline (identity only, no geometry):")
    print(f"  val  phi {tm_val[0]:6.1f} deg   psi {tm_val[1]:6.1f} deg")
    print(f"  test phi {tm_test[0]:6.1f} deg   psi {tm_test[1]:6.1f} deg")

    print("\nfitting the geometry-free shrinkage prior on the training frames...")
    prior = ShrinkagePrior().fit([f for s in tr_seq for f in by_seq[s]])
    n_specific = sum(1 for v in prior.hists.values() if v is not prior.pooled)
    print(f"  joint 2D Ramachandran prior: {n_specific} residue types with "
          f"their own histogram, {len(prior.hists) - n_specific} pooled")

    model = DenoiseMPNN(hidden=args.hidden, n_layers=args.layers)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"\nDenoiseMPNN hidden {args.hidden}, {args.layers} layers, "
          f"{n_par:,} params")
    print(f"noise schedule sigma = {args.sigmas} deg\n")

    best, best_state, t0 = float("inf"), None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        examples = build_epoch(train, args.sigmas, rng, args.cutoff)
        order = rng.permutation(len(examples))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            batch = [examples[j] for j in order[i:i + args.batch_size]]
            x, ei, ea, t, m = collate(batch)
            pred = model(x, ei, ea)
            loss = masked_mse(pred, t, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1

        rows = evaluate(model, val, args.sigmas, rng, args.cutoff,
                        args.batch_size)
        mean_err = np.mean([r["model_phi"] + r["model_psi"]
                            for r in rows.values()]) / 2
        flag = ""
        if mean_err < best:
            best, flag = mean_err, "  *"
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {tot / nb:.4f}  "
              f"val mean |err| {mean_err:6.2f} deg{flag}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\ntrained in {time.time() - t0:.0f}s, best val {best:.2f} deg")

    print("\n" + "=" * 78)
    print(f"{'TEST (held-out seqs)':<14}{'model phi':>11}{'shrink phi':>10}"
          f"{'ident phi':>10}   {'model psi':>9}{'shrink psi':>10}"
          f"{'ident psi':>10}")
    print("=" * 78)
    rows = evaluate(model, test, args.sigmas, rng, args.cutoff,
                    args.batch_size, prior=prior)
    for sigma, r in rows.items():
        print(f"  sigma {sigma:5.1f}  "
              f"{r['model_phi']:>9.2f} {r['shrinkage_phi']:>9.2f} "
              f"{r['identity_phi']:>9.2f}   "
              f"{r['model_psi']:>9.2f} {r['shrinkage_psi']:>9.2f} "
              f"{r['identity_psi']:>9.2f}")
    print("-" * 78)
    print(f"  per-residue-type mean (no geometry, no noisy angle): "
          f"phi {tm_test[0]:.1f}, psi {tm_test[1]:.1f}")
    print()
    print("  model vs shrinkage -- does message passing beat identity-only Bayes?")
    for sigma, r in rows.items():
        dp = 100 * (r['shrinkage_phi'] - r['model_phi']) / r['shrinkage_phi']
        ds = 100 * (r['shrinkage_psi'] - r['model_psi']) / r['shrinkage_psi']
        verdict = "model" if (dp + ds) / 2 > 0 else "shrinkage"
        print(f"    sigma {sigma:5.1f}: phi {dp:+6.1f}%  psi {ds:+6.1f}%  "
              f"-> {verdict} wins")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"test": {str(k): v for k, v in rows.items()},
                   "type_mean_baseline": {"val": tm_val, "test": tm_test},
                   "val_sequences": va_seq, "test_sequences": te_seq,
                   "config": vars(args) | {"shards": str(args.shards),
                                           "out": str(args.out)}},
                  fh, indent=2, default=str)
    ckpt = args.out.with_suffix(".pt")
    torch.save({"state_dict": model.state_dict(), "config": vars(args) | {
        "shards": str(args.shards), "out": str(args.out)}}, ckpt)
    print(f"\nwrote {args.out} and {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
