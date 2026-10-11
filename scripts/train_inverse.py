"""Train inverse folding on PDB peptides, against three baselines.

Splits are by **cluster group**, from clusters.json (RCSB 30% identity, then
single-linkage merged). Splitting by sequence or conformer would put point
mutants and NMR models of the same peptide on both sides.

The designed peptides are never in training; they are scored separately by
score_designs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.inverse import (IN_DIM, InverseMPNN, LocalOnly,  # noqa: E402
                              N_CLASSES, collate, make_example, scores)
from pepgraph.residues import AA3, identity_index  # noqa: E402


def load_corpus(npz, clusters):
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    groups = json.loads(Path(clusters).read_text())["groups"]
    items = []           # (group_id, sequence, coords_array)
    for gi, g in enumerate(groups):
        for member in g["members"]:
            m = meta[member]
            items.append((gi, m["sequence"], z[f"c{m['index']}"]))
    return items


def split_groups(n_groups, rng, frac_val=0.07, frac_test=0.07):
    order = rng.permutation(n_groups)
    n_v, n_t = int(frac_val * n_groups), int(frac_test * n_groups)
    return set(order[n_v + n_t:].tolist()), set(order[:n_v].tolist()), \
        set(order[n_v:n_v + n_t].tolist())


def sample_pool(items, keep, rng, per_group, max_items):
    """One or a few conformers per sequence, so NMR ensembles do not dominate."""
    pool = []
    for gi, seq, coords in items:
        if gi not in keep:
            continue
        idx = rng.permutation(len(coords))[:per_group]
        for k in idx:
            pool.append((gi, seq, coords[k]))
    rng.shuffle(pool)
    return pool[:max_items]


def pick_device(choice: str) -> str:
    """Resolve --device. CPU is the default on purpose: peptide graphs are small
    enough that MPS launch overhead exceeds the arithmetic (measured 46.5 vs
    34.5 ms/step), so MPS only pays off on much larger graphs or batches."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, local, pool, rates, rng, cutoff, batch_size, freq_class,
             device="cpu"):
    model.eval()
    local.eval()
    rows = {}
    for rate in rates:
        ex = [make_example(c, s, rng, rate, cutoff) for _, s, c in pool]
        agg = {k: [0.0, 0] for k in ("model", "local", "freq", "pos")}
        with torch.no_grad():
            for i in range(0, len(ex), batch_size):
                x, ei, ea, y, m, sc = collate(ex[i:i + batch_size])
                n = int(m.sum())
                if n == 0:
                    continue
                x, ei, ea = x.to(device), ei.to(device), ea.to(device)
                y, m, sc = y.to(device), m.to(device), sc.to(device)
                for name, logits in (
                        ("model", model(x, ei, ea)),
                        ("local", local(sc))):
                    rec, ppl = scores(logits, y, m)
                    agg[name][0] += rec * n
                    agg[name][1] += n
                # frequency baseline
                sel = m.bool()
                agg["freq"][0] += float((y[sel] == freq_class).float().sum())
                agg["freq"][1] += n
        rows[rate] = {k: (v[0] / v[1] if v[1] else float("nan"))
                      for k, v in agg.items() if v[1]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    default=Path("outputs/pdb_peptides/peptides.npz"))
    ap.add_argument("--clusters", type=Path,
                    default=Path("outputs/pdb_peptides/clusters.json"))
    ap.add_argument("--per-group", type=int, default=3,
                    help="conformers sampled per sequence per epoch")
    ap.add_argument("--train-items", type=int, default=6000)
    ap.add_argument("--eval-items", type=int, default=600)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.15, 0.5, 1.0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "mps", "cuda", "auto"],
                    help="cpu by default: measured faster than mps on these "
                         "small graphs (34.5 vs 46.5 ms/step). Use mps/auto "
                         "if graphs or batches grow much larger.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/inverse"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    items = load_corpus(args.npz, args.clusters)
    n_groups = max(gi for gi, _, _ in items) + 1
    print(f"{len(items)} sequences in {n_groups} cluster groups")

    tr, va, te = split_groups(n_groups, rng)
    print(f"split by group: {len(tr)} train / {len(va)} val / {len(te)} test")

    train = sample_pool(items, tr, rng, args.per_group, args.train_items)
    val = sample_pool(items, va, rng, 1, args.eval_items)
    test = sample_pool(items, te, rng, 1, args.eval_items)
    print(f"conformers: {len(train)} train, {len(val)} val, {len(test)} test")

    # frequency baseline, fit on training sequences only
    counts = Counter(c for _, s, _ in train for c in s)
    freq_letter = counts.most_common(1)[0][0]
    from pepgraph.residues import THREE_TO_ONE
    freq_class = identity_index({v: k for k, v in THREE_TO_ONE.items()}
                                .get(freq_letter, "UNK"))
    total = sum(counts.values())
    print(f"\nmost common training residue: {freq_letter} "
          f"({100 * counts[freq_letter] / total:.1f}% of residues)")
    print(f"composition: " + " ".join(
        f"{k}{100 * v / total:.0f}" for k, v in counts.most_common(8)))

    device = pick_device(args.device)
    print(f"device: {device}")
    model = InverseMPNN(hidden=args.hidden, n_layers=args.layers).to(device)
    local = LocalOnly(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    opt_l = torch.optim.Adam(local.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"\nInverseMPNN {sum(p.numel() for p in model.parameters()):,} params"
          f"  |  LocalOnly {sum(p.numel() for p in local.parameters()):,} params")
    print(f"mask rates {args.rates}\n")

    best, best_state, t0 = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        pool = sample_pool(items, tr, rng, args.per_group, args.train_items)
        model.train()
        local.train()
        tot = totl = nb = 0.0
        for i in range(0, len(pool), args.batch_size):
            batch = pool[i:i + args.batch_size]
            rate = float(rng.choice(args.rates))
            ex = [make_example(c, s, rng, rate, args.cutoff)
                  for _, s, c in batch]
            x, ei, ea, y, m, sc = collate(ex)
            if int(m.sum()) == 0:
                continue
            x, ei, ea = x.to(device), ei.to(device), ea.to(device)
            y, m, sc = y.to(device), m.to(device), sc.to(device)
            sel = m.bool()
            loss = F.cross_entropy(model(x, ei, ea)[sel], y[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            lossl = F.cross_entropy(local(sc)[sel], y[sel])
            opt_l.zero_grad(); lossl.backward(); opt_l.step()
            tot += float(loss.detach()); totl += float(lossl.detach()); nb += 1

        rows = evaluate(model, local, val, args.rates, rng, args.cutoff,
                        args.batch_size, freq_class, device)
        mean_rec = float(np.mean([r["model"] for r in rows.values()]))
        flag = ""
        if mean_rec > best:
            best, flag = mean_rec, "  *"
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            best_local = {k: v.detach().clone()
                          for k, v in local.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {tot / nb:.3f} (local {totl / nb:.3f})  "
              f"val recovery {100 * mean_rec:5.1f}%{flag}")

    if best_state:
        model.load_state_dict(best_state)
        local.load_state_dict(best_local)
    print(f"\ntrained in {time.time() - t0:.0f}s, best val recovery "
          f"{100 * best:.1f}%")

    rows = evaluate(model, local, test, args.rates, rng, args.cutoff,
                    args.batch_size, freq_class, device)
    print("\n" + "=" * 68)
    print(f"{'TEST (held-out clusters)':<26}{'graph':>9}{'local':>9}"
          f"{'freq':>9}{'graph gain':>13}")
    print("=" * 68)
    for rate, r in rows.items():
        gain = 100 * (r["model"] - r["local"]) / max(r["local"], 1e-9)
        print(f"  mask {100 * rate:3.0f}% of residues  "
              f"{100 * r['model']:8.1f}%{100 * r['local']:8.1f}%"
              f"{100 * r['freq']:8.1f}%{gain:+12.1f}%")
    print("-" * 68)
    print("'graph gain' is recovery relative to the local-geometry-only model,")
    print("i.e. what message passing adds over a residue's own phi/psi+burial.")

    torch.save({"model": model.state_dict(), "local": local.state_dict(),
                "config": vars(args) | {k: str(v) for k, v in vars(args).items()
                                        if isinstance(v, Path)},
                "freq_class": int(freq_class), "freq_letter": freq_letter,
                "test_groups": sorted(te), "val_groups": sorted(va)},
               args.out / "inverse.pt")
    json.dump({str(k): v for k, v in rows.items()},
              open(args.out / "metrics.json", "w"), indent=2)
    print(f"\nwrote {args.out / 'inverse.pt'} and {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
