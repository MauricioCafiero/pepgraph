"""Plumbing check: push peptide graphs through the MPNN from ../../mpnn.

This proves the tensors line up and gradients flow, nothing more -- the weights
are random, so the outputs are meaningless. What it rules out is the class of
bug where shapes happen to be compatible but the wiring is wrong: edge indices
out of range after batching, messages scattered to the wrong nodes, pooling
mixing graphs together.

The MPNN's own ``collate`` is reused unchanged. PeptideGraph exposes the same
x / edge_index / edge_attr / y attributes as MolGraph, so it duck-types in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
DEFAULT_MPNN = HERE.parent / "mpnn" / "code"   # sibling repo, if present
sys.path.insert(0, str(HERE))

from pepgraph import (EDGE_DIM, NODE_DIM, find_trajectories,  # noqa: E402
                      graph_from_structure, iter_trajectory_graphs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True,
                    help="directory holding the simulation/fold runs")
    ap.add_argument("--mpnn-repo", type=Path, default=DEFAULT_MPNN,
                    help="path to the sibling mpnn repo's code/ dir")
    ap.add_argument("--n-cofold", type=int, default=6)
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()

    mpnn_repo = args.mpnn_repo
    if not mpnn_repo.exists():
        print(f"mpnn repo not found at {mpnn_repo}; pass --mpnn-repo")
        return 1
    sys.path.insert(0, str(mpnn_repo))
    from molgraph.mpnn import MPNN, collate  # noqa: E402

    # a deliberately mixed batch: predicted structures and MD frames together,
    # with different peptide lengths, since that is what would expose a
    # batching bug
    # pick co-folds of *different* lengths on purpose: a batch of equal-length
    # graphs would not exercise the node-offset arithmetic at all
    graphs, seen_lengths = [], set()
    for p in sorted(args.runs.glob(
            "*/boltz/boltz_results_*/predictions/*/*_model_0.cif")):
        g = graph_from_structure(p, name=p.parent.name)
        if g is None or g.n_residues in seen_lengths:
            continue
        seen_lengths.add(g.n_residues)
        g.y = 0.0
        graphs.append(g)
        if len(graphs) >= args.n_cofold:
            break

    xtc, top = find_trajectories(args.runs)[0]
    for _, g in iter_trajectory_graphs(xtc, top, stride=500,
                                       max_frames=args.n_frames):
        g.y = 0.0
        graphs.append(g)

    print(f"batch of {len(graphs)} graphs, "
          f"lengths {sorted({g.n_residues for g in graphs})}")

    x, edge_index, edge_attr, batch, y = collate(graphs)
    n_nodes = x.shape[0]
    print(f"\ncollated: x {tuple(x.shape)}  edge_index {tuple(edge_index.shape)}  "
          f"edge_attr {tuple(edge_attr.shape)}  batch {tuple(batch.shape)}")

    # --- batching invariants
    assert x.shape[1] == NODE_DIM, x.shape
    assert edge_attr.shape[1] == EDGE_DIM, edge_attr.shape
    assert n_nodes == sum(g.n_residues for g in graphs)
    assert int(batch.max()) + 1 == len(graphs)
    assert int(edge_index.max()) < n_nodes, "edge index out of range after offset"
    assert int(edge_index.min()) >= 0
    # every edge must stay inside one graph: no edge may join two graphs
    cross = (batch[edge_index[0]] != batch[edge_index[1]]).sum()
    assert cross == 0, f"{cross} edges join different graphs"
    counts = torch.bincount(batch, minlength=len(graphs))
    assert counts.tolist() == [g.n_residues for g in graphs]
    print("batching invariants: node counts, edge offsets and graph isolation all OK")

    model = MPNN(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=64, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    out = model(x, edge_index, edge_attr, batch)
    print(f"\nMPNN({NODE_DIM} node / {EDGE_DIM} edge dims, hidden 64, 3 layers): "
          f"{n_params:,} params")
    print(f"output {tuple(out.shape)} -> one scalar per graph, "
          f"finite={bool(torch.isfinite(out).all())}")
    assert out.shape == (len(graphs),)

    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    grads = {n: float(p.grad.abs().max()) for n, p in model.named_parameters()
             if p.grad is not None}
    dead = [n for n, v in grads.items() if v == 0.0]
    print(f"\nbackward: loss {float(loss.detach()):.4f}, "
          f"{len(grads)}/{n_params and len(list(model.parameters()))} tensors have grads, "
          f"max |grad| {max(grads.values()):.3e}")
    print(f"parameters receiving zero gradient: {len(dead)}"
          + (f" -> {dead[:4]}" if dead else " (none, so every layer is reached)"))

    # permutation invariance: reordering the graphs must permute the outputs,
    # not change them. This is the check that catches pooling leaks.
    model.eval()
    with torch.no_grad():
        order = list(reversed(range(len(graphs))))
        xb, eib, eab, bb, _ = collate([graphs[i] for i in order])
        out2 = model(xb, eib, eab, bb)
        base = model(x, edge_index, edge_attr, batch)
    diff = float((base[torch.tensor(order)] - out2).abs().max())
    print(f"\nreordering the batch changes outputs by {diff:.2e} "
          f"({'invariant' if diff < 1e-4 else 'NOT INVARIANT - pooling leaks'})")

    # a graph alone vs in a batch must give the same answer
    with torch.no_grad():
        xs, eis, eas, bs, _ = collate([graphs[0]])
        alone = float(model(xs, eis, eas, bs)[0])
    print(f"graph 0 alone {alone:.6f} vs in batch {float(base[0]):.6f} "
          f"-> diff {abs(alone - float(base[0])):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
