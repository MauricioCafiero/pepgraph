"""Inverse folding: mask residue identities, keep the geometry, predict them back.

The one task in this project where the graph should be load-bearing -- the
geometry is the input and the sequence is the answer, so it cannot be solved
from a residue's own identity the way angle denoising could.

**The leak to avoid.** A node's features carry both the identity one-hot *and*
the physicochemical properties (charge, mass, hydropathy, volume, H-bond
counts). Those properties determine the identity exactly for the 20 canonical
residues, so masking the one-hot alone would leave the answer sitting in the
input. :func:`mask_nodes` zeroes both, and sets a flag so the model knows which
positions are hidden rather than genuinely uncharged and massless.

Three baselines, because an absolute recovery rate means nothing on its own
(see docs/decisions.md):

* ``frequency``   -- always the most common residue in training.
* ``position``    -- identity from N-term/middle/C-term and relative position.
* ``local``       -- that residue's **own** geometry only: phi/psi sin/cos plus
  a burial count, with no message passing. This is the baseline that decides
  whether the graph earns its place; it is the analogue of the shrinkage prior
  that cut angle denoising from an apparent 21% to a real 1-4%.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import EDGE_DIM, graph_from_residues
from .residues import (AA3, N_IDENTITY, N_PHYSCHEM, N_TERMINUS, THREE_TO_ONE,
                       identity_index)
from .structure import Residue

THREE = {v: k for k, v in THREE_TO_ONE.items()}
N_CLASSES = N_IDENTITY            # 20 canonical + "other"
IDENTITY_SLICE = slice(0, N_IDENTITY)
PHYS_SLICE = slice(N_IDENTITY + N_TERMINUS, N_IDENTITY + N_TERMINUS + N_PHYSCHEM)
# node features as the graph builds them, plus a "this position is masked" flag
IN_DIM = N_IDENTITY + N_TERMINUS + N_PHYSCHEM + 1 + 1
BURIAL_RADIUS = 10.0


def graph_from_coords(coords, sequence, contact_cutoff=8.0):
    """[n_res, 3, 3] of N/CA/C plus a one-letter sequence -> PeptideGraph."""
    names = [THREE.get(c, "UNK") for c in sequence]
    res_list = [
        Residue(name=names[i], seq_id=i + 1,
                atoms={"N": coords[i, 0], "CA": coords[i, 1], "C": coords[i, 2]})
        for i in range(len(sequence))
    ]
    return graph_from_residues(res_list, "x", contact_cutoff, use_plddt=False)


def burial(coords, radius=BURIAL_RADIUS):
    """Neighbour count within ``radius`` of each CA -- a crude burial proxy."""
    ca = coords[:, 1]
    d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    return ((d < radius).sum(axis=1) - 1).astype(np.float32)


def mask_nodes(x, rng, rate):
    """Hide identity at a random subset of positions. Returns (x', mask).

    Zeroes the identity one-hot **and** the physicochemical block, then appends
    the masked flag. Without zeroing the physicochemical block the identity is
    still fully determined and the task is trivial.
    """
    n = x.shape[0]
    k = max(1, int(round(rate * n)))
    idx = rng.choice(n, size=min(k, n), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    out = x.copy()
    out[mask, IDENTITY_SLICE] = 0.0
    out[mask, PHYS_SLICE] = 0.0
    out = np.concatenate([out, mask[:, None].astype(np.float32)], axis=1)
    return out, mask


def make_example(coords, sequence, rng, rate, contact_cutoff=8.0):
    g = graph_from_coords(coords, sequence, contact_cutoff)
    x, mask = mask_nodes(g.x, rng, rate)
    y = np.array([identity_index(THREE.get(c, "UNK")) for c in sequence],
                 dtype=np.int64)
    # local-only features, for the baseline that sees no neighbours
    phi, psi = g.phi, g.psi
    sc = np.stack([np.sin(np.radians(np.nan_to_num(phi))),
                   np.cos(np.radians(np.nan_to_num(phi))),
                   np.sin(np.radians(np.nan_to_num(psi))),
                   np.cos(np.radians(np.nan_to_num(psi))),
                   (~np.isnan(phi)).astype(np.float32),
                   burial(coords) / 20.0,
                   np.arange(len(sequence)) / max(len(sequence) - 1, 1)],
                  axis=1).astype(np.float32)
    return x, g.edge_index, g.edge_attr, y, mask, sc


def collate(examples):
    xs, eis, eas, ys, ms, scs, off = [], [], [], [], [], [], 0
    for x, ei, ea, y, m, sc in examples:
        xs.append(torch.from_numpy(x))
        if ei.shape[1]:
            eis.append(torch.from_numpy(ei) + off)
            eas.append(torch.from_numpy(ea))
        ys.append(torch.from_numpy(y))
        ms.append(torch.from_numpy(m))
        scs.append(torch.from_numpy(sc))
        off += x.shape[0]
    return (torch.cat(xs),
            torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long),
            torch.cat(eas) if eas else torch.zeros(0, EDGE_DIM),
            torch.cat(ys), torch.cat(ms), torch.cat(scs))


class InverseMPNN(nn.Module):
    """Same message passing as the denoiser; a 21-way classifier per node."""

    def __init__(self, node_dim: int = IN_DIM, edge_dim: int = EDGE_DIM,
                 hidden: int = 128, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.node_emb = nn.Linear(node_dim, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.node_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, N_CLASSES))

    def forward(self, x, edge_index, edge_attr):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        src, dst = edge_index[0], edge_index[1]
        for layer in range(self.n_layers):
            if edge_index.shape[1]:
                msg = self.edge_mlps[layer](torch.cat([h[src], h[dst], e], -1))
                agg = torch.zeros_like(h)
                agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            else:
                agg = torch.zeros_like(h)
            upd = self.node_mlps[layer](torch.cat([h, agg], -1))
            h = self.norms[layer](h + self.dropout(upd))
        return self.head(h)


class LocalOnly(nn.Module):
    """The baseline with no message passing: one residue's own geometry."""

    def __init__(self, in_dim: int = 7, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_CLASSES))

    def forward(self, sc):
        return self.net(sc)


def scores(logits, y, mask):
    """(recovery, perplexity) over masked positions."""
    sel = mask.bool()
    if sel.sum() == 0:
        return float("nan"), float("nan")
    lg, tg = logits[sel], y[sel]
    rec = float((lg.argmax(-1) == tg).float().mean())
    ppl = float(torch.exp(F.cross_entropy(lg, tg)))
    return rec, ppl


def sequence_log_prob(model, coords, sequence, contact_cutoff=8.0,
                      device="cpu"):
    """Mean log-probability of ``sequence`` on this backbone, all positions
    masked at once.

    This is the fold-quality score: how much a backbone "wants" the sequence it
    was given. Comparable across fold lanes for one sequence, and against a
    shuffled sequence on the same backbone.
    """
    x, ei, ea, y, mask, sc = make_example(
        coords, sequence, np.random.default_rng(0), rate=1.0,
        contact_cutoff=contact_cutoff)
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device),
                       torch.from_numpy(ei).to(device),
                       torch.from_numpy(ea).to(device))
        lp = F.log_softmax(logits, dim=-1)
        tgt = torch.from_numpy(y).to(device)
        per_res = lp.gather(1, tgt[:, None]).squeeze(1)
    return float(per_res.mean()), per_res.cpu().numpy()
