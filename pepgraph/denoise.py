"""Angle denoising: recover clean phi/psi from a corrupted backbone.

An example is made by perturbing a frame's phi/psi, rebuilding the backbone
from the corrupted torsions, and featurising *that*. Nothing derived from the
clean coordinates survives into the input -- in particular the CA-CA contact
distances come from the rebuilt structure, so they cannot leak the answer.

The task is measured in torsion space, where it is exactly recoverable, rather
than in coordinate space where lever-arm amplification along the chain
dominates.

**The model predicts a correction, not an angle.** Predicting the angle itself
makes the identity baseline -- return the input unchanged -- something the model
has to learn to imitate to within a degree, and it starts arbitrarily far from
it; the first version of this did exactly that and settled into predicting a
flat prior that ignored the input. Predicting ``clean - noisy`` instead puts the
identity baseline at *zero output*, so an untrained model is already as good as
the baseline and training can only improve on it. This is the same reason
diffusion models regress the noise rather than the signal.

The noisy angles are therefore given to each node directly, as sin/cos, along
with the noise level, so a correction can be read off locally and refined by
message passing.

The message passing mirrors ``molgraph.mpnn`` from the sibling mpnn repo; the
difference is the head, which is per-node here instead of pooling to one
scalar per graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import EDGE_DIM, NODE_DIM, graph_from_residues
from .kinematics import build_backbone, perturb_torsions, torsion_chain
from .residues import THREE_TO_ONE
from .structure import Residue

SIGMA_SCALE = 15.0      # the noise schedule saturates well before this
# node features, plus the noise level, plus sin/cos of the noisy phi and psi
IN_DIM = NODE_DIM + 1 + 4


def _residue_list(names, coords):
    return [
        Residue(name=names[r], seq_id=r + 1,
                atoms={"N": coords[r, 0], "CA": coords[r, 1], "C": coords[r, 2]})
        for r in range(len(names))
    ]


def sincos(deg):
    rad = np.radians(deg)
    return np.sin(rad), np.cos(rad)


@dataclass
class Frame:
    """A clean frame, kept as torsions so examples are cheap to regenerate."""
    names: list
    sequence: str
    phi: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    source: str


def frames_from_shard(path, limit=None):
    """Clean torsions per frame, without building any graphs."""
    import json

    from .graph import backbone_torsions

    with np.load(path, allow_pickle=False) as z:
        coords = z["coords"]
        meta = json.loads(str(z["meta"]))
    three = {v: k for k, v in THREE_TO_ONE.items()}
    names = [three.get(c, "UNK") for c in meta["sequence"]]

    n = len(coords) if limit is None else min(limit, len(coords))
    out = []
    for i in range(n):
        phi, psi, omega = backbone_torsions(_residue_list(names, coords[i]))
        out.append(Frame(names=names, sequence=meta["sequence"], phi=phi,
                         psi=psi, omega=omega,
                         source=f"{meta['run']}:{meta['stage']}"))
    return out


def wrap(deg):
    """Wrap degrees into (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def make_example(frame: Frame, sigma: float, rng, contact_cutoff: float = 8.0):
    """(x, edge_index, edge_attr, target, mask, noisy_phi, noisy_psi).

    ``target`` is the correction ``clean - noisy`` for phi and psi, in units of
    SIGMA_SCALE degrees. Zero output therefore reproduces the identity baseline.
    """
    phi_n, psi_n, omega_n = perturb_torsions(frame.phi, frame.psi, sigma, rng,
                                             frame.omega)
    coords = build_backbone(torsion_chain(phi_n, psi_n, omega_n),
                            n_residues=len(frame.names))
    g = graph_from_residues(_residue_list(frame.names, coords), frame.source,
                            contact_cutoff, use_plddt=False)

    mask = (~np.isnan(frame.phi) & ~np.isnan(frame.psi)).astype(np.float32)
    target = np.stack([wrap(frame.phi - phi_n), wrap(frame.psi - psi_n)],
                      axis=1) / SIGMA_SCALE
    target = np.nan_to_num(target).astype(np.float32)

    # the noisy angles go in as sin/cos, with the noise level alongside
    sp, cp = sincos(np.nan_to_num(phi_n))
    ss, cs = sincos(np.nan_to_num(psi_n))
    extra = np.stack(
        [sp, cp, ss, cs,
         np.full(len(frame.names), sigma / SIGMA_SCALE)], axis=1)
    x = np.concatenate([g.x, extra.astype(np.float32)], axis=1)
    return x, g.edge_index, g.edge_attr, target, mask, phi_n, psi_n


def collate(examples):
    xs, eis, eas, ts, ms, offset = [], [], [], [], [], 0
    for x, ei, ea, t, m, *_ in examples:
        xs.append(torch.from_numpy(x))
        if ei.shape[1]:
            eis.append(torch.from_numpy(ei) + offset)
            eas.append(torch.from_numpy(ea))
        ts.append(torch.from_numpy(t))
        ms.append(torch.from_numpy(m))
        offset += x.shape[0]
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas),
            torch.cat(ts), torch.cat(ms))


class DenoiseMPNN(nn.Module):
    def __init__(self, node_dim: int = IN_DIM, edge_dim: int = EDGE_DIM,
                 hidden: int = 128, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
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
        # per-node head: the phi and psi corrections, in SIGMA_SCALE degrees.
        # The last layer starts at zero so an untrained model *is* the identity
        # baseline rather than something arbitrarily worse.
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_attr):
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        src, dst = edge_index[0], edge_index[1]
        for layer in range(self.n_layers):
            msg = self.edge_mlps[layer](torch.cat([h[src], h[dst], e], dim=-1))
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            upd = self.node_mlps[layer](torch.cat([h, agg], dim=-1))
            h = self.norms[layer](h + self.dropout(upd))
        return self.head(h)


def angular_error(pred, target, mask):
    """Mean residual |error| in degrees for phi and psi, over masked nodes.

    Both ``pred`` and ``target`` are corrections, so the error of the denoised
    angle is just how far the predicted correction is from the true one.
    """
    sel = mask > 0
    if sel.sum() == 0:
        return float("nan"), float("nan")
    d = (pred[sel] - target[sel]).abs() * SIGMA_SCALE
    return float(d[:, 0].mean()), float(d[:, 1].mean())


def masked_mse(pred, target, mask):
    w = mask.unsqueeze(1)
    return ((pred - target) ** 2 * w).sum() / (w.sum() * 2).clamp(min=1)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def identity_baseline(examples):
    """Return the corrupted input unchanged: a correction of zero.

    Its error is therefore the size of the true correction, which is the noise
    that was injected.
    """
    errs = []
    for _, _, _, target, mask, _, _ in examples:
        sel = mask > 0
        errs.append(np.abs(target[sel]) * SIGMA_SCALE)
    errs = np.concatenate(errs)
    return float(errs[:, 0].mean()), float(errs[:, 1].mean())


def fit_type_means(frames):
    """Circular mean phi/psi per residue letter, from the training frames."""
    acc = {}
    for f in frames:
        for i, letter in enumerate(f.sequence):
            if np.isnan(f.phi[i]) or np.isnan(f.psi[i]):
                continue
            s = acc.setdefault(letter, [0.0, 0.0, 0.0, 0.0])
            sp, cp = sincos(f.phi[i])
            ss, cs = sincos(f.psi[i])
            s[0] += sp; s[1] += cp; s[2] += ss; s[3] += cs
    return {k: (np.degrees(np.arctan2(v[0], v[1])),
                np.degrees(np.arctan2(v[2], v[3]))) for k, v in acc.items()}


class ShrinkagePrior:
    """Bayes-optimal denoising from the residue identity alone -- no geometry.

    This is the baseline that decides whether the graph earns its place. Some
    error reduction is free: phi/psi are not uniformly distributed, so pulling
    any noisy angle toward the alpha-R basin helps on average without knowing
    anything about the structure. To attribute a gain to message passing, the
    model has to beat an estimator that gets this for free.

    It is given every advantage short of geometry: the full **joint** 2D
    Ramachandran prior per residue type (not separate phi and psi marginals),
    the true noise level, and the estimator that minimises mean absolute
    angular error rather than squared error, since MAE is what is reported.
    """

    def __init__(self, n_bins: int = 72, blur_bins: float = 2.0,
                 min_count: int = 200):
        self.n_bins = n_bins
        self.blur_bins = blur_bins
        self.min_count = min_count
        self.width = 360.0 / n_bins
        self.centers = -180.0 + self.width * (np.arange(n_bins) + 0.5)
        # |wrapped difference| between every pair of bin centres
        d = np.abs(self.centers[:, None] - self.centers[None, :])
        self.dist = np.minimum(d, 360.0 - d)
        self.hists = {}
        self.pooled = None

    def _bin(self, deg):
        return np.clip(((deg + 180.0) / self.width).astype(int), 0,
                       self.n_bins - 1)

    def _blur(self, h):
        """Circular Gaussian smoothing, separable over the two axes."""
        if self.blur_bins <= 0:
            return h
        k = int(np.ceil(3 * self.blur_bins))
        offs = np.arange(-k, k + 1)
        w = np.exp(-0.5 * (offs / self.blur_bins) ** 2)
        w /= w.sum()
        for axis in (0, 1):
            acc = np.zeros_like(h)
            for o, wi in zip(offs, w):
                acc += wi * np.roll(h, int(o), axis=axis)
            h = acc
        return h

    def fit(self, frames):
        totals = np.zeros((self.n_bins, self.n_bins))
        per_letter = {}
        for f in frames:
            ok = ~np.isnan(f.phi) & ~np.isnan(f.psi)
            bi = self._bin(f.phi[ok])
            bj = self._bin(f.psi[ok])
            letters = np.array(list(f.sequence))[ok]
            for letter in np.unique(letters):
                sel = letters == letter
                h = per_letter.setdefault(
                    letter, np.zeros((self.n_bins, self.n_bins)))
                np.add.at(h, (bi[sel], bj[sel]), 1.0)
            np.add.at(totals, (bi, bj), 1.0)

        self.pooled = self._normalise(totals)
        for letter, h in per_letter.items():
            if h.sum() >= self.min_count:
                self.hists[letter] = self._normalise(h)
            else:
                # too few observations to trust: fall back to the pooled prior
                self.hists[letter] = self.pooled
        return self

    def _normalise(self, h):
        h = self._blur(h)
        h = h + 1e-6 * h.sum() / h.size     # keep every bin reachable
        return h / h.sum()

    def predict(self, letter, phi_noisy, psi_noisy, sigma):
        prior = self.hists.get(letter, self.pooled)
        s = max(sigma, 1e-3)
        dphi = np.minimum(np.abs(self.centers - phi_noisy),
                          360.0 - np.abs(self.centers - phi_noisy))
        dpsi = np.minimum(np.abs(self.centers - psi_noisy),
                          360.0 - np.abs(self.centers - psi_noisy))
        like = (np.exp(-0.5 * (dphi / s) ** 2)[:, None]
                * np.exp(-0.5 * (dpsi / s) ** 2)[None, :])
        post = prior * like
        total = post.sum()
        if total <= 0:
            return phi_noisy, psi_noisy
        post /= total
        # the angle minimising expected |wrapped error| under each marginal
        phi_hat = self.centers[np.argmin(self.dist @ post.sum(axis=1))]
        psi_hat = self.centers[np.argmin(self.dist @ post.sum(axis=0))]
        return float(phi_hat), float(psi_hat)


def shrinkage_baseline(examples, sequences, prior: ShrinkagePrior, sigma):
    """Error of the identity-only Bayes estimator on the same examples."""
    errs_phi, errs_psi = [], []
    for (_, _, _, target, mask, phi_n, psi_n), seq in zip(examples, sequences):
        sel = np.nonzero(mask > 0)[0]
        clean_phi = phi_n + target[:, 0] * SIGMA_SCALE
        clean_psi = psi_n + target[:, 1] * SIGMA_SCALE
        for i in sel:
            ph, ps = prior.predict(seq[i], phi_n[i], psi_n[i], sigma)
            errs_phi.append(abs(wrap(ph - clean_phi[i])))
            errs_psi.append(abs(wrap(ps - clean_psi[i])))
    return float(np.mean(errs_phi)), float(np.mean(errs_psi))


def type_mean_baseline(frames, means, default=(-70.0, -30.0)):
    """Predict each residue's angles from its identity alone, ignoring geometry."""
    errs_phi, errs_psi = [], []
    for f in frames:
        for i, letter in enumerate(f.sequence):
            if np.isnan(f.phi[i]) or np.isnan(f.psi[i]):
                continue
            mp, ms = means.get(letter, default)
            errs_phi.append(abs((mp - f.phi[i] + 180.0) % 360.0 - 180.0))
            errs_psi.append(abs((ms - f.psi[i] + 180.0) % 360.0 - 180.0))
    return float(np.mean(errs_phi)), float(np.mean(errs_psi))
