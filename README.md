# pepgraph

Residue-level graph neural networks for short peptides, with the baselines that
tell you whether the graph is doing any work.

Nodes are residues, edges are peptide bonds and spatial contacts. The package
builds graphs from any PDB or mmCIF structure, trains an inverse-folding model
on PDB peptides, and uses it to **rank structure-prediction pipelines** by how
well the backbones they produce fit the sequences they were given.

It also contains two honest negative results, which is most of why it exists.

## What it is for

**Ranking fold pipelines.** Given several predicted structures of the same
peptide, which pipeline produced geometry that actually wants that sequence?
Scoring takes seconds per structure, against a dock-and-simulate cycle to find
out any other way.

**Flagging failed predictions.** Each score comes with a `z` against the same
backbone's own shuffled sequences. `z ≤ 0` means the structure carries no
sequence information — in practice, an extended or disordered chain — and the
tool says so without you inspecting the geometry.

**Comparing candidate sequences on one backbone.** Swap the sequence, keep the
geometry.

It is **not** a binding-affinity predictor and it **cannot validate a design**:
scrambling a sequence and then folding it yields a structure that scores just as
well, so a good score means the backbone is self-consistent, not that the
peptide is good. See [docs/inverse-folding.md](docs/inverse-folding.md).

## Results in brief

| target | verdict |
|---|---|
| **Inverse folding** | message passing adds **18–30%** recovery over a local-geometry-only baseline |
| **Fold-pipeline ranking** | co-folded with ligand beats apo; one pipeline failed on 7/21 peptides, emitting extended chains |
| Angle denoising | graph adds only **1–4%** over a geometry-free Ramachandran prior |
| Intra-peptide contact prediction | deterministic from torsions (F1 = 0.801 analytically); nothing to learn |

Two of those reversed sign once a proper baseline existed — angle denoising
looked like a 21% win until a geometry-free Bayes estimator showed 19 of those
points came from the Ramachandran distribution alone. Full numbers and the
baselines behind them are in [docs/](#documentation).

## Install

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy matplotlib mdtraj torch joblib \
    pandas scikit-learn
```

Only `numpy` is needed to build graphs. `mdtraj` is imported lazily and only by
the trajectory/mining modules; `torch` only by the models. `scripts/forward_pass.py`
additionally needs `rdkit`, because it imports a sibling repo's package for an
interoperability check — install it only if you run that script.

## Quick start

Score a peptide structure — does this backbone want its sequence?

```sh
./.venv/bin/python scripts/score_structure.py path/to/fold.pdb --suggest
```

```
fold.pdb
  residues        33
  mean log P/res  -2.409
  shuffled        -3.120 +- 0.211   -> z +3.4
  PDB band        -2.600 +- 0.349   -> +0.5 sd from experimental mean
```

Build the training corpus from the PDB (3,824 entries → 52,423 conformers,
~30 min, 532 MB) and train:

```sh
./.venv/bin/python scripts/fetch_pdb_peptides.py --raw raw_pdb --out outputs/pdb_peptides
./.venv/bin/python scripts/train_inverse.py
```

Scripts that read a project's own simulation or fold outputs take an explicit
`--runs` directory; there are no hidden defaults or environment variables.
`scripts/forward_pass.py` similarly takes `--mpnn-repo`.

## Graph representation

**Node features (32):** one-hot residue identity (20 canonical + "other");
one-hot N-terminal / middle / C-terminal; charge at pH 7, residue mass,
aromatic, Kyte–Doolittle hydropathy, side-chain volume, H-bond donors and
acceptors; pLDDT when the source carries it.

**Edge features (10):** sin/cos of ψ, ω and φ; backbone and contact flags;
direction; CA–CA distance. Both directions of every peptide bond, plus contact
edges for residue pairs more than 3 apart in sequence and within 8 Å.

Two decisions worth knowing:

- **The torsions describe the bond, not the residue.** The three torsions
  spanning the link from residue *i* to *i*+1 are ψᵢ, ωᵢ and φᵢ₊₁, so that
  triplet is what the backbone edge carries.
- **Angles are stored as sin/cos**, so −179° and +179° are adjacent rather than
  maximally distant. This also gives a free missing-value encoding: an undefined
  angle is sin = cos = 0, which no real angle can produce.

Details in [docs/features.md](docs/features.md).

## Layout

```
pepgraph/
├── pepgraph/
│   ├── residues.py     # amino-acid property tables
│   ├── structure.py    # mmCIF / PDB / .gz backbone reader, numpy only
│   ├── graph.py        # torsions, node/edge features, PeptideGraph
│   ├── kinematics.py   # torsions <-> coordinates (NeRF), noise injection
│   ├── trajectory.py   # MD frames -> graphs
│   ├── mining.py       # frame selection and coordinate shards
│   ├── denoise.py      # angle denoiser + geometry-free shrinkage baseline
│   └── inverse.py      # inverse folding + local-geometry-only baseline
├── scripts/            # entry points
└── docs/               # methods, results, validation, design decisions
```

## Documentation

| doc | contents |
|---|---|
| [docs/features.md](docs/features.md) | node and edge features, and why |
| [docs/inverse-folding.md](docs/inverse-folding.md) | corpus, training, inference, pipeline comparison |
| [docs/angle-denoising.md](docs/angle-denoising.md) | formulation, baselines, why the graph adds little |
| [docs/kinematics.md](docs/kinematics.md) | NeRF rebuild, round-trip test, lever-arm measurement |
| [docs/validation.md](docs/validation.md) | geometry checks and batching invariants |
| [docs/data.md](docs/data.md) | corpus construction, deduplication, domain gap |
| [docs/decisions.md](docs/decisions.md) | **approaches rejected, and why** |

## Validation

The featuriser is checked before anything is trained on it, because a wrong
atom ordering still produces well-shaped tensors and trains without complaint.

| check | outcome |
|---|---|
| Torsions vs `mdtraj.compute_phi/psi/omega` | agree to **6×10⁻⁵ degrees** |
| Backbone rebuild from measured internal coordinates | **8×10⁻¹⁴ Å** RMSD |
| Coordinate shard round-trip vs direct read | **0.000e+00** on all tensors |
| Batched graphs: edges crossing between graphs | **0** |
| ω in a predicted-structure set | 100% trans, mean 177.6° |
| α-L basin occupancy | 88% glycine |

## A note on method

Every result here is reported against a baseline that does the same job without
the graph — a geometry-free Bayes estimator for angle denoising, a
local-geometry-only classifier for inverse folding. That convention is not
decoration: it is what turned an apparent 21% improvement into a real 4% one,
and what repeatedly showed where a graph was not earning its parameters.

An absolute number for a graph model means very little on its own.

## Licence

MIT. See [LICENSE](LICENSE).
