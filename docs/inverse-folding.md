# Inverse folding on PDB peptides, with the designs as hold-out

The task: mask residue identities, keep the geometry, predict the identities
back. Unlike a sequence-only language model this genuinely needs the graph —
the geometry is the input and the sequence is the answer. It was rejected
earlier for having only 21 sequences; the PDB supplies ~2,250 clusters of
isolated short peptides, which removes that objection.

## Data

`fetch_pdb_peptides.py`. "Isolated" means the whole deposited entry is 8–50
residues, so no large partner chain — absent from the graph — is dictating the
conformation. That confound is exactly the kind that produced this project's
earlier negative results.

```
3,824 isolated peptide entries        ~2,246 clusters at 30% identity
  84% solution NMR  (1,935 clusters)  -> 10-20 models each, an experimental ensemble
  14% X-ray         (327 clusters)
```

A resolution filter would delete the entire NMR set, since NMR entries carry no
resolution. Do not add one.

Non-canonical residues are kept when the backbone is complete and map to the
one-hot's "other" slot; short peptide structures frequently contain a D-amino
acid or a modified residue, and dropping a whole peptide for one of them would
discard much of the set.

## What was actually fetched (2026-10-11)

```
3,824/3,824 entries downloaded, 0 failed        raw_pdb/, 532 MB
52,423 (entry, model) conformers from 3,621 entries
2,956 distinct sequences                        peptides.npz, 42.1 MB
length 8-49 (median 29)
dropped: 1,138 too many non-canonical, 252 no complete backbone,
         203 entries contributed nothing
non-canonical: 13.3% of conformers contain >=1, 1.3% of residues
```

**2,956 sequences against the project's 21** — 141×. The 13.3% figure is why
non-canonical residues are kept rather than filtered: strict canonical-only
would discard about a seventh of the set over 1.3% of residues.

### No leakage, but the naive check misfires

Zero exact matches and zero substring containments against the 21 designs. A
plain identity threshold nevertheless flags
an alanine-rich 33-residue design at 69% against a 13-residue
poly-alanine PDB peptide — **low-complexity similarity, not homology**: two
poly-alanine sequences match at high identity by construction, and normalising
by the shorter length inflates it further. Do not treat a sequence-identity
cutoff as a leakage test on homopolymer-like designs.

It does reveal something real: the Ala-rich and Gly-rich designs sit in a
region of sequence space the PDB samples with short homopolymer-ish peptides,
so those three may behave oddly for reasons unrelated to fold quality.

### The domain gap, measured

| | alpha-R | beta/PPII | alpha-L | outlier | mean abs(omega) |
|---|---|---|---|---|---|
| PDB isolated (experimental) | 51.7% | **34.9%** | 5.0% | **8.4%** | **177.6°** |
| Designs, Boltz/MD frames | 56.3% | **21.6%** | 5.5% | **16.6%** | **171.8°** |
| Designs, Boltz co-folds only | 60.4% | 24.4% | 6.3% | **9.0%** | **177.6°** |

Two effects, and they must not be conflated:

**The outlier and omega gap is MD thermal broadening, not fold quality.** The
Boltz co-folds match experimental geometry almost exactly (9.0% vs 8.4%
outliers, 177.6° vs 177.6°). It is the 300 K frames that deviate. **So score
the fold-lane structures, not MD frames** — otherwise fold quality is
confounded with thermal motion and the folds will look worse than they are.

**The beta/PPII gap is genuine**: 34.9% vs 21.6%, not explained by MD. PDB
short peptides carry far more extended and PPII content (collagen-like,
polyproline) than these Gly-rich, helix-biased designs. A model trained on the
PDB will feel this, and it is a property of the design set rather than a defect.

## Why the hold-out design is the interesting part

The designed peptides are **not in training**. Two separate held-out sets,
because they answer different questions:

1. **Held-out PDB clusters** — measures the model. Same distribution as
   training, so a poor score here means the model is weak.
2. **The 21 designed peptides, in their Boltz/MD structures** — measures the
   *structures*. A different distribution (predicted and simulated, not
   experimental), so a drop here is the domain gap plus whatever is wrong with
   the folds.

Keeping both means model quality and fold quality can be told apart. With only
the second, every bad score is ambiguous.

## The part that makes it a fold-quality probe

If a sequence–structure compatibility model says a backbone does not want its
own sequence, that is evidence about the **fold**, not the model. Two things in
this project turn that into a real measurement:

**Five fold lanes for the same sequences.** ESMFold apo, OpenFold3 ± ligand,
RosettaFold3 ± ligand, and Boltz co-fold. Scoring the same sequence on each
lane's backbone asks which lane produces geometry that actually wants the
sequence it was given — a comparison that currently costs a dock-and-simulate
cycle per lane.

**Shuffle controls as a built-in negative.** The `shuffle_control` /
`ox2_shuffle` runs hold shuffled sequences. A shuffled sequence on a real
backbone should score clearly worse than the real sequence on that backbone.
**If it does not, the model is measuring nothing** and no lane comparison from
it means anything. Run this check first; it is the cheapest possible way to
invalidate the whole experiment.

## Result (2026-10-11)

Trained on 1,957 cluster groups, 6,000 conformers/epoch, 16 epochs, 264 s CPU.
Recovery on **held-out clusters**:

| mask rate | graph | local-only | frequency | graph gain |
|---|---|---|---|---|
| 15% | **23.8%** | 18.3% | 8.6% | **+29.9%** |
| 50% | **24.1%** | 18.6% | 9.8% | **+29.5%** |
| 100% | **22.0%** | 18.6% | 9.0% | **+18.1%** |

**This is the first target where the graph clearly earns its place** — 18–30%
over local-only, against 1–4% for angle denoising. The local baseline's loss
flatlined at ~2.66 from epoch 5 while the graph kept falling 2.75 → 2.33, so it
is information-limited rather than capacity-limited, which is what licenses
attributing the gap to neighbour context.

Absolute recovery of 22–24% is modest (ProteinMPNN reaches 40–50% on proteins),
but short peptides give far less context and this is a 487k-parameter model
trained for four minutes. Validation plateaued around 26% by epoch 14–16 while
training loss kept dropping — mild overfitting; more conformers would help more
than more epochs.

Corpus composition note: **cysteine is the most common residue at 9%** (against
~1.5% in typical proteins) because short PDB peptides are disulfide-rich
conotoxins and defensins. Recovery numbers should be read with that in mind.

## Scoring the designs (`score_designs.py`)

### The shuffle control passes

```
real sequence      -2.698 log P/res
shuffled sequence  -3.120
real beats shuffled on 74/76 structures (97%),  mean z +2.75   -> PASS
```

### The designs sit inside the experimental band

```
120 held-out PDB peptides: mean -2.600, sd 0.349, 10th-90th [-2.984, -2.119]
76 Boltz co-folds:         mean -2.698
89% of designs score above the PDB 10th percentile
```

So the designed backbones want their sequences about as much as experimental
peptide structures do.

### But scrambled-sequence folds score just as well — and that is the limit

| group | n | mean log P/res |
|---|---|---|
| `orig` conditioned | 23 | **-2.569** |
| `shuffle_control` (scrambled, then folded) | 4 | -2.573 |
| `esm1` conditioned | 24 | -2.703 |
| `esm2` conditioned | 24 | -2.819 |

**Two different shuffle tests, and only one helps.** Scrambling the sequence on
a *fixed* backbone drops the score sharply (97% of the time) — that validates
the model. Scrambling the sequence and then *folding it* yields a structure
that scores as well as a genuine design, marginally better in fact.

Boltz therefore produces self-consistent geometry for essentially any sequence,
so this score **cannot reject a design for having an implausible fold** — a
nonsense sequence earns an equally self-consistent backbone. It measures the
folding machinery's self-consistency, not design quality.

Caveat: only **4** shuffle co-folds exist in the Boltz set, so this rests on
thin evidence. Pushing more shuffled sequences through the co-fold lane would
settle it cheaply.

The lane signal (esm1 vs esm2, +0.117 log P/res, **+2.3σ**) is detectable but
modest; suggestive rather than settled.

## Inference workflow (`score_structure.py`)

```sh
./.venv/bin/python scripts/score_structure.py STRUCTURE [STRUCTURE ...] \
    [--sequence SEQ] [--suggest] [--device cpu|mps|cuda] [--json out.json]
```

**Input**: any `.pdb` / `.cif` / `.cif.gz` holding an 8-50 residue peptide
chain. Only backbone N/CA/C is read -- no side chains, no ligand, no MSA, so it
works on a backbone-only model. `--sequence` scores a *different* sequence of
the same length on that backbone.

**Output** per structure:

| | |
|---|---|
| `mean log P/res` | the score: how much this geometry wants this sequence |
| `shuffled ... z` | the same backbone with the sequence scrambled, 20×. **z > 2 means the backbone discriminates its own sequence**; z <= 0 means it does not, and the score is uninformative for that structure |
| `PDB band` | distance in sd from 120 held-out experimental peptides (-2.600 +- 0.349) |
| `--suggest` | per-position top-3 preferred residues, per-residue log P, and recovery % |

### Worked example: ranking fold lanes

One 33-residue glycine-rich design on six
backbones:

| lane | log P/res | z | vs PDB mean | Rg (A) | contacts |
|---|---|---|---|---|---|
| **esmfold** (apo) | **-3.479** | **-1.3** | -2.5 sd | **27.9** | **0** |
| rf3apo | -3.014 | +4.0 | -1.2 sd | 7.7 | 125 |
| of3apo | -2.488 | +2.3 | +0.3 sd | 8.4 | 71 |
| boltz co-fold | -2.409 | +3.4 | +0.5 sd | 7.5 | 94 |
| openfold3 | -2.320 | +2.8 | +0.8 sd | 7.1 | 116 |
| **rf3** (ligand) | **-2.190** | **+4.2** | **+1.2 sd** | 7.7 | 117 |

**Ligand-bound beats apo in both matched pairs** (openfold3 > of3apo,
rf3 > rf3apo) -- the ligand shapes a pocket that then makes sequence sense.

**ESMFold is degenerate for this peptide**, and the z-score catches it without
anyone looking at the structure: z = -1.3 means the real sequence scores *worse*
than its own shuffles, and the shuffled spread collapses to 0.033 against
0.14-0.21 elsewhere. The geometry explains it -- 93.5 A end-to-end against a
~115 A fully-extended limit, Rg 27.9 A, **0 contacts**, 0% alpha-R. ESMFold
declined to fold it. With no contact edges the graph is a bare path, so the
model has almost no information and goes indifferent to sequence.

**Detecting a non-structure is therefore a real capability** of this score, and
it is cheap: seconds per structure against a dock-and-simulate cycle.

### All lanes, all peptides (`compare_lanes.py`)

181 structures, 20 shuffles each. Reported as **perplexity** (effective number
of residue types the model is choosing between; 21 = no information) and **z**
(same backbone, sequence shuffled -- isolates whether the backbone cares about
residue *order* rather than composition).

**Paired on the 21 sequences present in all five fold lanes**, so no lane can
look good by holding easier peptides:

| lane | median perplexity | mean z | best-of-lane wins |
|---|---|---|---|
| **openfold3** (ligand) | **9.1** | +3.6 | **12 / 21** |
| rf3 (ligand) | 11.9 | +4.2 | 3 |
| of3apo | 12.0 | +3.4 | 3 |
| rf3apo | 13.9 | +3.6 | 2 |
| esmfold | 16.1 | +1.8 | 1 |

uniform = 21.0, PDB held-out = 13.5

**Boltz paired per lane** (13 shared sequences; Boltz has different coverage so
it is excluded from the five-lane set above):

| lane | lane px | boltz px | diff | sigma |
|---|---|---|---|---|
| **openfold3** | **11.4** | 14.3 | **+2.9** | **+3.2** |
| of3apo | 13.4 | 14.3 | +0.9 | +1.1 |
| rf3 | 13.8 | 14.3 | +0.5 | +0.4 |
| rf3apo | 17.9 | 14.3 | -3.6 | -2.1 |
| esmfold | 26.6 | 14.3 | -12.3 | -3.3 |

**Three findings.**

**OpenFold3 with ligand is the best lane**, beating Boltz -- the current primary
co-fold lane -- by +3.2 sigma.

**Ligand-bound beats apo.** Paired on identical sequences: openfold3 vs of3apo
is +1.4 perplexity at **+3.6 sigma** (15/21 sequences), rf3 vs rf3apo is +2.6 at
+1.7 sigma (13/21). Same direction, firm for OF3, suggestive for RF3.

**ESMFold fails systematically, with one clean mechanism.** Of the 8 structures
in 181 with z <= 0, **7 are ESMFold and 6 of those have exactly zero contacts**
(Rg 19-28 A) -- extended chains, not folds. ESMFold is apo-only and these are
ligand-dependent designs, so it has nothing to fold around; a coherent
explanation rather than a defect.

### Caveats on the lane comparison

- It measures **self-consistency, not correctness**. A lane can produce a
  confidently wrong fold and score well.
- The designs are far more helical than PDB peptides (frac alpha-R 0.69-0.83 vs
  0.52), so scoring better than the experimental band partly reflects helices
  being easy to recognise. The **lane** comparison is unaffected, being paired
  on identical sequences.
- n = 13-21 throughout.
- It cannot validate a design: scrambled sequences that are then folded score
  just as well (see above).

### What it is for, and what it is not

Good for: ranking fold lanes for one sequence; flagging extended or disordered
predictions; comparing candidate sequences on a fixed backbone.

Not for: **binding affinity** (no ligand in the model); **filtering designs**
(scrambled sequences that are then folded score just as well, see above);
**trusting `--suggest` literally** -- PDB short peptides are 9% cysteine, so the
model proposes C and P far more often than this design space warrants.

## Baselines, before any result is believed

The pattern that caught both earlier negative results applies here too — see
[decisions.md](decisions.md).

- **Residue frequency.** Predict the most common residue, always. On Gly-rich
  designs this is strong; on PDB peptides it is weaker but not negligible.
- **Position only.** Identity from terminus/middle plus chain position, no
  geometry at all.
- **Local-geometry only.** A per-residue classifier on that residue's own φ/ψ
  and burial, with no message passing. **This is the one that matters**: it
  isolates what the graph adds beyond purely local structure, which is where
  angle denoising collapsed from an apparent 21% to a real 1–4%.

Report perplexity and recovery rate against all three, at every stage.

## Device: CPU, deliberately

Measured on a batch of 32 (~744 nodes, ~3,936 edges):

```
featurisation   2.41 ms/example  ->  77 ms per batch   (69% of wall time)
CPU  fwd+bwd   34.45 ms/step
MPS  fwd+bwd   46.47 ms/step     ->  35% SLOWER than CPU
```

Peptide graphs are small enough that MPS kernel-launch overhead exceeds the
arithmetic, reproducing the finding already recorded in the sibling `mpnn`
repo. Moving to MPS would make the end-to-end loop ~11% slower.

**The bottleneck is Python featurisation, not compute.** `graph_from_coords` is
deterministic per conformer — edges and base node features never change between
epochs, only the mask does. Caching the built graph and re-masking each epoch
would cut epochs from ~21 s to ~7 s. Worth doing before scaling to all 52,423
conformers or running many seeds; a GPU is not.

## Splits

By **sequence cluster**, never by entry or conformer. NMR models of one peptide
are near-duplicates, and two entries of the same peptide are the same data
point. RCSB's 30%-identity clustering provides the grouping.

Also verify no designed sequence appears in the PDB set before training — the
alanine-rich and shuffled designs could plausibly collide with deposited
engineered peptides, and it is a cheap check.
