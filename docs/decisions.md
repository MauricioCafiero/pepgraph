# Approaches considered and rejected

Recorded so they are not re-proposed without the reason they were dropped.
All of these were decided on 2026-10-10.

## Graph autoencoder — rejected

The original idea: reconstruct the peptide graph through a bottleneck and use
the latent space as a representation.

**Why not.** An autoencoder's loss is reconstruction, and if the sequence is
handed to the model as node features, reconstructing it is solved by copying —
the encoder is under no pressure to build a meaningful latent. What makes
autoencoders produce useful representations is scale plus a bottleneck tight
enough to force abstraction. With 76 Boltz co-folds, really ~21 independent
sequences all produced by the same pipeline, the model memorises instead. The
MD frames do not fix this: 10,000 frames of one peptide are 10,000 correlated
views of one data point, not 10,000 data points.

## Masked residue prediction / inverse folding — UNBLOCKED and done

Originally rejected for lack of sequences (below). The PDB supplied 1,957
cluster groups, which removed that objection, and it is the **only target so
far where the graph clearly earns its place**: 18–30% recovery over a
local-geometry-only baseline, against 1–4% for angle denoising. Full result in
[inverse-folding.md](inverse-folding.md).

The original reasoning, kept because the data argument was correct at the time:

## (original) Masked residue prediction / inverse folding — rejected

Mask the residue identity on some nodes, keep the geometry, predict the
identity back. This is what ProteinMPNN does, and unlike a sequence-only
language model it genuinely needs the graph, since the geometry is the input
and the sequence is the answer.

**Why not here.** The data has **21 distinct sequences**. There is no sequence
variety to learn from, and the Gly-rich composition means the baseline of
"always guess G" is strong (568 of 2326 residues in the co-folds). It was not
rejected because it is sequence-only — that was a misreading worth correcting —
but because this dataset has conformations in abundance and sequences in
almost none.

It would become viable on PDB-derived peptide fragments (thousands of
sequences), with this project's 21 as an evaluation set. That path is still
open; it is a dataset problem, not a design problem.

## Angle denoising — done, and it under-delivered

See [angle-denoising.md](angle-denoising.md). Chosen because the data's
abundance is conformational. It works, but the graph contributes only 1–4% over
a geometry-blind Bayes estimator, because φ/ψ denoising is intrinsically local:
a residue's clean angles are mostly set by which basin it occupies, which its
own noisy angles and identity already say.


## Intra-peptide contact prediction — rejected, the task is deterministic

Predicting which non-sequential residue pairs are in contact, from the graph.

**Why not.** The graph's edges carry φ/ψ/ω, and the contact map is a
deterministic function of the torsions — `kinematics.py` computes it. Measured
over 360 frames, the purely analytic route (torsions → `build_backbone` →
contacts at 8 Å) gives:

```
precision 0.743   recall 0.870   F1 0.801
contact density 13.3% of eligible pairs
("always contact" baseline: F1 0.234)
```

F1 = 0.801 rather than 1.000 only because ideal bond lengths and angles are
used; passing the *measured* internals makes the round-trip exact to
8×10⁻¹⁴ Å. So the information is fully present in the input and a network would
be learning to approximate forward kinematics. If the contact map is wanted,
compute it.

## Ligand contact prediction — the best-posed target so far, not yet attempted

"Given this peptide conformation, which residues does the ligand touch?" Not
determined by the peptide alone, genuinely non-local, and the labels come free
from the trajectories.

Unlike dG the labels really do vary frame to frame (stride 50, 4.5 Å cutoff):

| run | frames | mean contacts | within-run Hamming | frames changing | entropy |
|---|---|---|---|---|---|
| `bg33_1_dock1` | 400 | 6.9 | 7.1 | 76% | 0.47 bits/res |
| `bgox31_3_dock1` | 400 | 10.1 | 2.4 | 79% | 0.19 |
| `ox2_shuffle_o3cof` | 40 | 2.1 | 4.8 | 51% | 0.30 |
| `orig_f12` | 400 | 9.7 | 8.9 | 81% | 0.59 |

Two random frames of one run differ in 2.4–8.9 contacting residues out of ~31,
so this is ~2.6M varying binary labels rather than 68 scalars.

**Three caveats that decide whether it is worth it.**

1. **It bundles two questions.** *Where* on the peptide the ligand binds is
   fixed per run by the initial dock, so that question has ~68 independent
   examples — the same wall dG hit. *Which* residues are in contact in a given
   frame is abundant.
2. **Much of the frame-level variation is thermal jitter.** Entropy of
   0.19–0.59 bits against a mean of 7–10 contacts indicates a stable core with
   a flickering periphery: residues hovering at the 4.5 Å cutoff crossing it
   back and forth. Predicting that is predicting noise. A persistence-weighted
   label, or a wider cutoff, would separate signal from flicker.
3. **Induced fit is a shortcut.** These frames were equilibrated *with* the
   ligand bound, so the pocket is already moulded around it. A model may score
   well by finding the concave region — pocket detection rather than binding
   prediction. Useful, but not what it would appear to be learning.

**Next step if pursued: probe, do not build.** A per-residue logistic
regression on burial/solvent exposure, local concavity, hydrophobicity and
sequence position, scored leave-one-sequence-out. That is the move that settled
dG, where a linear model reached R² = 0.905 and removed the case for a network.
Only if a real gap survives that does the graph have an argument — and this is
the one target where it would be load-bearing.

## Sequence-structure compatibility as a design filter — does not work

Scoring a design by how much its backbone "wants" its own sequence passes every
internal check — the shuffle control separates real from scrambled sequences on
a fixed backbone 97% of the time, and the designs sit inside the experimental
PDB band. But scrambling a sequence and then **folding it** produces a structure
that scores as well as a genuine design. Boltz gives self-consistent geometry to
any sequence, so the score cannot reject a design for an implausible fold.

Useful as a check on the folding machinery; useless as a design filter. Based on
only 4 shuffle co-folds, so worth firming up with more.

## The pattern worth keeping

Both negative results only appeared because a baseline was built:

- angle denoising looked like a **21%** win until the shrinkage prior showed
  19 of those points were the Ramachandran distribution;
- a separate energy-regression target turned out to be solved by a linear model
  on a handful of descriptors, leaving a network nothing to do.

Any future claim in this project needs its geometry-free or descriptor-only
baseline reported **alongside** it, not afterwards. The absolute number is
meaningless on its own.

## Deliberately out of scope

**No ligand in the graph.** Decided at the outset. It is the main structural
reason the graph could not contribute to energy regression: binding energy is a peptide–ligand
property and the graph describes only the peptide, so the seven interface
descriptors carrying nearly all the signal have no graph equivalent. Any future
energy target needs this revisited — predicting an interaction energy while
blind to one of the two partners is not a modelling problem that better
architecture fixes.
