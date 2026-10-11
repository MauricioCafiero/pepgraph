# Angle denoising


An example is made by perturbing a frame's φ/ψ, **rebuilding the backbone from
the corrupted torsions**, and featurising that. Nothing derived from the clean
coordinates reaches the input — in particular the CA–CA contact distances come
from the rebuilt structure, so they cannot leak the answer. ω is left
unperturbed, since a ~20 kcal/mol barrier keeps it planar in reality and
noising it would teach the model to undo something that does not happen.

### The model predicts a correction, not an angle

The first version regressed the clean angle directly and settled into a flat
~47°/80° error that **did not vary with σ at all** — the signature of a model
ignoring its input and emitting a prior. The reason is structural: at low σ the
optimal denoiser is almost exactly the identity, so predicting the angle makes
the model learn to reproduce its own input to within a degree, starting from
arbitrarily far away.

Regressing `clean − noisy` instead puts the identity baseline at **zero
output**. The head's last layer is zero-initialised, so an untrained model *is*
the baseline and training can only improve on it. This is the same reason
diffusion models regress the noise rather than the signal. The noisy angles are
given to each node directly as sin/cos, with σ alongside, so a correction can
be read off locally and then refined by message passing.

### Baselines

Two, and both are reported at every noise level, because without them an error
in degrees means nothing:

- **Identity** — return the input unchanged, i.e. a zero correction. Its error
  *is* the injected noise. This is the baseline that embarrasses denoisers, and
  at σ = 1° it sits at 0.8°, so there is very little room to win.
- **Per-residue-type mean** — the circular mean φ/ψ for each residue letter,
  fit on the training sequences, ignoring all geometry. This lands around
  50–70°, which is the cost of knowing only the sequence.

### Split

**By sequence, never by frame.** Frames of one peptide are correlated to the
point of near-duplication, so a frame-level split would put near-copies of the
training set into validation and report a number that means nothing. With 21
sequences the default is 15 train / 3 val / 3 test.

### Result

3000 train frames, 15 train sequences, hidden 128, 4 layers, 484,866 params,
30 epochs in 678 s. Mean absolute error in degrees on the three **held-out
sequences**:

| σ | model φ | identity φ | model ψ | identity ψ | gain |
|---|---|---|---|---|---|
| 1° | 0.80 | 0.80 | 0.82 | 0.80 | none |
| 2° | 1.60 | 1.60 | 1.60 | 1.60 | none |
| 5° | 3.80 | 3.99 | 3.85 | 4.03 | ~5% |
| 10° | 6.74 | 7.97 | 7.01 | 7.94 | ~13% |
| 15° | 9.26 | 11.93 | 9.65 | 12.02 | ~21% |

per-residue-type mean: 54.0° φ, 68.5° ψ

The shape is what the formulation predicts: nothing at low noise, where the
identity baseline is already near-optimal, and a real improvement at σ ≥ 10°.
It transfers to sequences never trained on.

Validation plateaued at **epoch 11** (4.20°) while training loss kept drifting
down — mild overfitting, and the last 19 epochs bought nothing. About 12 epochs
is enough.

### The graph is worth 1-4%, not 21% (`eval_denoise.py`)

The 21% above is the improvement over the **identity** baseline, and
attributing it to the graph was wrong. φ/ψ are not uniformly distributed, so
pulling any noisy angle toward the αR basin helps on average while knowing
nothing whatsoever about the structure. `ShrinkagePrior` measures exactly that:
the Bayes-optimal estimator given the residue identity, the noise level and the
full **joint** 2D Ramachandran prior — but no geometry. It is also given the
estimator that minimises mean absolute error, since MAE is what is reported.

φ on the held-out sequences (ψ is near-identical):

| σ | identity | shrink 5° | shrink 2° | shrink 1° | model | graph gain |
|---|---|---|---|---|---|---|
| 1° | 0.79 | 1.44 | 0.91 | 0.82 | 0.79 | 3.6% |
| 2° | 1.61 | 1.99 | 1.66 | 1.61 | 1.59 | 1.0% |
| 5° | 3.97 | 4.00 | 3.85 | 3.82 | 3.75 | 2.0% |
| 10° | 8.02 | 7.13 | 7.03 | 7.03 | 6.80 | 3.2% |
| 15° | 11.97 | 9.78 | 9.70 | 9.70 | 9.32 | 4.0% |

At σ = 15°: identity 11.97 → prior 9.70 (19%) → model 9.32 (a further 4%). So
**the Ramachandran prior accounts for nearly all of the apparent gain** and
message passing adds 1–4%.

Two notes on reading the table. The 5° prior is *worse than identity* at low σ,
which is impossible for a Bayes estimator: it returns bin centres, so
quantisation floors its error near 1.25°. That is why the finer priors exist,
and why the original "model beats shrinkage by 45% at σ = 1°" was spurious. The
2° and 1° priors then agree to 0.00°, so the baseline is resolution-saturated
and the remaining gap is real rather than a discretisation artifact.

### What this says about the task, not the architecture

The gain grows monotonically with σ (1% at 2°, 4% at 15°), which is the right
direction — more corruption leaves more for context to repair. But the ceiling
is low, and the reason is that **φ/ψ denoising is intrinsically local**: a
residue's clean angles are mostly determined by which basin it occupies, and
its own noisy angles plus its identity already say that. Distant contacts have
little left to contribute, so the graph has little room to earn its place. No
amount of tuning turns 4% of headroom into a large result.

A task where the graph would be load-bearing needs a genuinely **non-local**
target — the contact map itself, which residues touch the ligand, or the
binding-energy value sitting in each MD run — none of which can be recovered from a
residue's own identity and angles.

Caveat throughout: three held-out sequences. Directional, not a benchmark.

