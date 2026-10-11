# Validation

## Geometry validation

`scripts/validate_rama.py` featurises all 76 Boltz co-folds under
`<runs>/*/boltz/` (2326 residues) and reports:

```
ω: 100.0% trans, 0 cis, mean |ω| 177.6°
φ/ψ defined for 2174 of 2326 residues   (the 152 missing = 76 × 2 chain termini)

alpha-R      60.4%
beta/PPII    24.4%
alpha-L       6.3%
outlier       9.0%
```

This is the check that matters, because a featuriser that reads the backbone
atoms in the wrong order still produces well-shaped tensors and trains without
complaint — it just learns from noise. Four things confirm the geometry:

- **ω is 100% trans at a mean of 177.6°.** Scrambled atom ordering would spread
  ω over the whole circle, so this is the single strongest signal.
- **αR dominates at 60%,** which fixes the sign convention. A sign error mirrors
  the plot, putting that density in αL instead.
- **αL is 88% glycine,** and 21% of all glycines sit there. 6.3% in αL is above
  the 2–4% typical of folded proteins, but these are glycine-rich designed
  peptides (G is 568 of 2326 residues), and glycine populates αL freely. So the
  high number is the sequence composition, not a bug.
- **Exactly 152 undefined φ/ψ,** matching two chain termini per structure.

`dihedral()` also returns 0.0° for cis-planar and 180.0° for trans-planar test
geometry, and agrees with `mdtraj.compute_phi/psi/omega` on the same 50 frames
to within **6×10⁻⁵ degrees** — float32 rounding, i.e. an exact match against an
independent implementation, sign convention included.


## Plumbing check against the MPNN (`forward_pass.py`)

The graphs are built in the layout `../../mpnn` already consumes, so that
repo's `collate` and `MPNN` are reused unmodified — `PeptideGraph` exposes the
same `x` / `edge_index` / `edge_attr` / `y` attributes as `MolGraph` and
duck-types straight in. Only the dimensions change, `MPNN(node_dim=32,
edge_dim=10)`, giving 102,401 parameters at hidden 64 and 3 layers.

The weights are random, so the outputs mean nothing. What this rules out is the
class of bug where the shapes are compatible but the wiring is wrong:

```
batch of 11 graphs, lengths [25, 27, 31, 33, 34]
collated: x (348, 32)  edge_index (2, 1394)  edge_attr (1394, 10)

node counts, edge offsets and graph isolation ..... OK
output (11,), one scalar per graph, all finite .... OK
all 38 parameter tensors receive gradient ........ OK  (none dead)
reordering the batch permutes outputs ............ 4.77e-07
one graph alone == the same graph in a batch ..... 1.91e-06
```

The batch deliberately mixes peptide lengths and mixes predicted structures
with MD frames, because a batch of equal-length graphs never exercises the
node-offset arithmetic. The asserted invariants are that `batch` node counts
match the per-graph residue counts, that no edge index exceeds the stacked node
count, and that **no edge joins two different graphs** — that last one is what
catches an offset error, since a wrong offset silently connects one peptide to
another and the model trains anyway. The two 10⁻⁶ residuals are float32
summation order, which is unavoidable once graph sizes differ.

