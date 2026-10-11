# Data: trajectories, selection and shards

## MD trajectories

`trajectory.py` reads the `traj_wrapped.xtc` / `traj_wrapped.pdb` pairs under
`runs/*/md/*/`. Frames are read in chunks via `mdtraj.iterload`, since the
longest trajectories hold 20k frames each. Coordinates are converted from
mdtraj's nm to ångström; the peptide chain is picked as the chain with the most
canonical residues, which drops the `LIG` chain without naming it.

```python
from pepgraph import iter_trajectory_graphs, find_trajectories

for xtc, top in find_trajectories("<runs>"):
    for frame_no, g in iter_trajectory_graphs(xtc, top, stride=100):
        ...   # g.x [n_res, 32], g.edge_index, g.edge_attr
```

`torsion_arrays()` is the faster path when only the angles are wanted: it
returns `[n_frames, n_residues]` arrays without building a graph per frame.

### Inventory (`scan_trajectories.py`)

```
113 trajectories          892,040 frames total
                           21 distinct sequences
  system B   575,000 frames
  system A   317,040 frames

featurisation: 513 frames/s  ->  the whole set in ~0.5 h
a 33-residue frame: 33 nodes, 122 edges (64 backbone + 58 contact)
```

Every stage spans exactly the time its name claims, at one of two save
intervals:

| Stage | Files | Frames | dt | Span |
|---|---|---|---|---|
| `prod_20ns` | 38 | 20,000 | 1 ps | 20 ns |
| `prod_25ns` | 1 | 25,000 | 1 ps | 25 ns |
| `seg_20_25ns` | 1 | 5,000 | 1 ps | 5 ns |
| `prod_L1_modal` | 30 | 2,000 | 10 ps | 20 ns |
| `first_15ns` / `first_10ns` / `first_5ns` | 14 each | 1,500 / 1,000 / 500 | 10 ps | 15 / 10 / 5 ns |
| `rate_test` | 1 | 40 | 1 ps | 0.04 ns |

**Two traps follow from this, and the mining step has to handle both.**

*The save intervals differ by 10×.* A frame from `prod_20ns` is 1 ps from its
neighbour; one from `prod_L1_modal` is 10 ps. Pooling them raw over-weights the
1 ps trajectories with frames that are almost copies of each other. Striding the
1 ps trajectories by 10 puts everything on a uniform 10 ps footing — and turns
each `prod_20ns` into exactly 2,000 frames, matching the Modal legs.

*42 of the 113 files contain no unique frames.* For all 14 runs that have them,
`first_5ns` ⊂ `first_10ns` ⊂ `first_15ns` ⊂ `prod_L1_modal`, verified by exact
coordinate comparison (max difference 0.000000 nm) — they are checkpoint dumps
written during the run, not separate sampling. The same holds for
`orig_f12_dock5`, where `prod_20ns` and `seg_20_25ns` are both contained in its
`prod_25ns`. So the rule is: per run, keep the longest-spanning production
stage and skip the rest.

After deduplication and striding to a uniform 10 ps, the real budget is roughly
**136,000 frames**, not 892,040.

**Frames are abundant, sequences are not, and the two are badly unbalanced.**
One sequence has 100,040 frames and another 4,000 — a 25× spread — so anything
trained on frames needs per-sequence capping or weighting, or it will mostly
learn the peptide that happened to get the most simulation time.

The 21 sequences do span distinct composition regimes, which is more useful
than the raw count suggests: glycine-rich pipeline designs, alanine-rich
helices, charged helical designs, and shuffled-sequence nulls. Sequences are
withheld.

### MD geometry vs the co-folds (`rama_md.py`)

4475 frames sampled at stride 200, 129,339 residue observations:

| | alpha-R | beta/PPII | alpha-L | outlier | mean abs(omega) |
|---|---|---|---|---|---|
| Boltz co-folds | 60.4% | 24.4% | 6.3% | 9.0% | 177.6° |
| MD ensemble | 50.3% | 25.1% | 6.9% | 17.7% | 171.8° |

The basins agree; MD is simply broader, which is what 300 K thermal motion does
to an idealised predicted structure. Two checks confirm the trajectory reader
rather than a units or wrapping error:

- **The C–N peptide bond stays within 1.238–1.435 Å in every frame checked**, and
  no bond comes out broken. Despite the `_wrapped` filenames, no peptide is
  split across the periodic boundary — had one been, dihedrals spanning the
  split would be garbage and the elevated outlier fraction would be an artifact.
- **Every cis omega is Gly→Gly** (119 of 119 in the sampled frames, 0.22% of
  bonds). That is the one context where cis is physically plausible, glycine
  having no side chain to penalise it. The trans↔cis barrier is far too high to
  cross in 20 ns, so these are cis bonds built in upstream by the structure
  prediction and preserved by the simulation — not a featuriser bug. Worth
  knowing because the Boltz co-folds contain no cis bonds at all, so the cis
  geometry enters from one of the other fold or docking lanes.

## Mining frames to disk (`mine_frames.py`)

```sh
./.venv/bin/python scripts/mine_frames.py --dry-run   # see the selection first
./.venv/bin/python scripts/mine_frames.py             # write outputs/frames/
```

Result: **68 shards, 84,000 frames, 49 MB, 7 seconds**, plus a `manifest.csv`.

### What is stored, and why it is coordinates

Shards hold the **backbone N/CA/C coordinates** as `[n_frames, n_residues, 3, 3]`
float32, not finished graphs. Two reasons:

- Node features are identical in every frame of a sequence — identity and
  terminus do not change, and pLDDT is 0 for MD — so storing graphs per frame
  would duplicate several hundred MB of unchanging data.
- Coordinates keep the contact cutoff a **load-time parameter**. The same shard
  gives 242 edges at 8 Å and 408 at 10 Å, so that choice can be revisited
  without re-reading 1.4 GB of trajectory.

`load_shard(path, contact_cutoff=...)` rebuilds `PeptideGraph` objects, and
`iter_shards(dir)` walks them all.

### Selection

```
113 trajectories on disk            892,040 frames
  -> dedup by run                    68 kept, 45 dropped
  -> stride to a uniform 10 ps
  -> cap 4,000 frames per sequence   84,000 frames
```

The cap lands at exactly 4,000 for **all 21 sequences**, because 4,000 is also
the smallest number any sequence can supply. The result is a perfectly balanced
set — no peptide is over-represented because it happened to get more simulation
time, which was the risk the raw 25× frame spread created.

### Round-trip check

Graphs rebuilt from a shard are identical to graphs built directly from the
source trajectory at the same stride — node features, edge features, edge
indices and torsions all agree to **0.000e+00** over 25 frames. So the stored
coordinates are a lossless substitute for re-reading the trajectory, and the
`.xtc` files are never needed again unless the stride or the dedup rule changes.


## Data available

| Source | Count | Notes |
|---|---|---|
| Boltz co-fold CIFs | 76 | sequence + bound conformation, carries pLDDT |
| MD `traj_wrapped.xtc` | 113 files, 1.4 GB | **892,040 frames**, 21 sequences |
| MD `frames_last.pdb`, `frame_medoid.pdb` | 94 runs | ~10 frames each, no mdtraj needed |
