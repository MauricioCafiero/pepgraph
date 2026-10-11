# Forward kinematics


`build_backbone()` turns torsions back into N/CA/C coordinates, placing each
atom from the previous three by NeRF. The backbone is treated as one atom chain
N(0), CA(0), C(0), N(1), … along which the torsion placing `N(r)` is ψ(r−1),
`CA(r)` is ω(r−1) and `C(r)` is φ(r), so the per-residue arrays map onto the
chain exactly.

This exists for denoising. Corrupting torsions is only half the task — the
other half is recomputing everything the graph derives from coordinates, above
all the CA–CA contact distances, which otherwise still describe the **clean**
structure and hand the model the answer.

```
1. measured bonds + angles + torsions   RMSD max 8.01e-14 A   -> PASS
2. ideal bonds + angles, measured phi/psi/omega   RMSD mean 1.23 A, max 2.10 A

3. backbone RMSD caused by perturbing phi/psi (ideal geometry)
     sigma 1 deg ->  0.64 A       sigma 10 deg ->  6.57 A
     sigma 2 deg ->  1.25 A       sigma 20 deg -> 11.21 A
     sigma 5 deg ->  3.00 A       sigma 45 deg -> 12.60 A
```

Test 1 rebuilds from the structure's own measured internal coordinates and
recovers it to **8×10⁻¹⁴ Å**, so the frame construction and the torsion-to-atom
mapping are right. Everything else rests on that.

Test 3 is the one that shapes the task. **The chain is a lever**: 1° of
per-residue torsion noise already displaces the backbone by 0.64 Å, because
small local errors compound along 33 residues, and by σ = 20° the RMSD has
saturated near 12 Å — the fold is simply gone. So a useful noise schedule lives
in roughly **σ ∈ [1°, 15°]**, and coordinate RMSD is a bad training metric
here: it is dominated by lever-arm amplification at the chain ends rather than
by how well the angles were recovered. Per-residue angular error in degrees is
the honest measure.

Test 2 matters only for the inputs. Ideal bond lengths and angles cannot
reproduce a 30-residue peptide closer than ~1.2 Å, but that is **not** a floor
on the denoising task, because the targets are torsions, which are exactly
recoverable. It only means the rebuilt coordinates feeding the contact map are
approximate — and consistently so for clean and corrupted alike, which is what
keeps the leak closed. Passing the measured `bonds`/`angles` into
`build_backbone` removes that 1.2 Å entirely and leaks very little, since bond
lengths and angles are local covalent geometry and carry almost no information
about the fold.

