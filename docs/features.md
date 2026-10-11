# Graph features

## Node features (32)

| Slice | Content |
|---|---|
| `[0:21]` | one-hot residue identity — 20 canonical plus one "other" slot |
| `[21:24]` | one-hot position: N-terminal / middle / C-terminal |
| `[24:31]` | charge at pH 7, residue mass, aromatic, Kyte-Doolittle hydropathy, side-chain volume, side-chain H-bond donors, acceptors |
| `[31]` | pLDDT / 100, or 0 when the source carries no confidence (MD frames) |

For the 20 canonical residues the one-hot already determines every
physicochemical value, so `[24:31]` is strictly redundant. It is included
because it helps on small data and because it is what lets a noncanonical
residue mean anything at all — without it, an unrecognised residue would be a
single undifferentiated "other" bit. The H-bond counts and the 0.1 charge on
histidine are hand-assigned approximations.

## Edge features (10)

Both directions of every peptide bond, plus symmetric contact edges for
residue pairs more than 3 apart in sequence whose CA atoms are within 8 Å.

| Slice | Content |
|---|---|
| `[0:2]` | sin, cos of ψ |
| `[2:4]` | sin, cos of ω |
| `[4:6]` | sin, cos of φ |
| `[6]` | is a backbone edge |
| `[7]` | is a contact edge |
| `[8]` | direction: +1 for N→C, −1 for C→N, 0 for a contact |
| `[9]` | CA–CA distance / 10 |

Two decisions worth knowing about:

**The torsions describe the link, not the residue.** φ and ψ are defined per
residue, but the three torsions that span the bond from residue *i* to *i*+1 are
ψᵢ, ωᵢ and φᵢ₊₁, so that triplet is what the backbone edge carries. The
per-residue arrays are kept on the `PeptideGraph` as `phi`/`psi`/`omega` for
inspection.

**Angles are stored as sin/cos, not degrees.** Otherwise −179° and +179° look
maximally far apart when they are nearly the same angle. This also gives a free
missing-value encoding: an undefined angle is stored as sin = cos = 0, which no
real angle can produce since sin² + cos² = 1, so no separate mask bit is needed.

