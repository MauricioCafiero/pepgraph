from .graph import (
    PeptideGraph, graph_from_structure, graph_from_residues,
    backbone_torsions, dihedral, NODE_DIM, EDGE_DIM,
)
from .structure import read_structure

__all__ = [
    "PeptideGraph", "graph_from_structure", "graph_from_residues",
    "backbone_torsions", "dihedral", "read_structure", "NODE_DIM", "EDGE_DIM",
]

# mdtraj-backed; imported lazily inside the functions so the rest works without it
from .trajectory import (  # noqa: E402
    iter_trajectory_graphs, find_trajectories, torsion_arrays,
)
