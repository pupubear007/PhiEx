"""
phytologue.core.state — entities (∃).

These dataclasses are pure data.  No physics, no I/O, no tool-specific
types leaking in.  Every adapter (PDB fetcher, ESMFold, OpenMM, MACE, …)
translates ITS native types into these.  That is the entire point of
the adapter layer.

Conventions:
    * coordinates in Angstroms
    * energies in eV unless an adapter explicitly says otherwise
    * residue indices are 1-based, matching PDB / UniProt convention
    * `Atom.kind` keeps the chemistry-meaningful element symbol; topology
      lives on the parent Residue / Cofactor / Ligand
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ────────────────────────────────────────────────────────────────────────
# atomic primitives
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Atom:
    index: int             # 0-based atom index within its parent entity
    name: str              # PDB atom name (e.g. "CA", "FE", "O1")
    element: str           # element symbol — "C", "N", "O", "FE"
    xyz: tuple[float, float, float]  # Å
    bfactor: float = 0.0   # for predicted structures we stash pLDDT here

    def with_xyz(self, xyz: tuple[float, float, float]) -> "Atom":
        return Atom(self.index, self.name, self.element, xyz, self.bfactor)


@dataclass(frozen=True)
class Residue:
    index: int             # 1-based residue number (PDB-style)
    name: str              # 3-letter code, e.g. "ARG"
    chain: str             # chain id, e.g. "A"
    atoms: tuple[Atom, ...] = ()

    @property
    def ca(self) -> Optional[Atom]:
        for a in self.atoms:
            if a.name == "CA":
                return a
        return None


# ────────────────────────────────────────────────────────────────────────
# Protein  (the macromolecular ϕ target)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Protein:
    """A protein structure + its sequence.

    Either or both can be present:
        * a Protein from PDB has structure (residues populated) + sequence
        * a Protein from ESMFold has sequence + predicted structure;
          plddt is per-residue confidence in [0,100]
        * a "design" target may have only a sequence, no structure yet
    """
    name: str
    sequence: str = ""
    residues: tuple[Residue, ...] = ()
    source: str = "unknown"          # "pdb", "esmfold", "user-upload"
    plddt: tuple[float, ...] = ()    # per-residue confidence (predicted only)
    pdb_text: Optional[str] = None   # cached PDB text if we have it
    metadata: dict = field(default_factory=dict)

    @property
    def n_residues(self) -> int:
        return len(self.residues) if self.residues else len(self.sequence)

    def all_atoms(self) -> Iterable[Atom]:
        for r in self.residues:
            yield from r.atoms


# ────────────────────────────────────────────────────────────────────────
# Cofactor  (endogenous, e.g. heme)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Cofactor:
    """A non-protein, covalently or coordinately bound prosthetic group.

    For APX this is heme (HEM in PDB).  For other systems this could be
    NAD(P)H, FAD, [4Fe-4S], etc.  The key field for downstream reasoning
    is `metal_atom`: many cofactor-aware heuristics (e.g. "active-site
    pocket is the one closest to heme Fe") need the metal coordinates.
    """
    name: str                        # e.g. "HEME"
    resname: str = ""                # PDB residue name, e.g. "HEM"
    atoms: tuple[Atom, ...] = ()
    metal_atom: Optional[Atom] = None
    smiles: str = ""                 # optional, for ligand-pocket fingerprinting
    metadata: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────
# Ligand  (exogenous, e.g. ascorbate)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Ligand:
    name: str
    smiles: str
    atoms: tuple[Atom, ...] = ()     # may be empty until placed by docking
    sdf_text: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────
# Complex state  (∃ for the whole system)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexState:
    """The full system at one point in time.

    This is what gets fed to a Calculator for energy/forces and what gets
    snapshotted into a Trajectory frame.  Keeping protein, cofactors and
    ligands as separate fields (rather than one flat atom list) is the
    architectural choice that lets us route different regions to different
    Calculators — classical for the protein bulk, MACE for the active-site
    region, etc.  This is the v0 stand-in for a QM/MM-style partition.
    """
    protein: Protein
    cofactors: tuple[Cofactor, ...] = ()
    ligands: tuple[Ligand, ...] = ()
    time_ps: float = 0.0             # simulation time, picoseconds
    step: int = 0
    energy_ev: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def n_atoms(self) -> int:
        n = sum(len(r.atoms) for r in self.protein.residues)
        n += sum(len(c.atoms) for c in self.cofactors)
        n += sum(len(l.atoms) for l in self.ligands)
        return n


# ────────────────────────────────────────────────────────────────────────
# downstream entities (∃ produced by adapters)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Pocket:
    """Output of a pocket detector (P2Rank).

    `score` is the detector's own confidence — interpret per-tool.
    `nearby_residues` is what the rest of the pipeline consumes.
    """
    id: str
    center: tuple[float, float, float]
    score: float
    nearby_residues: tuple[int, ...] = ()
    is_active_site: bool = False     # set by the heme-proximity heuristic
    metadata: dict = field(default_factory=dict)


@dataclass
class DockingPose:
    """Output of Vina / DiffDock — one ligand pose with score + uncertainty.

    `score_uncertainty` is None for deterministic scorers (Vina single-pose);
    for ML scorers (DiffDock confidence model) it carries the predicted SD.
    """
    ligand: Ligand
    score: float
    score_uncertainty: Optional[float] = None
    rmsd_to_best: float = 0.0
    pose_id: int = 0
