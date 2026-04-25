"""
phytologue.analysis — distillation of trajectory s∃ into scalar features.

Stage 7 (s∃ ⟳ tϕ) of the pipeline turns Trajectories into:
    * contact maps               (ligand vs. each protein residue)
    * per-residue RMSF
    * residence time             (ascorbate near heme Fe)
    * water occupancy            (placeholder in v0)

Outputs are dicts of scalars + numpy arrays — small, JSON-serialisable so
the Slurm-ready BatchRunner can pass them through stdin/stdout.
"""

from .contacts import ligand_residue_contacts, contact_frequency
from .rmsf import rmsf_per_residue
from .residence import residence_time

__all__ = ["ligand_residue_contacts", "contact_frequency",
           "rmsf_per_residue", "residence_time"]
