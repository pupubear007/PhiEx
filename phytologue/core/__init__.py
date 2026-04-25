"""
phytologue.core — the framework primitives.

These types are the architecture.  Every adapter in `phytologue.adapters/`
translates between an external tool's API and these types and nothing else.

    State        ∃   atomic / sequence / embedding states
    Calculator   ϕ   energies + forces (the classical theory)
    LearnedModel ϕ   first-class ML model with uncertainty
    Trajectory   s∃  a sampled run
    FittedTheory tϕ  the inferred / refined theory after iteration

The skeleton patterns come from the project's engine.py.  This file extends
them to multi-entity biomolecular systems (Protein + Cofactor + Ligand)
without inventing anything new.
"""

from .state import (
    Atom, Residue, Protein, Cofactor, Ligand, ComplexState,
    Pocket, DockingPose,
)
from .calculator import Calculator, ToyBarrierCalculator
from .learned import LearnedModel, Prediction, ModelLoadError
from .trajectory import Trajectory, Frame
from .theory import FittedTheory, TheoryRegistry

__all__ = [
    "Atom", "Residue", "Protein", "Cofactor", "Ligand", "ComplexState",
    "Pocket", "DockingPose",
    "Calculator", "ToyBarrierCalculator",
    "LearnedModel", "Prediction", "ModelLoadError",
    "Trajectory", "Frame",
    "FittedTheory", "TheoryRegistry",
]
