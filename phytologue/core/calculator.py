"""
phytologue.core.calculator — ϕ as classical theory.

The Calculator is the ASE-style coat that every classical physics backend
wears.  In v0 the only real Calculator-shaped backend is OpenMMCalculator
in adapters/openmm_calc.py.  ToyBarrierCalculator below is the engine.py
1-D barrier carried over so unit tests can exercise the framework with
zero scientific dependencies.

Note:  ML potentials (MACE-OFF23, ANI, AIMNet2) wear *both* coats — they
are Calculators (energy/forces) AND LearnedModels (predict + uncertainty).
See adapters/mace.py for an example of dual implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math

from .state import ComplexState


@runtime_checkable
class Calculator(Protocol):
    """ASE-style classical theory.  Same shape as engine.py's Calculator
    but acts on ComplexState instead of the toy 1-D State."""

    name: str

    def energy(self, state: ComplexState) -> float:
        """Total potential energy in eV."""
        ...

    def forces(self, state: ComplexState) -> "list[tuple[float, float, float]]":
        """Per-atom forces in eV/Å.  Order matches state.all_atoms()."""
        ...


# ────────────────────────────────────────────────────────────────────────
# the toy carry-over from engine.py — used only by tests, kept here so the
# core package has at least one fully self-contained Calculator
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ToyBarrierCalculator:
    """1-D Eckart-like barrier — the engine.py demo, dressed in the new
    coat.  It does NOT operate on real Cartesian coordinates; it reads a
    reaction-coordinate ξ ∈ [0,1] from `state.metadata["xi"]`.  Useful as
    an architecture smoke-test that doesn't need OpenMM or PyTorch."""

    Ea: float = 1.60
    dE: float = -1.40
    name: str = "toy-barrier-1d"

    def _xi(self, state: ComplexState) -> float:
        return float(state.metadata.get("xi", 0.0))

    def _v(self, xi: float) -> float:
        barrier = self.Ea * math.exp(-((xi - 0.5) / 0.18) ** 2)
        slope = self.dE * (1.0 / (1.0 + math.exp(-10.0 * (xi - 0.5))))
        return barrier + slope

    def energy(self, state: ComplexState) -> float:
        return self._v(self._xi(state))

    def forces(self, state: ComplexState):
        eps = 1e-4
        xi = self._xi(state)
        f = -(self._v(xi + eps) - self._v(xi - eps)) / (2 * eps)
        # one-atom dummy — adapter clients should not unpack this for a
        # toy state; it's here only to satisfy the Calculator shape
        return [(f, 0.0, 0.0)]
