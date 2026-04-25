"""
PhiEx.core.learned — ϕ as ML model.

LearnedModel is the second coat in the wardrobe.  Where Calculator is
the classical physics interface (energy + forces), LearnedModel is the
generic ML interface:  `predict(input) → (output, uncertainty)`.

EVERY ML model in the sandbox wears this coat:

    ESMFold              sequence            → (Protein with plddt, sd of plddt)
    ESM-2 embedder       sequence            → (per-residue embedding, attention SD)
    P2Rank pocket model  Protein             → (list of Pocket, per-pocket score-CI)
    Vina scorer          DockingPose         → (score, 0.0)              ← deterministic
    DiffDock             Protein, Ligand     → (DockingPose, score-CI)
    MACE-OFF23 potential subset of atoms     → (energy/forces, ensemble-σ)
    Surrogate (GBR/GNN)  perturbation vector → (Δactivity, σ)            ← v0 default

The `Prediction` dataclass keeps the convention enforceable in code.  Any
adapter that returns a bare scalar without uncertainty is wrong by
construction.

Iteration `i` updates the registry of LearnedModels (re-fit, fine-tune,
posterior update).  The session's `theories: list[LearnedModel]` is
literally a list of these objects — see core.theory.TheoryRegistry.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Any, runtime_checkable, Optional


class ModelLoadError(RuntimeError):
    """Raised when a learned model cannot be loaded — checkpoint missing,
    backend not installed, etc.  Adapters MUST raise this rather than a
    generic Exception so the FastAPI layer can fall back to stub mode and
    log a clear message to the ticker."""


@dataclass
class Prediction:
    """A point estimate plus an explicit uncertainty.

    `value` is whatever the model predicts (a tensor, a list of poses, a
    Protein object …) — keep its type stable per adapter.

    `uncertainty` is in the same units as `value` for scalars.  For
    structured outputs (Protein, list of poses) it is a *summary* (mean
    pLDDT SD, ensemble σ over scores) — adapters document what.

    `meta` carries adapter-specific extras (which model variant ran, on
    which device, how many ensemble members, etc.) so the reasoning ticker
    can log them.  Required keys: "model", "device".
    """
    value: Any
    uncertainty: Any = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class LearnedModel(Protocol):
    """The single ML interface.

    Implementations live in PhiEx.adapters.  An implementation MUST:
        * carry a `name` attribute (free-form string for ticker logging)
        * carry a `device` attribute that came from PhiEx.device
        * raise ModelLoadError (not generic Exception) on missing weights
        * return Prediction (not raw values) from predict()
        * NEVER hold tool-specific types in its public input/output
    """

    name: str
    device: str
    is_stub: bool   # True if this adapter is a placeholder for a heavier real tool

    def predict(self, *args, **kwargs) -> Prediction: ...

    def info(self) -> dict:
        """Diagnostic info logged to the ticker on load."""
        ...


# ────────────────────────────────────────────────────────────────────────
# Mixin that adapter classes can use to satisfy the protocol cheaply.
# ────────────────────────────────────────────────────────────────────────

class LearnedModelBase:
    """Convenience base — adapters can inherit instead of re-implementing
    the boilerplate `info()` and `__repr__`.  Not required by the protocol,
    but every v0 adapter uses it for consistency."""

    name: str = "unnamed-learned-model"
    device: str = "cpu"
    is_stub: bool = False
    backend: str = ""        # e.g. "fair-esm 2.0.0", "stub"
    weights_path: Optional[str] = None

    def info(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "stub": self.is_stub,
            "backend": self.backend,
            "weights": self.weights_path,
        }

    def __repr__(self) -> str:  # pragma: no cover - convenience
        suffix = "  [STUB]" if self.is_stub else ""
        return f"<LearnedModel {self.name} on {self.device}{suffix}>"
