"""
PhiEx.core.theory — tϕ, the refined theory.

Two pieces of state make up "the current theory":

    1. FittedTheory — a small dataclass of fitted parameters from
       trajectory data.  In v0 this carries surrogate-model statistics
       (RMSE, R², per-residue importance) plus iteration counts.

    2. TheoryRegistry — the live list of LearnedModels loaded this
       session.  This is what the spec means by
       `session.theories: list[LearnedModel]`.  Iteration `i` of the
       active-learning loop updates this list (re-fit a surrogate,
       reload a fine-tuned ESM-2 head, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .learned import LearnedModel


@dataclass
class FittedTheory:
    """Dataclass mirror of engine.py's FittedTheory, generalised."""

    iteration: int = 0
    n_trials: int = 0
    n_successes: int = 0
    rmse: Optional[float] = None
    r2: Optional[float] = None
    per_residue_importance: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def success_rate(self) -> float:
        return self.n_successes / self.n_trials if self.n_trials else 0.0


class TheoryRegistry:
    """The live list of learned models active in the current session.

    Adapters register themselves on load (see the FastAPI startup hook).
    The reasoning ticker reads the registry to attribute each ML call to a
    named tϕ.  The active-learning loop replaces or refits entries here
    when iteration `i` advances.
    """

    def __init__(self) -> None:
        self._models: dict[str, LearnedModel] = {}

    def register(self, model: LearnedModel) -> None:
        self._models[model.name] = model

    def unregister(self, name: str) -> None:
        self._models.pop(name, None)

    def get(self, name: str) -> Optional[LearnedModel]:
        return self._models.get(name)

    def all(self) -> list[LearnedModel]:
        return list(self._models.values())

    def names(self) -> list[str]:
        return list(self._models.keys())

    def info(self) -> list[dict]:
        return [m.info() for m in self._models.values()]


# session-level singleton (one per process, like the ticker)
THEORIES = TheoryRegistry()
