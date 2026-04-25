"""
phytologue.al.loop — the AL outer loop.

Public API:

    loop = ActiveLearningLoop(surrogate, acquisition, evaluator)
    loop.seed(initial_perturbations)            # first ~5 cheap runs
    result = loop.iterate()                     # one i ← i+1 step

`evaluator` is any callable `(perturbation: dict) -> (y: float, sigma: float)`.
For the v0 APX example we plug in a tiny MACE-region energy difference
(see phytologue.pipeline.apx).  Replace with full MD residence-time once
the user has the compute.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.learned import Prediction
from ..ticker import log
from .acquisition import AcquisitionFunction, UCBAcquisition


Evaluator = Callable[[dict], "tuple[float, float]"]


@dataclass
class ALIterationResult:
    iteration: int
    chosen: dict
    predicted: float
    predicted_sd: float
    observed: float
    observed_sd: float
    surrogate_rmse: Optional[float]
    surrogate_r2: Optional[float]
    converged_residues: list[int] = field(default_factory=list)


@dataclass
class ActiveLearningLoop:
    surrogate: object                       # GBRSurrogate (or compatible)
    evaluator: Evaluator
    acquisition: AcquisitionFunction = field(default_factory=UCBAcquisition)
    candidates: list[dict] = field(default_factory=list)
    history: list[ALIterationResult] = field(default_factory=list)
    iteration: int = 0

    # ──────────────────────────────────────────────────────────────
    def seed(self, perturbations: list[dict]) -> None:
        """Run the seed batch (≈5 in v0) before iteration 1.  These pairs
        give the surrogate a starting tϕ.  Logged with i=0."""
        log("i", f"───── iteration i = 0 (seed batch, n={len(perturbations)}) ─────")
        X, y = [], []
        from ..adapters.surrogate import encode_perturbation
        for p in perturbations:
            obs, sd = self.evaluator(p)
            x = encode_perturbation(p)
            X.append(x); y.append(obs)
            log("s", f"seed run: {self._render_pert(p)} → y={obs:.3f} ± {sd:.3f}")
        report = self.surrogate.fit(X, y)
        log("t", f"tϕ seeded: surrogate fit on {report.n_samples} samples")

    # ──────────────────────────────────────────────────────────────
    def set_candidates(self, candidates: list[dict]) -> None:
        self.candidates = list(candidates)

    # ──────────────────────────────────────────────────────────────
    def iterate(self) -> ALIterationResult:
        """One s∃ ⟳ tϕ step."""
        self.iteration += 1
        log("i", f"───── iteration i = {self.iteration} ─────")
        if not self.candidates:
            raise RuntimeError("no candidates to score; call set_candidates first")

        # 1) score every candidate
        scored: list[tuple[float, dict, Prediction]] = []
        for p in self.candidates:
            pred = self.surrogate.predict(p)
            s = self.acquisition.score(pred)
            scored.append((s, p, pred))
        scored.sort(key=lambda r: -r[0])     # higher score = more interesting
        score, chosen, predicted = scored[0]
        log("t", f"tϕ suggests: {self._render_pert(chosen)}, "
                  f"predicted Δ = {predicted.value:.3f} ± {predicted.uncertainty:.3f} "
                  f"(acq={self.acquisition.name} score={score:.3f})")

        # 2) actually evaluate the chosen one (the s∃ part of s∃ ⟳ tϕ)
        observed, observed_sd = self.evaluator(chosen)
        log("s", f"observed: y = {observed:.3f} ± {observed_sd:.3f}")

        # 3) refit
        from ..adapters.surrogate import encode_perturbation
        x = encode_perturbation(chosen)
        if hasattr(self.surrogate, "_x_buffer"):
            X = self.surrogate._x_buffer + [x]
            y = self.surrogate._y_buffer + [observed]
        else:
            X, y = [x], [observed]
        report = self.surrogate.fit(X, y)
        log("t", f"tϕ refit: n={report.n_samples}  "
                  f"RMSE={report.rmse}  R²={report.r2}")

        # 4) record + return
        result = ALIterationResult(
            iteration=self.iteration, chosen=chosen,
            predicted=float(predicted.value),
            predicted_sd=float(predicted.uncertainty or 0.0),
            observed=observed, observed_sd=observed_sd,
            surrogate_rmse=report.rmse, surrogate_r2=report.r2,
        )
        self.history.append(result)
        return result

    # ──────────────────────────────────────────────────────────────
    def _render_pert(self, p: dict) -> str:
        if p.get("type") == "mutation":
            return f"mutate {p.get('from','?')}{p.get('residue','?')} → {p.get('to','?')}"
        if p.get("type") == "ligand_variant":
            return f"ligand variant {p.get('smiles','?')[:24]}"
        if p.get("type") == "temperature":
            return f"T = {p.get('K')} K"
        return str(p)


# ────────────────────────────────────────────────────────────────────────
# helper: small candidate panels around the active site
# ────────────────────────────────────────────────────────────────────────

def generate_mutation_panel(active_site_residues: list[int],
                            sequence: str,
                            substitutions: list[str] = None) -> list[dict]:
    """Build a small mutation panel for the AL loop.  By default we
    mutate every active-site residue to A, K, and the conservative
    substitute (R→K, D→E, F→Y …).  Documented in README as the v0
    recipe — replace with a learned generator later."""
    subs = substitutions or ["A", "K"]
    panel: list[dict] = []
    conservative = {"R":"K","K":"R","D":"E","E":"D","F":"Y","Y":"F",
                    "S":"T","T":"S","I":"V","V":"I","L":"V","M":"I"}
    for resid in active_site_residues:
        if resid - 1 < 0 or resid - 1 >= len(sequence):
            continue
        wt = sequence[resid - 1].upper()
        for to in subs:
            if to != wt:
                panel.append({"type": "mutation", "residue": resid,
                              "from": wt, "to": to})
        cons = conservative.get(wt)
        if cons and cons not in subs:
            panel.append({"type": "mutation", "residue": resid,
                          "from": wt, "to": cons})
    return panel
