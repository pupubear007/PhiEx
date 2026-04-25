"""
phytologue.al — Bayesian active-learning loop (i: s∃ ⟳ tϕ).

This is where the framework earns its keep.  Each iteration:

    1. The surrogate (tϕ) ranks candidate perturbations by an acquisition
       function (UCB by default, swap-targets EI / Thompson sampling).
    2. The top-k go into a sampled-existence batch (s∃) — each is run
       through a cheap simulation (MACE-only on the active-site region in
       v0; full MD in production) to produce a (perturbation, scalar)
       pair.
    3. The surrogate re-fits on the augmented dataset (tϕ ← s∃).
    4. Iteration i increments.

The UI exposes "run next iteration (i ← i+1)" with the chosen perturbation
shown explicitly: "tϕ suggests: mutate R38 → K, predicted Δactivity = 0.3 ± 0.4".
"""

from .acquisition import (
    AcquisitionFunction, UCBAcquisition,
    ExpectedImprovementAcquisition, ThompsonAcquisition,
)
from .loop import ActiveLearningLoop, ALIterationResult, generate_mutation_panel

__all__ = [
    "AcquisitionFunction", "UCBAcquisition",
    "ExpectedImprovementAcquisition", "ThompsonAcquisition",
    "ActiveLearningLoop", "ALIterationResult", "generate_mutation_panel",
]
