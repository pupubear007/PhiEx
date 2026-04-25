"""
phytologue — visual sandbox for in planta / in vitro biomolecular simulation.

Framework (Wang 2025):

    ϕ   theory          a Calculator OR a LearnedModel that defines an outcome
                        from a state (energy/forces, embedding, prediction).
    ∃   existence       a State of the system (atoms, sequence, embedding).
    s∃  study           a Trajectory — a sampled sequence of States.
    tϕ  refined theory  a FittedTheory — model fit from many Trajectories
                        (in v0 this is a gradient-boosting surrogate over
                        perturbation → outcome pairs, plus the registry of
                        learned models that have been loaded this session).
    i   iteration       a step of the Bayesian active-learning loop.
                        s∃ ⟳ tϕ : sampled existence cycles refined theory.
    ↑↓  polarity        deductive (ϕ → ∃) vs inductive (s∃ ⟳ tϕ).

This package is laid out around that vocabulary.

    phytologue.core/        State, Calculator, LearnedModel, Trajectory, FittedTheory
    phytologue.adapters/    one thin wrapper per external tool / ML model
    phytologue.analysis/    contact maps, RMSF, residence time
    phytologue.al/          Bayesian active-learning loop (UCB by default)
    phytologue.runners/     in-process AND JSON-stdio runners (Slurm-ready)
    phytologue.pipeline/    end-to-end pipelines (APX is the v0 example)
    phytologue.device       single source of truth for device selection
    phytologue.ticker       reasoning-ticker logger that streams to the UI

Architectural rule:  every external tool wears the same coat as engine.py's
ToyBarrier wears Calculator.  Adapters are the architecture; do not bypass
them even where a direct call would be shorter.
"""

__version__ = "0.1.0"
