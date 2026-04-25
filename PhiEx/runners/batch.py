"""
PhiEx.runners.batch — active-learning batch payload runner.

Stand-alone shape:

    payload = {
        "candidates": [<perturbations>],
        "n_iterations": 3,
        "acquisition": "ucb",
        "evaluator": "synthetic"     # placeholder until production wires
                                     # in real MACE-region energy delta
    }

The "synthetic" evaluator returns a deterministic-but-noisy scalar from
the perturbation features so the v0 sandbox demonstrates the AL loop
without spinning up MACE for every candidate.  The real evaluator is
wired by the FastAPI handler when running in-process.
"""

from __future__ import annotations
import json
import math
import random
import sys

from ..adapters.surrogate import GBRSurrogate, encode_perturbation
from ..al import (ActiveLearningLoop, UCBAcquisition,
                  ExpectedImprovementAcquisition, ThompsonAcquisition)
from ..ticker import log


def synthetic_evaluator(p: dict) -> tuple[float, float]:
    """A deterministic-but-stochastic toy evaluator.

    The "true" target is a smooth function of (residue_position, blosum,
    temperature) plus Gaussian noise.  The AL loop should converge to the
    perturbation that maximises the target.  Used in tests and in the
    standalone-script path.  Replace with a real evaluator (MACE region
    energy delta, residence-time MD, …) by wiring it in-process from
    FastAPI."""
    res = float(p.get("residue", 0))
    pos = res / 250.0
    target = 1.5 * math.sin(6.0 * pos) - 0.5 * math.cos(10.0 * pos) \
             + 0.05 * (pos - 0.4) ** 2
    rng = random.Random(hash((res, p.get("from"), p.get("to"))) & 0xFFFFFFFF)
    return target + rng.gauss(0, 0.1), 0.1


_ACQS = {
    "ucb": UCBAcquisition,
    "ei": ExpectedImprovementAcquisition,
    "thompson": ThompsonAcquisition,
}


def run_batch_payload(payload: dict) -> dict:
    candidates = payload["candidates"]
    n_iter = int(payload.get("n_iterations", 3))
    acq_name = payload.get("acquisition", "ucb")
    acq_cls = _ACQS.get(acq_name, UCBAcquisition)

    surrogate = GBRSurrogate()
    loop = ActiveLearningLoop(surrogate=surrogate,
                              evaluator=synthetic_evaluator,
                              acquisition=acq_cls())
    seed_n = min(5, max(2, len(candidates)//4))
    loop.seed(candidates[:seed_n])
    loop.set_candidates(candidates[seed_n:])

    results = []
    for _ in range(n_iter):
        if not loop.candidates:
            break
        r = loop.iterate()
        # remove the chosen candidate so we don't re-pick it
        loop.candidates = [c for c in loop.candidates if c != r.chosen]
        results.append(_result_to_json(r))
    return {
        "n_iterations": len(results),
        "history": results,
    }


def _result_to_json(r) -> dict:
    return {
        "iteration": r.iteration,
        "chosen": r.chosen,
        "predicted": r.predicted,
        "predicted_sd": r.predicted_sd,
        "observed": r.observed,
        "observed_sd": r.observed_sd,
        "rmse": r.surrogate_rmse,
        "r2": r.surrogate_r2,
    }


if __name__ == "__main__":   # pragma: no cover
    payload = json.load(sys.stdin)
    out = run_batch_payload(payload)
    json.dump(out, sys.stdout, indent=2)
