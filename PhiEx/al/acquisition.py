"""
PhiEx.al.acquisition — acquisition functions.

The default is UCB (upper confidence bound) because it's transparent, has
one knob (κ), and works perfectly with the surrogate's quantile heads.
EI (expected improvement) and Thompson sampling are stubbed at the
correct interface for swap-recipe purposes — the README documents the
swap.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.learned import Prediction


@runtime_checkable
class AcquisitionFunction(Protocol):
    """Score a candidate prediction for "should we sample this next?"."""
    name: str
    def score(self, prediction: Prediction) -> float: ...


@dataclass
class UCBAcquisition:
    """upper confidence bound:  μ + κ · σ."""
    kappa: float = 1.5
    name: str = "ucb"
    def score(self, prediction: Prediction) -> float:
        mu = float(prediction.value or 0.0)
        sd = float(prediction.uncertainty or 0.0)
        return mu + self.kappa * sd


@dataclass
class ExpectedImprovementAcquisition:
    """expected improvement over an incumbent best.

    v0 implementation uses a Gaussian assumption — cheap and correct for
    the surrogate's calibrated quantile heads.  Documented in README.
    """
    incumbent: float = 0.0
    name: str = "ei"
    def score(self, prediction: Prediction) -> float:
        import math
        mu = float(prediction.value or 0.0)
        sd = float(prediction.uncertainty or 1e-6)
        z = (mu - self.incumbent) / max(sd, 1e-6)
        # closed-form Gaussian EI
        from math import erf, sqrt
        Phi = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return (mu - self.incumbent) * Phi + sd * phi


@dataclass
class ThompsonAcquisition:
    """Thompson sampling — draw one sample from the predictive distribution.

    v0 uses the surrogate's μ ± σ as a Gaussian.  Replace with a true
    posterior sample once you swap to a Bayesian model.

    The RNG lives for the lifetime of the acquisition object so that each
    candidate gets an independent draw.  (Re-seeding per call would hand
    every candidate the same standard-normal z, collapsing Thompson
    sampling into a fixed μ + z·σ ranking.)  `seed` keeps runs reproducible.
    """
    seed: int = 0
    name: str = "thompson"
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def score(self, prediction: Prediction) -> float:
        mu = float(prediction.value or 0.0)
        sd = float(prediction.uncertainty or 0.0)
        return self._rng.gauss(mu, sd)


ACQUISITIONS = {
    "ucb": UCBAcquisition,
    "ei": ExpectedImprovementAcquisition,
    "thompson": ThompsonAcquisition,
}


def make_acquisition(name: str) -> AcquisitionFunction:
    """Build an acquisition function by its short name ("ucb", "ei",
    "thompson").  Used by the JSON-stdio batch runner and the API."""
    try:
        return ACQUISITIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown acquisition {name!r}; "
                         f"expected one of {sorted(ACQUISITIONS)}") from None
