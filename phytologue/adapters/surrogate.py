"""
phytologue.adapters.surrogate — tϕ refined by induction.

The surrogate is the ML model the active-learning loop fits and refits:
it takes a *perturbation* (a small structured edit — mutation, ligand
variant, temperature) and predicts a cheap-to-compute scalar outcome
(binding free-energy estimate, residence time, MACE energy of the
active-site core, …) with calibrated uncertainty.

v0 default:  scikit-learn GradientBoostingRegressor with quantile heads
at 0.16 / 0.50 / 0.84 (≈ ±1σ) so the AL loop has an explicit interval.
README documents the swap to a small GNN once you have torch_geometric.

Perturbation encoding (explicit heuristic, in README):
    A perturbation is a dict like
        {"type": "mutation", "residue": 38, "from": "R", "to": "K"}
        {"type": "ligand_variant", "smiles": "..."}
        {"type": "temperature", "K": 320}
    The encoder concatenates a few features:
        * one-hot of perturbation type
        * residue index normalised by sequence length
        * BLOSUM62 substitution score for mutations (cached table)
        * temperature offset from 300K
        * RDKit Morgan fingerprint length-256 if a ligand SMILES is given
    Anything missing is zero-padded.  Crude on purpose; replace with a
    learned encoder later.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.learned import LearnedModelBase, Prediction
from ..ticker import log
from ..device import select_device


# ────────────────────────────────────────────────────────────────────────
# encoding
# ────────────────────────────────────────────────────────────────────────

_AA_INDEX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-")}
# tiny BLOSUM62 substring (diagonal-only + a few representative off-diags)
_BLOSUM62 = {
    ("A","A"):4,("R","R"):5,("N","N"):6,("D","D"):6,("C","C"):9,("Q","Q"):5,
    ("E","E"):5,("G","G"):6,("H","H"):8,("I","I"):4,("L","L"):4,("K","K"):5,
    ("M","M"):5,("F","F"):6,("P","P"):7,("S","S"):4,("T","T"):5,("W","W"):11,
    ("Y","Y"):7,("V","V"):4,
    ("R","K"):2,("K","R"):2,
    ("D","E"):2,("E","D"):2,
    ("F","Y"):3,("Y","F"):3,
    ("S","T"):1,("T","S"):1,
    ("I","V"):3,("V","I"):3,("I","L"):2,("L","I"):2,("L","V"):1,("V","L"):1,
    ("A","S"):1,("S","A"):1,
}


def _morgan_256(smiles: str) -> "list[float]":
    """Morgan fingerprint of length 256, RDKit if available, otherwise a
    deterministic hash-based fallback so the surrogate keeps working."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [0.0]*256
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        return [float(b) for b in fp]
    except Exception:
        out = [0.0]*256
        for i, ch in enumerate(smiles or ""):
            out[(i * 31 + ord(ch)) % 256] = 1.0
        return out


def encode_perturbation(p: dict, sequence_length: int = 250,
                        ligand_smiles: str = "") -> "list[float]":
    """Public encoder used by the AL loop and the surrogate adapter."""
    type_onehot = [0.0]*4
    types = ["mutation", "ligand_variant", "temperature", "other"]
    type_onehot[types.index(p.get("type", "other")) if p.get("type") in types else 3] = 1.0

    res_pos = float(p.get("residue", 0)) / max(1, sequence_length)
    blosum = 0.0
    if p.get("type") == "mutation":
        f = p.get("from", "X").upper(); t = p.get("to", "X").upper()
        blosum = _BLOSUM62.get((f, t), 0)
    temp_off = (float(p.get("K", 300.0)) - 300.0) / 50.0

    smi = p.get("smiles", "") or ligand_smiles
    fp = _morgan_256(smi) if smi else [0.0]*256

    return type_onehot + [res_pos, blosum, temp_off] + fp


# ────────────────────────────────────────────────────────────────────────
# surrogate adapter
# ────────────────────────────────────────────────────────────────────────

@dataclass
class GBRSurrogate(LearnedModelBase):
    name: str = "surrogate-gbr"
    device: str = "cpu"          # sklearn is CPU-only; flagged for clarity
    is_stub: bool = False        # this one always works (sklearn in env)
    backend: str = "scikit-learn GradientBoostingRegressor (quantiles)"
    sequence_length: int = 250

    # estimators per quantile head
    _q16: object = None
    _q50: object = None
    _q84: object = None
    _trained: bool = False
    _x_buffer: list = field(default_factory=list)
    _y_buffer: list = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        except ImportError:
            self.is_stub = True
            self.backend = "stub (scikit-learn missing)"
            log("t", f"Surrogate adapter: {self.backend}")
            return
        log("t", f"Surrogate adapter ready: {self.backend}")

    def fit(self, X: "list[list[float]]", y: "list[float]") -> "FitReport":
        """Fit (or refit) the surrogate on ALL accumulated (X, y) pairs.

        We do NOT support warm-start incrementally: re-fitting from scratch
        with each iteration is the simplest correct shape for the AL loop
        and matches sklearn's API.  Bigger datasets later → swap to GNN
        and you get incremental fine-tuning for free.
        """
        if self.is_stub:
            return self._fit_stub(X, y)
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        self._x_buffer = list(X)
        self._y_buffer = list(y)
        n = len(X)
        if n < 3:
            log("t", f"surrogate: only {n} samples — fit deferred")
            return FitReport(n_samples=n, rmse=None, r2=None, deferred=True)

        params = dict(n_estimators=120, max_depth=3, learning_rate=0.05,
                      random_state=0)
        self._q50 = GradientBoostingRegressor(loss="squared_error", **params).fit(X, y)
        self._q16 = GradientBoostingRegressor(loss="quantile", alpha=0.16, **params).fit(X, y)
        self._q84 = GradientBoostingRegressor(loss="quantile", alpha=0.84, **params).fit(X, y)
        pred = self._q50.predict(X)
        rmse = mean_squared_error(y, pred) ** 0.5
        r2 = r2_score(y, pred)
        self._trained = True
        log("t", f"surrogate tϕ refit: n={n}  RMSE={rmse:.3f}  R²={r2:.3f}")
        return FitReport(n_samples=n, rmse=rmse, r2=r2)

    def _fit_stub(self, X, y):
        self._x_buffer = list(X); self._y_buffer = list(y)
        self._trained = len(X) >= 1
        # naive mean-of-y predictor
        log("t", f"surrogate[STUB] tϕ refit: n={len(X)} (mean fallback)")
        return FitReport(n_samples=len(X), rmse=None, r2=None, stub=True)

    def predict(self, perturbation: dict) -> Prediction:
        x = encode_perturbation(perturbation, self.sequence_length)
        if not self._trained:
            return Prediction(value=0.0, uncertainty=1.0,
                              meta={"model": self.name, "device": self.device,
                                    "untrained": True})
        if self.is_stub or self._q50 is None:
            mean = sum(self._y_buffer)/len(self._y_buffer) if self._y_buffer else 0.0
            sd = _stddev(self._y_buffer) if len(self._y_buffer) > 1 else 1.0
            return Prediction(value=mean, uncertainty=sd,
                              meta={"model": self.name, "device": self.device,
                                    "stub": True})
        import numpy as np
        x_arr = np.asarray(x).reshape(1, -1)
        mean = float(self._q50.predict(x_arr)[0])
        lo = float(self._q16.predict(x_arr)[0])
        hi = float(self._q84.predict(x_arr)[0])
        sd = max(1e-3, (hi - lo) / 2.0)
        return Prediction(value=mean, uncertainty=sd,
                          meta={"model": self.name, "device": self.device,
                                "lo": lo, "hi": hi})


@dataclass
class FitReport:
    n_samples: int
    rmse: Optional[float]
    r2: Optional[float]
    deferred: bool = False
    stub: bool = False


def _stddev(xs):
    if len(xs) < 2: return 0.0
    m = sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs) / (len(xs)-1)) ** 0.5
