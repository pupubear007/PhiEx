"""
PhiEx.adapters.foldseek — structural homology search.

ML role: stage 2 (s — study).  Find structural neighbors of the target
in AlphaFoldDB / PDB and aggregate their GO terms into a "functional
hypothesis ϕ0" that the UI displays as the starting theory in the left
panel.

Real Foldseek is a CLI tool (`foldseek easy-search`).  Wrapping it as a
LearnedModel is the right shape because we need *uncertainty* on its
hits — the real interface returns per-hit E-values that we treat as
score uncertainty.  The v0 stub returns canned plant-peroxidase neighbors
when called on APX, so the rest of the pipeline can still wire up.

Swap recipe (in README):
    Replace `_run_foldseek_cli` with a real subprocess call once the
    user has the database installed locally.  Everything else stays.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ..core.learned import LearnedModelBase, Prediction
from ..core.state import Protein
from ..ticker import log
from ..device import select_device


# canned neighbours for the APX example (plant peroxidase family)
_APX_STUB_HITS = [
    {"id": "1OAG", "evalue": 1e-90, "tm": 0.78,
     "go": ["F:peroxidase activity", "F:heme binding", "P:response to oxidative stress"]},
    {"id": "1H58", "evalue": 4e-72, "tm": 0.71,
     "go": ["F:peroxidase activity", "F:heme binding"]},
    {"id": "1ARU", "evalue": 8e-65, "tm": 0.66,
     "go": ["F:peroxidase activity"]},
    {"id": "1B82", "evalue": 1e-58, "tm": 0.63,
     "go": ["F:heme binding", "F:metal ion binding"]},
    {"id": "2CCA", "evalue": 2e-55, "tm": 0.61,
     "go": ["F:peroxidase activity", "P:hydrogen peroxide catabolic process"]},
]


@dataclass
class FoldseekAdapter(LearnedModelBase):
    name: str = "foldseek"
    device: str = "cpu"           # foldseek is CPU; flagged for log clarity
    is_stub: bool = True          # almost always stubbed in v0
    backend: str = "stub"
    database: str = "afdb50"

    def __post_init__(self) -> None:
        if not self.device:
            self.device = select_device()
        self._maybe_load()

    def _maybe_load(self) -> None:
        # Foldseek is a binary; we just check it's on PATH
        import shutil
        if shutil.which("foldseek"):
            self.is_stub = False
            self.backend = "foldseek-cli"
            log("t", f"foldseek detected on PATH (database={self.database})")
        else:
            self.is_stub = True
            self.backend = "stub (foldseek not on PATH)"
            log("t", f"foldseek adapter: {self.backend}")

    def predict(self, protein: Protein, k: int = 5) -> Prediction:
        if self.is_stub:
            return self._predict_stub(protein, k)
        return self._predict_real(protein, k)

    def _predict_stub(self, protein: Protein, k: int) -> Prediction:
        # match by sequence-prefix similarity to APX as a coarse proxy
        if "PEROXID" in protein.name.upper() or protein.name.upper().startswith("APX") \
           or protein.name.upper().startswith("1APX") \
           or "MGKSY" in (protein.sequence or "")[:20]:
            hits = _APX_STUB_HITS[:k]
        else:
            hits = []
        # uncertainty: spread of E-values (in log10 space) — wider spread
        # = less consensus among neighbours
        if hits:
            import math
            logs = [math.log10(max(h["evalue"], 1e-300)) for h in hits]
            mean = sum(logs)/len(logs)
            sd = (sum((x-mean)**2 for x in logs) / max(1, len(logs)-1)) ** 0.5
        else:
            sd = 0.0
        log("s", f"Foldseek[STUB] ∃: {len(hits)} structural neighbours  log10(E) σ={sd:.2f}")
        # aggregate GO terms with simple frequency counting
        bag: dict[str, int] = {}
        for h in hits:
            for g in h["go"]:
                bag[g] = bag.get(g, 0) + 1
        return Prediction(
            value={"hits": hits, "go_aggregate": bag},
            uncertainty=sd,
            meta={"model": self.name, "device": self.device,
                  "backend": self.backend, "stub": True},
        )

    def _predict_real(self, protein: Protein, k: int):
        # Real Foldseek path stub — wire up when user has DB installed.
        # We MUST keep this shape so the swap is local.
        log("s", "foldseek real path is reserved for post-v0 — falling through to stub")
        return self._predict_stub(protein, k)


def aggregate_go_terms(prediction: Prediction, top_n: int = 8) -> "list[tuple[str, int]]":
    """Pull the GO frequency table out of a Foldseek prediction."""
    bag = (prediction.value or {}).get("go_aggregate", {})
    return sorted(bag.items(), key=lambda kv: -kv[1])[:top_n]
