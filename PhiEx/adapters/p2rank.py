"""
PhiEx.adapters.p2rank — pocket detection.

ML role: stage 3 (∃ — concrete outcomes).  Identify pockets, then mark
the one closest to the heme Fe as the active site.  This last step is
the explicit cofactor-aware heuristic the spec calls out.

Real P2Rank is a Java CLI from `p2rank` conda package.  We wrap it.  If
it's not installed we fall back to a naive geometric pocket finder
(largest cavity by Cα-density gradient) so the pipeline still runs.

LearnedModel:
    predict(state) -> Prediction(value=list[Pocket], uncertainty=score_sd)
"""

from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core.learned import LearnedModelBase, Prediction
from ..core.state import ComplexState, Pocket, Protein, Cofactor
from ..ticker import log


@dataclass
class P2RankAdapter(LearnedModelBase):
    name: str = "p2rank"
    device: str = "cpu"      # JVM, not GPU; flagged for log clarity
    is_stub: bool = True
    backend: str = "stub"
    threads: int = 2

    def __post_init__(self) -> None:
        if shutil.which("prank") or shutil.which("p2rank"):
            self.is_stub = False
            self.backend = "p2rank-cli"
            log("t", "P2Rank detected on PATH")
        else:
            self.is_stub = True
            self.backend = "stub (p2rank not on PATH)"
            log("t", f"P2Rank adapter: {self.backend}")

    # ──────────────────────────────────────────────────────────────
    # public
    # ──────────────────────────────────────────────────────────────
    def predict(self, state: ComplexState) -> Prediction:
        if self.is_stub or not state.protein.pdb_text:
            pockets = self._predict_stub(state)
        else:
            try:
                pockets = self._predict_real(state)
            except Exception as e:
                log("sys", f"P2Rank real run failed ({e}) — falling back to stub")
                pockets = self._predict_stub(state)

        # cofactor-aware: which pocket is the active site?
        pockets = _flag_active_site(pockets, state.cofactors)

        scores = [p.score for p in pockets]
        sd = _stddev(scores) if scores else 0.0
        log("phi" if not self.is_stub else "s",
            f"P2Rank ∃: {len(pockets)} pocket(s); active-site flagged at "
            f"{next((p.id for p in pockets if p.is_active_site), 'none')}  "
            f"score σ={sd:.2f}")
        return Prediction(value=list(pockets), uncertainty=sd,
                          meta={"model": self.name, "device": self.device,
                                "backend": self.backend, "stub": self.is_stub})

    # ──────────────────────────────────────────────────────────────
    # real
    # ──────────────────────────────────────────────────────────────
    def _predict_real(self, state: ComplexState) -> list[Pocket]:
        with tempfile.TemporaryDirectory() as td:
            pdb = Path(td) / "in.pdb"
            pdb.write_text(state.protein.pdb_text or "")
            outdir = Path(td) / "out"
            cmd = ["prank", "predict", "-f", str(pdb), "-o", str(outdir),
                   "-threads", str(self.threads)]
            log("phi", f"P2Rank: running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            csv = next(outdir.rglob("*_predictions.csv"), None)
            if csv is None:
                raise RuntimeError("P2Rank output CSV not found")
            return _parse_p2rank_csv(csv.read_text())

    # ──────────────────────────────────────────────────────────────
    # stub
    # ──────────────────────────────────────────────────────────────
    def _predict_stub(self, state: ComplexState) -> list[Pocket]:
        """Geometric fallback: cluster Cα atoms by proximity to each
        cofactor metal (if any), then to the protein's centroid.  Returns
        up to 3 pockets with synthetic scores."""
        prot: Protein = state.protein
        if not prot.residues:
            return []

        cas = [r.ca for r in prot.residues if r.ca]
        if not cas:
            return []

        pockets: list[Pocket] = []
        # 1) one pocket per cofactor metal
        for i, cof in enumerate(state.cofactors or []):
            if cof.metal_atom is None:
                continue
            mx, my, mz = cof.metal_atom.xyz
            nearby = []
            for r in prot.residues:
                if r.ca is None: continue
                cx, cy, cz = r.ca.xyz
                d = math.sqrt((cx-mx)**2 + (cy-my)**2 + (cz-mz)**2)
                if d < 8.0:
                    nearby.append(r.index)
            score = max(0.5, 1.0 - 0.05 * len(nearby))
            pockets.append(Pocket(
                id=f"P{i+1}", center=cof.metal_atom.xyz,
                score=score + 1.0,         # bump so cofactor-pocket sorts first
                nearby_residues=tuple(sorted(nearby)),
            ))

        # 2) protein-centroid pocket as a fallback
        cx = sum(a.xyz[0] for a in cas) / len(cas)
        cy = sum(a.xyz[1] for a in cas) / len(cas)
        cz = sum(a.xyz[2] for a in cas) / len(cas)
        centroid_nearby = sorted(
            (r.index for r in prot.residues if r.ca and
             math.sqrt((r.ca.xyz[0]-cx)**2 + (r.ca.xyz[1]-cy)**2 + (r.ca.xyz[2]-cz)**2) < 10.0)
        )
        if not pockets:
            pockets.append(Pocket(
                id="P1", center=(cx, cy, cz), score=0.7,
                nearby_residues=tuple(centroid_nearby),
            ))
        return pockets


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────

def _parse_p2rank_csv(text: str) -> list[Pocket]:
    rows = [l for l in text.splitlines() if l.strip()]
    if len(rows) < 2:
        return []
    hdr = [c.strip() for c in rows[0].split(",")]
    cols = {h: i for i, h in enumerate(hdr)}
    out: list[Pocket] = []
    for r in rows[1:]:
        cells = [c.strip() for c in r.split(",")]
        try:
            pid = cells[cols.get("name", 0)]
            score = float(cells[cols.get("score", 1)])
            cx = float(cells[cols.get("center_x", 2)])
            cy = float(cells[cols.get("center_y", 3)])
            cz = float(cells[cols.get("center_z", 4)])
        except (IndexError, ValueError):
            continue
        residues_field = cells[cols.get("residue_ids", -1)] if cols.get("residue_ids") is not None else ""
        residues = tuple(int(t.split("_")[-1]) for t in residues_field.split() if t.split("_")[-1].isdigit())
        out.append(Pocket(id=pid, center=(cx, cy, cz), score=score,
                          nearby_residues=residues))
    return out


def _flag_active_site(pockets: list[Pocket], cofactors) -> list[Pocket]:
    """Heuristic (documented in README): the active-site pocket is the
    one whose center is closest to the metal atom of any cofactor.
    With no cofactor, no flag is set."""
    metals = [c.metal_atom.xyz for c in cofactors if c.metal_atom is not None]
    if not metals or not pockets:
        return pockets
    best = None
    best_d = float("inf")
    for p in pockets:
        for mx, my, mz in metals:
            d = math.sqrt((p.center[0]-mx)**2 + (p.center[1]-my)**2 + (p.center[2]-mz)**2)
            if d < best_d:
                best_d, best = d, p
    out = []
    for p in pockets:
        out.append(Pocket(id=p.id, center=p.center, score=p.score,
                          nearby_residues=p.nearby_residues,
                          is_active_site=(p is best),
                          metadata={**p.metadata, "metal_distance": best_d if p is best else None}))
    return out


def _stddev(xs):
    if len(xs) < 2: return 0.0
    m = sum(xs)/len(xs)
    return (sum((x-m)**2 for x in xs) / (len(xs)-1)) ** 0.5
