"""
PhiEx.adapters.diffdock — ML docking (stubbed in v0).

ML role: stage 5 alternative to Vina.  DiffDock is a diffusion model that
emits an ensemble of poses with a confidence head; the ensemble naturally
provides uncertainty.

v0 ships a STUB whose interface matches the real adapter exactly, so the
swap to real DiffDock is a single-file replacement.  The README documents
the recipe.

LearnedModel:
    predict(state, ligand, pocket) -> Prediction(value=list[DockingPose],
                                                 uncertainty=score_sd)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass

from ..core.learned import LearnedModelBase, Prediction
from ..core.state import ComplexState, Pocket, Ligand, DockingPose, Atom
from ..ticker import log
from ..device import select_device


@dataclass
class DiffDockAdapter(LearnedModelBase):
    name: str = "diffdock"
    device: str = ""
    is_stub: bool = True              # v0 default: stubbed
    backend: str = "stub (DiffDock weights too heavy for v0)"
    n_poses: int = 10

    def __post_init__(self) -> None:
        if not self.device:
            self.device = select_device()
        # In v0 we don't try to import DiffDock at all — it would pull in
        # torch_geometric and ESM model weights that we explicitly defer.
        log("t", f"DiffDock adapter: {self.backend} on {self.device}")

    def predict(self, state: ComplexState, ligand: Ligand,
                pocket: Pocket) -> Prediction:
        # The shape MUST be identical to the real DiffDock adapter once
        # implemented:  return a list of DockingPose with confidence-derived
        # score_uncertainty per pose, plus aggregate uncertainty.
        rng = random.Random(0xDFD0CC ^ hash(ligand.smiles))
        cx, cy, cz = pocket.center
        poses: list[DockingPose] = []
        for k in range(self.n_poses):
            jitter = (rng.gauss(0, 1.8), rng.gauss(0, 1.8), rng.gauss(0, 1.8))
            pos = (cx + jitter[0], cy + jitter[1], cz + jitter[2])
            confidence = max(0.0, min(1.0, 0.85 - 0.05*k + rng.gauss(0, 0.05)))
            score = -confidence * 8.0                      # confidence → score
            score_sd = (1.0 - confidence) * 1.5            # uncertainty
            atoms = (
                Atom(index=0, name="C1", element="C", xyz=pos),
                Atom(index=1, name="O1", element="O", xyz=(pos[0]+1.0, pos[1], pos[2])),
            )
            poses.append(DockingPose(
                ligand=Ligand(name=ligand.name, smiles=ligand.smiles,
                              atoms=atoms, sdf_text=ligand.sdf_text),
                score=score, score_uncertainty=score_sd,
                pose_id=k, rmsd_to_best=float(0.5*k),
            ))
        poses.sort(key=lambda p: p.score)
        agg_sd = sum(p.score_uncertainty or 0.0 for p in poses) / max(1, len(poses))
        log("s", f"DiffDock[STUB] s∃: {len(poses)} poses; "
                  f"best score={poses[0].score:.2f}±{poses[0].score_uncertainty:.2f}")
        return Prediction(value=poses, uncertainty=agg_sd,
                          meta={"model": self.name, "device": self.device,
                                "backend": self.backend, "stub": True})
