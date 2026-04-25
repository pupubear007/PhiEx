"""
phytologue.adapters.esmfold — ESMFold structure prediction.

ML role: this is the parallel ϕ-path of stage 1.  Where adapters/pdb.py
fetches an experimental structure, ESMFold predicts one from sequence
alone.  Both run in parallel; the UI compares them and reports per-residue
pLDDT and overall RMSD.

LearnedModel:
    predict(sequence: str) -> Prediction(value=Protein, uncertainty=plddt_sd)

Stub policy:
    Real ESMFold weights are ~3 GB and take 30 s+ on M-series CPU/MPS for
    a 250-residue protein.  If `esm` isn't installed or the weights are
    missing, we return a synthetic prediction:  copy the experimental
    structure if available, perturb coordinates by a random ~0.5 Å vector
    per residue, and produce per-residue pLDDT samples drawn from a
    confidence-shaped distribution.  Logged loudly as [STUB] in the ticker.

This stub matches the *interface* of real ESMFold — the rest of the
pipeline cannot tell the difference.  Replacing the stub is a single-file
swap; the README documents the recipe.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from ..core.learned import LearnedModelBase, Prediction, ModelLoadError
from ..core.state import Protein, Atom, Residue
from ..ticker import log
from ..device import select_device, describe_device


@dataclass
class ESMFoldAdapter(LearnedModelBase):
    name: str = "esmfold"
    device: str = ""
    weights_path: Optional[str] = None
    backend: str = ""
    is_stub: bool = False
    _model: object = None
    # Optional reference structure used by the stub to generate
    # plausible coordinates from real backbone topology.
    _reference: Optional[Protein] = None

    def __post_init__(self) -> None:
        if not self.device:
            self.device = select_device()
        self._maybe_load()

    def _maybe_load(self) -> None:
        try:
            import esm  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.is_stub = True
            self.backend = "stub (esm package not installed)"
            log("t", f"ESMFold adapter: {self.backend} → predict() will synthesise")
            return

        if self.weights_path and not _weights_exist(self.weights_path):
            self.is_stub = True
            self.backend = f"stub (weights missing at {self.weights_path})"
            log("t", f"ESMFold adapter: {self.backend}")
            return

        try:
            import esm
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            try:
                model = model.to(self.device)
            except Exception as e:
                log("sys", f"ESMFold .to({self.device}) failed: {e}; using cpu")
                self.device = "cpu"
                model = model.to("cpu")
            self._model = model
            self.backend = f"fair-esm ESMFold v1 on {self.device}"
            log("t", f"ESMFold loaded: {self.backend}, "
                     f"info={describe_device(self.device)}")
        except Exception as e:
            self.is_stub = True
            self.backend = f"stub (ESMFold load failed: {e})"
            log("sys", f"ESMFold load failed, falling back to stub: {e}")

    def set_reference(self, reference: Protein) -> None:
        """Optionally provide an experimental structure so the stub path
        can reuse its backbone geometry."""
        self._reference = reference

    def predict(self, sequence: str, name: str = "predicted") -> Prediction:
        """Predict a Protein from sequence.  Always returns a Prediction
        with `value: Protein` and `uncertainty: float` (mean pLDDT SD)."""
        if not sequence:
            raise ValueError("ESMFoldAdapter.predict: empty sequence")
        if self.is_stub or self._model is None:
            return self._predict_stub(sequence, name)
        return self._predict_real(sequence, name)

    # ──────────────────────────────────────────────────────────────
    # real path
    # ──────────────────────────────────────────────────────────────
    def _predict_real(self, sequence: str, name: str) -> Prediction:
        import torch
        log("phi", f"ESMFold: predicting {len(sequence)}-residue structure on {self.device}")
        with torch.no_grad():
            output = self._model.infer_pdb(sequence)
        # output is PDB text. parse with our PDB adapter so we stay in core types
        from .pdb import parse_pdb_text
        protein, _ = parse_pdb_text(output, name=name)
        # ESMFold writes pLDDT into the bfactor column
        plddt = tuple(_mean_residue_bfactor(r) for r in protein.residues)
        protein = Protein(
            name=name, sequence=sequence,
            residues=protein.residues, source="esmfold",
            plddt=plddt, pdb_text=output,
            metadata={"backend": self.backend},
        )
        sd = _stddev(plddt) if plddt else 0.0
        mean = sum(plddt)/len(plddt) if plddt else 0.0
        log("phi", f"ESMFold ∃: predicted structure  pLDDT = {mean:.1f} ± {sd:.1f}")
        return Prediction(value=protein, uncertainty=sd,
                          meta={"model": self.name, "device": self.device,
                                "mean_plddt": mean, "backend": self.backend})

    # ──────────────────────────────────────────────────────────────
    # stub path
    # ──────────────────────────────────────────────────────────────
    def _predict_stub(self, sequence: str, name: str) -> Prediction:
        """Synthetic prediction.  If a reference structure is available we
        copy its backbone and add ~0.5 Å Gaussian noise per atom; otherwise
        we lay residues out along an alpha-helical axis."""
        rng = random.Random(hash(sequence) & 0xFFFFFFFF)
        residues: list[Residue] = []
        plddt: list[float] = []
        if self._reference and self._reference.residues:
            ref = self._reference
            for i, r in enumerate(ref.residues[:len(sequence)]):
                jittered = []
                for a in r.atoms:
                    dx = rng.gauss(0, 0.5)
                    dy = rng.gauss(0, 0.5)
                    dz = rng.gauss(0, 0.5)
                    jittered.append(Atom(
                        index=a.index, name=a.name, element=a.element,
                        xyz=(a.xyz[0]+dx, a.xyz[1]+dy, a.xyz[2]+dz),
                        bfactor=a.bfactor,
                    ))
                # confidence high in the core, lower at termini and for noisier residues
                conf = 90.0 - 20.0 * (1 - math.exp(-((i - len(ref.residues)/2)/(len(ref.residues)/3))**2))
                conf = max(20.0, min(95.0, conf + rng.gauss(0, 5)))
                plddt.append(conf)
                residues.append(Residue(r.index, r.name, r.chain, tuple(jittered)))
        else:
            # geometric placeholder: helical CA trace
            for i, aa in enumerate(sequence):
                rise, radius, twist = 1.5, 2.3, math.radians(100)
                x = radius * math.cos(twist * i)
                y = radius * math.sin(twist * i)
                z = rise * i
                ca = Atom(index=i, name="CA", element="C", xyz=(x, y, z), bfactor=70.0)
                residues.append(Residue(index=i+1, name=_one_to_three(aa),
                                        chain="A", atoms=(ca,)))
                plddt.append(70.0 + rng.gauss(0, 8))

        plddt_t = tuple(max(0.0, min(100.0, p)) for p in plddt)
        residues_t = tuple(residues)
        # serialise to a minimal PDB text so the rest of the pipeline (P2Rank,
        # the FastAPI /api/structure route, the Mol* viewer) has something to
        # consume.  Real ESMFold returns PDB text natively.
        stub_pdb = _serialise_pdb_minimal(residues_t, plddt_t, name)
        protein = Protein(
            name=name, sequence=sequence, residues=residues_t,
            source="esmfold-stub", plddt=plddt_t,
            pdb_text=stub_pdb,
            metadata={"backend": self.backend, "stub": True},
        )
        sd = _stddev(plddt_t) if plddt_t else 0.0
        mean = sum(plddt_t)/len(plddt_t) if plddt_t else 0.0
        log("phi", f"ESMFold[STUB] ∃: synthetic structure  pLDDT = {mean:.1f} ± {sd:.1f}  "
                  f"(swap recipe in README)")
        return Prediction(value=protein, uncertainty=sd,
                          meta={"model": self.name, "device": self.device,
                                "mean_plddt": mean, "backend": self.backend,
                                "stub": True})


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────

def _weights_exist(path: str) -> bool:
    import os
    return os.path.exists(path)


def _mean_residue_bfactor(r: Residue) -> float:
    if not r.atoms:
        return 0.0
    return sum(a.bfactor for a in r.atoms) / len(r.atoms)


def _stddev(xs: tuple[float, ...]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


_ONE_TO_THREE = {v: k for k, v in {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}.items()}


def _one_to_three(aa: str) -> str:
    return _ONE_TO_THREE.get(aa.upper(), "GLY")


def _serialise_pdb_minimal(residues: tuple[Residue, ...],
                            plddt: tuple[float, ...],
                            name: str) -> str:
    """Tiny PDB writer for the stub path.  Writes one ATOM record per atom
    and stashes pLDDT in the bfactor column where ESMFold normally writes it.
    Sufficient for Mol* rendering and any downstream ATOM-record parser."""
    lines = [f"REMARK   1 PHYTOLOGUE STUB  {name}"]
    atom_serial = 1
    for ri, r in enumerate(residues):
        b = plddt[ri] if ri < len(plddt) else 0.0
        for a in r.atoms:
            x, y, z = a.xyz
            lines.append(
                f"ATOM  {atom_serial:>5d} {a.name:<4s} {r.name:<3s} "
                f"{r.chain:>1s}{r.index:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{b:>6.2f}"
                f"          {a.element:>2s}"
            )
            atom_serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────────
# overlay metric — RMSD between predicted and experimental Cα traces
# ────────────────────────────────────────────────────────────────────────

def ca_rmsd(pred: Protein, exp: Protein) -> Optional[float]:
    """Cα RMSD over residues present in both structures (no alignment).

    For the v0 sandbox we don't run a full superposition; this is a
    sanity-check metric, not a benchmark.  README documents this
    simplification.
    """
    if not pred.residues or not exp.residues:
        return None
    pred_by = {(r.chain, r.index): r for r in pred.residues}
    exp_by = {(r.chain, r.index): r for r in exp.residues}
    common = sorted(set(pred_by) & set(exp_by))
    if not common:
        return None
    sq_sum, n = 0.0, 0
    for k in common:
        ca_p, ca_e = pred_by[k].ca, exp_by[k].ca
        if not (ca_p and ca_e): continue
        dx = ca_p.xyz[0] - ca_e.xyz[0]
        dy = ca_p.xyz[1] - ca_e.xyz[1]
        dz = ca_p.xyz[2] - ca_e.xyz[2]
        sq_sum += dx*dx + dy*dy + dz*dz
        n += 1
    return math.sqrt(sq_sum / n) if n else None
