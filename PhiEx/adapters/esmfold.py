"""
PhiEx.adapters.esmfold — ESMFold structure prediction.

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
import os
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
        """Load real ESMFold, or fall back to the stub.

        Order of attempts:
            0. gate on PHIEX_ESMFOLD (auto|off|force) and on physical RAM —
               ESMFold runs ESM-2 3B underneath (~12 GB fp32 weights), so on
               a 16 GB Mac `auto` keeps the stub instead of swapping hard.
            1. fair-esm `esm.pretrained.esmfold_v1()`  (needs `openfold`,
               which rarely builds on Apple Silicon)
            2. Hugging Face `transformers.EsmForProteinFolding`
               (facebook/esmfold_v1) — no openfold dependency.
        Both expose `infer_pdb(sequence) -> str`, so `_predict_real` is shared.
        """
        mode = os.environ.get("PHIEX_ESMFOLD", "auto").strip().lower()
        if mode == "off":
            self.is_stub = True
            self.backend = "stub (disabled via PHIEX_ESMFOLD=off)"
            log("t", f"ESMFold adapter: {self.backend}")
            return
        min_gb = float(os.environ.get("PHIEX_ESMFOLD_MIN_GB", "32"))
        ram_gb = _physical_ram_gb()
        if mode != "force" and ram_gb is not None and ram_gb < min_gb:
            self.is_stub = True
            self.backend = (f"stub ({ram_gb:.0f} GB RAM < {min_gb:.0f} GB needed; "
                            f"set PHIEX_ESMFOLD=force to try anyway)")
            log("t", f"ESMFold adapter: {self.backend}")
            return

        try:
            import torch  # noqa: F401
        except ImportError:
            self.is_stub = True
            self.backend = "stub (torch not installed)"
            log("t", f"ESMFold adapter: {self.backend} → predict() will synthesise")
            return

        if self.weights_path and not _weights_exist(self.weights_path):
            self.is_stub = True
            self.backend = f"stub (weights missing at {self.weights_path})"
            log("t", f"ESMFold adapter: {self.backend}")
            return

        errors: list[str] = []
        model, label = None, ""
        # 1) fair-esm
        try:
            import esm
            model = esm.pretrained.esmfold_v1()
            label = "fair-esm ESMFold v1"
        except Exception as e:
            errors.append(f"fair-esm: {e}")
        # 2) Hugging Face transformers
        if model is None:
            try:
                from transformers import EsmForProteinFolding  # type: ignore
                log("t", "ESMFold: fair-esm unavailable; loading Hugging Face "
                         "facebook/esmfold_v1 (first load downloads several GB)")
                model = EsmForProteinFolding.from_pretrained(
                    "facebook/esmfold_v1", cache_dir=_hf_cache_dir(),
                    low_cpu_mem_usage=True)
                label = "HF transformers ESMFold v1"
            except Exception as e:
                errors.append(f"transformers: {e}")

        if model is None:
            self.is_stub = True
            self.backend = f"stub (ESMFold load failed: {'; '.join(errors)})"
            log("sys", f"ESMFold load failed, falling back to stub: {'; '.join(errors)}")
            return

        model = model.eval()
        self._configure_for_device(model)
        try:
            model = model.to(self.device)
        except Exception as e:
            log("sys", f"ESMFold .to({self.device}) failed: {e}; using cpu")
            self.device = "cpu"
            model = model.float().to("cpu")
        self._model = model
        self.is_stub = False
        self.backend = f"{label} on {self.device}"
        log("t", f"ESMFold loaded: {self.backend}, "
                 f"info={describe_device(self.device)}")

    def _configure_for_device(self, model) -> None:
        """Memory savers.  The ESM-2 3B language-model trunk in fp16 halves
        its footprint on GPU/MPS (CPU keeps fp32: fp16 matmuls are slow or
        unsupported there).  Chunking the folding trunk trades speed for
        peak memory."""
        if self.device in ("mps", "cuda") and hasattr(model, "esm"):
            try:
                model.esm = model.esm.half()
            except Exception as e:  # pragma: no cover
                log("sys", f"ESMFold: fp16 trunk not applied ({e})")
        trunk = getattr(model, "trunk", None)
        if trunk is not None and hasattr(trunk, "set_chunk_size"):
            trunk.set_chunk_size(int(os.environ.get("PHIEX_ESMFOLD_CHUNK", "64")))

    def predict_chain(self, protein: Protein, name: str = "predicted") -> Prediction:
        """Fold ONE chain of `protein`.

        `Protein.sequence` from a PDB entry is every chain concatenated
        (1APX: 4 × 249 = 996 residues).  ESMFold would fold that as a single
        996-residue chain — slow and physically meaningless.  We fold the
        first chain (or PHIEX_ESMFOLD_CHAIN) and record which one, so the
        RMSD compares like with like.
        """
        chain, seq = select_chain_sequence(
            protein, os.environ.get("PHIEX_ESMFOLD_CHAIN") or None)
        if chain and len(seq) < len(protein.sequence or ""):
            log("phi", f"ESMFold: folding chain {chain} only "
                       f"({len(seq)} of {len(protein.sequence)} residues)")
        pred = self.predict(seq, name=name)
        if chain:
            pred.value.metadata["query_chain"] = chain
            pred.meta["query_chain"] = chain
        return pred

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
        try:
            with torch.no_grad():
                output = self._model.infer_pdb(sequence)
        except Exception as e:
            if self.device == "cpu":
                raise
            log("sys", f"ESMFold forward on {self.device} failed "
                       f"({type(e).__name__}: {e}); retrying on cpu")
            self.device = "cpu"
            self._model = self._model.float().to("cpu")
            self.backend = self.backend.rsplit(" on ", 1)[0] + " on cpu"
            with torch.no_grad():
                output = self._model.infer_pdb(sequence)
        output = _rescale_plddt_if_fractional(output)
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
    lines = [f"REMARK   1 PHIEX STUB  {name}"]
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
    """Cα RMSD after optimal superposition (Kabsch).

    Pairing:
        * if most predicted residues share (chain, index) keys with the
          experimental structure (the stub copies them), pair by key;
        * otherwise (real ESMFold numbers its single chain A 1..N), pair
          by position along the folded chain — `metadata["query_chain"]`,
          or the first experimental chain.
    Real predictions live in an arbitrary frame, so superposition is
    required; without it the number would be tens of Å regardless of
    prediction quality.
    """
    if not pred.residues or not exp.residues:
        return None
    pred_by = {(r.chain, r.index): r for r in pred.residues}
    exp_by = {(r.chain, r.index): r for r in exp.residues}
    common = sorted(set(pred_by) & set(exp_by))
    pairs: list[tuple[Atom, Atom]] = []
    if len(common) >= 0.9 * len(pred.residues):
        for k in common:
            a, b = pred_by[k].ca, exp_by[k].ca
            if a and b:
                pairs.append((a, b))
    else:
        chain = (pred.metadata or {}).get("query_chain") or exp.residues[0].chain
        exp_chain = [r for r in exp.residues if r.chain == chain]
        for rp, re_ in zip(pred.residues, exp_chain):
            if rp.ca and re_.ca:
                pairs.append((rp.ca, re_.ca))
    if len(pairs) < 3:
        return None
    return _kabsch_rmsd([p.xyz for p, _ in pairs], [e.xyz for _, e in pairs])


def _kabsch_rmsd(P, Q) -> float:
    import numpy as np
    P = np.asarray(P, dtype=float); Q = np.asarray(Q, dtype=float)
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    diff = (R @ P.T).T - Q
    return float(np.sqrt((diff ** 2).sum() / len(P)))


# ────────────────────────────────────────────────────────────────────────
# chain / environment helpers
# ────────────────────────────────────────────────────────────────────────

_THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}


def select_chain_sequence(protein: Protein,
                          chain: Optional[str] = None) -> "tuple[Optional[str], str]":
    """(chain_id, one-letter sequence) for one chain of a structure.
    Sequence-only proteins return (None, protein.sequence)."""
    if not protein.residues:
        return None, protein.sequence or ""
    chains: list[str] = []
    for r in protein.residues:
        if r.chain not in chains:
            chains.append(r.chain)
    pick = chain if chain in chains else chains[0]
    seq = "".join(_THREE_TO_ONE.get(r.name.upper(), "X")
                  for r in protein.residues if r.chain == pick)
    return pick, seq


def _physical_ram_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 1e9
    except Exception:
        return None


def _hf_cache_dir() -> Optional[str]:
    """Same place `make weights` puts Hugging Face downloads."""
    if os.environ.get("HF_HOME"):
        return None                      # let transformers honour HF_HOME
    models = os.environ.get("PHIEX_MODELS_DIR")
    return os.path.join(os.path.abspath(models), "hf", "hub") if models else None


def _rescale_plddt_if_fractional(pdb_text: str) -> str:
    """Some ESMFold builds write pLDDT as 0–1 in the B-factor column.
    The rest of PhiEx (and the Mol* colouring) expects 0–100."""
    rows = [l for l in pdb_text.splitlines() if l.startswith("ATOM")]
    try:
        bmax = max(float(l[60:66]) for l in rows) if rows else 100.0
    except ValueError:
        return pdb_text
    if bmax > 1.0:
        return pdb_text
    out = []
    for l in pdb_text.splitlines():
        if l.startswith("ATOM") and len(l) >= 66:
            l = l[:60] + f"{float(l[60:66]) * 100:6.2f}" + l[66:]
        out.append(l)
    return "\n".join(out) + "\n"
