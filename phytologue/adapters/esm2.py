"""
phytologue.adapters.esm2 — ESM-2 sequence embeddings + attention.

ML role: stage 2 (s — study).  Compute per-residue embeddings used as
input to the surrogate, plus self-attention maps used to corroborate
trajectory-flagged functional residues.

LearnedModel:
    predict(sequence) -> Prediction(value=embeddings, uncertainty=ensemble_sd)

The v0 default checkpoint is `esm2_t12_35M_UR50D` (the smallest released
variant) so the sandbox is responsive on CPU/MPS.  The README documents
how to swap to esm2_t33_650M_UR50D on a GPU node.

Stub: same policy as ESMFold — if the package or weights are missing,
emit synthetic embeddings so downstream code keeps working.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.learned import LearnedModelBase, Prediction
from ..ticker import log
from ..device import select_device


@dataclass
class ESM2Adapter(LearnedModelBase):
    name: str = "esm2-t12-35M"
    device: str = ""
    variant: str = "esm2_t12_35M_UR50D"   # v0 default; swap to t33 650M on GPU
    is_stub: bool = False
    backend: str = ""
    embedding_dim: int = 480              # 480 for t12, 1280 for t33
    _model: object = None
    _alphabet: object = None
    _batch_converter: object = None

    def __post_init__(self) -> None:
        if not self.device:
            self.device = select_device()
        self._maybe_load()

    def _maybe_load(self) -> None:
        try:
            import esm
            import torch
        except ImportError:
            self.is_stub = True
            self.backend = "stub (esm package not installed)"
            log("t", f"ESM-2 adapter: {self.backend}")
            return
        try:
            import esm
            loader = getattr(esm.pretrained, self.variant)
            model, alphabet = loader()
            model = model.eval()
            try:
                model = model.to(self.device)
            except Exception as e:
                log("sys", f"ESM-2 .to({self.device}) failed: {e}; using cpu")
                self.device = "cpu"
                model = model.to("cpu")
            self._model = model
            self._alphabet = alphabet
            self._batch_converter = alphabet.get_batch_converter()
            self.embedding_dim = model.embed_dim
            self.backend = f"fair-esm {self.variant} on {self.device}"
            log("t", f"ESM-2 loaded: {self.backend} (dim={self.embedding_dim})")
        except Exception as e:
            self.is_stub = True
            self.backend = f"stub (load failed: {e})"
            log("sys", f"ESM-2 load failed, falling back to stub: {e}")

    def predict(self, sequence: str, name: str = "target") -> Prediction:
        if self.is_stub or self._model is None:
            return self._predict_stub(sequence, name)
        return self._predict_real(sequence, name)

    def _predict_real(self, sequence: str, name: str) -> Prediction:
        import torch
        with torch.no_grad():
            data = [(name, sequence)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self.device)
            n_layers = getattr(self._model, "num_layers", 12)
            out = self._model(tokens, repr_layers=[n_layers],
                              need_head_weights=True)
            reps = out["representations"][n_layers]
            # drop BOS/EOS
            emb = reps[0, 1:-1, :].cpu().numpy()
            # attention: shape (layers, heads, L, L) — take last layer mean over heads
            attn = out["attentions"][0, -1].mean(0)[1:-1, 1:-1].cpu().numpy()
        # uncertainty proxy: SD of per-residue embedding norm (NOT Bayesian
        # — documented as a heuristic in README)
        norms = (emb**2).sum(axis=1) ** 0.5
        sd = float(norms.std())
        log("s", f"ESM-2 ∃: per-residue embedding {emb.shape}  norm σ={sd:.3f}")
        return Prediction(
            value={"embedding": emb, "attention": attn},
            uncertainty=sd,
            meta={"model": self.name, "device": self.device,
                  "backend": self.backend, "dim": self.embedding_dim},
        )

    def _predict_stub(self, sequence: str, name: str) -> Prediction:
        rng = random.Random(hash(sequence) & 0xFFFFFFFF)
        L = len(sequence)
        D = self.embedding_dim
        # plausible embedding: amino-acid-keyed seed offsets so the same
        # residue gets similar vectors; lets the surrogate still learn
        # *something* in stub mode
        aa_seed = {aa: rng.gauss(0, 1) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        try:
            import numpy as np  # type: ignore
        except ImportError:
            log("sys", "ESM-2 stub: numpy missing; returning empty Prediction")
            return Prediction(value={"embedding": [], "attention": []},
                              uncertainty=0.0,
                              meta={"model": self.name, "device": self.device,
                                    "stub": True})
        emb = np.zeros((L, D), dtype="float32")
        for i, aa in enumerate(sequence):
            base = aa_seed.get(aa, 0.0)
            for d in range(D):
                emb[i, d] = base + rng.gauss(0, 0.3)
        # attention: a smooth band-diagonal toy with peaks at local-contact range
        attn = np.zeros((L, L), dtype="float32")
        for i in range(L):
            for j in range(L):
                d = abs(i - j)
                attn[i, j] = math.exp(-d / 6.0) + 0.1 * rng.random()
        attn = attn / attn.sum(axis=1, keepdims=True)
        sd = float((emb**2).sum(axis=1).std())
        log("s", f"ESM-2[STUB] ∃: synthetic embedding L={L} D={D}  σ={sd:.3f}")
        return Prediction(
            value={"embedding": emb, "attention": attn},
            uncertainty=sd,
            meta={"model": self.name, "device": self.device,
                  "backend": self.backend, "stub": True, "dim": D},
        )


def attention_residue_importance(attention) -> "list[float]":
    """Reduce an L×L attention map to a per-residue importance score.

    Heuristic: column-wise mean of the last-layer mean-over-heads attention.
    This is what we cross-check trajectory-flagged residues against.
    Documented in the README as a known simplification — proper attention
    probing would require per-head supervised analysis.
    """
    try:
        import numpy as np
        a = np.asarray(attention)
        return list(a.mean(axis=0).astype(float))
    except Exception:
        return []
