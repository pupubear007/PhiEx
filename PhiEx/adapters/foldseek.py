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

Real path:
    `foldseek easy-search` of ONE chain of the query structure against a
    local Foldseek database (default: the PDB database that
    `make weights` / `foldseek databases PDB ...` puts under
    models/foldseek/pdb; override with PHIEX_FOLDSEEK_DB).  Hits are PDB
    entries, so each is mapped to GO terms via the PDBe SIFTS API
    (cached in models/foldseek/go_cache.json).  If the binary or the
    database is missing we stay on the stub and say why.
"""

from __future__ import annotations
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
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
        exe = shutil.which("foldseek")
        db = _foldseek_db_path()
        if not exe:
            self.is_stub = True
            self.backend = "stub (foldseek not on PATH)"
        elif not _db_exists(db):
            self.is_stub = True
            self.backend = (f"stub (foldseek found but no database at {db}; "
                            f"run `make weights` or set PHIEX_FOLDSEEK_DB)")
        else:
            self.is_stub = False
            self.database = db
            self.backend = f"foldseek-cli ({Path(db).name})"
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
        query_pdb, chain = _single_chain_pdb(protein)
        if not query_pdb:
            log("sys", "foldseek: query has no coordinates — using stub")
            return self._predict_stub(protein, k)
        try:
            rows = self._run_easy_search(query_pdb)
        except Exception as e:
            log("sys", f"foldseek run failed ({e}) — falling back to stub")
            return self._predict_stub(protein, k)

        self_id = (protein.name or "")[:4].lower()
        best: dict[str, dict] = {}
        for target, evalue, tm, prob in rows:
            pid = _pdb_id_from_target(target)
            if not pid or pid == self_id:
                continue                       # drop the query's own entry
            if pid not in best or evalue < best[pid]["evalue"]:
                best[pid] = {"id": pid.upper(), "target": target,
                             "evalue": evalue, "tm": round(tm, 3), "prob": prob}
        hits = sorted(best.values(), key=lambda h: h["evalue"])[:k]
        go_map = _go_terms_for([h["id"] for h in hits])
        for h in hits:
            h["go"] = go_map.get(h["id"].lower(), [])

        logs = [math.log10(max(h["evalue"], 1e-300)) for h in hits]
        sd = 0.0
        if len(logs) > 1:
            m = sum(logs) / len(logs)
            sd = (sum((x - m) ** 2 for x in logs) / (len(logs) - 1)) ** 0.5
        bag: dict[str, int] = {}
        for h in hits:
            for g in h["go"]:
                bag[g] = bag.get(g, 0) + 1
        log("s", f"Foldseek ∃: {len(hits)} structural neighbours of chain {chain} "
                 f"({len(rows)} raw hits)  log10(E) σ={sd:.2f}")
        return Prediction(
            value={"hits": hits, "go_aggregate": bag},
            uncertainty=sd,
            meta={"model": self.name, "device": self.device,
                  "backend": self.backend, "query_chain": chain},
        )

    def _run_easy_search(self, query_pdb: str) -> "list[tuple[str, float, float, float]]":
        threads = str(os.environ.get("PHIEX_FOLDSEEK_THREADS", os.cpu_count() or 4))
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "query.pdb"
            q.write_text(query_pdb)
            out = Path(td) / "hits.m8"
            cmd = ["foldseek", "easy-search", str(q), self.database, str(out),
                   str(Path(td) / "tmp"),
                   "--format-output", "target,evalue,alntmscore,prob",
                   "-e", "1e-3", "--threads", threads]
            log("s", f"foldseek: easy-search vs {Path(self.database).name} "
                     f"({threads} threads)")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                raise RuntimeError(f"exit {proc.returncode}: {' | '.join(tail)}")
            rows = []
            for line in out.read_text().splitlines():
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                try:
                    rows.append((cols[0], float(cols[1]), float(cols[2]), float(cols[3])))
                except ValueError:
                    continue
            return rows


# ────────────────────────────────────────────────────────────────────────
# helpers for the real path
# ────────────────────────────────────────────────────────────────────────

def _models_dir() -> Path:
    return Path(os.environ.get("PHIEX_MODELS_DIR", "./models")).resolve()


def _foldseek_db_path() -> str:
    return os.environ.get("PHIEX_FOLDSEEK_DB") or str(_models_dir() / "foldseek" / "pdb")


def _db_exists(db: str) -> bool:
    # a Foldseek DB is a family of files: <db>, <db>.dbtype, <db>.index, ...
    return Path(db + ".dbtype").exists() or Path(db).exists()


def _single_chain_pdb(protein: Protein) -> "tuple[str, Optional[str]]":
    """ATOM records of the first chain only, so a homotetramer like 1APX
    is searched once rather than four times."""
    text = protein.pdb_text or ""
    atoms = [l for l in text.splitlines() if l.startswith("ATOM")]
    if not atoms:
        return "", None
    chain = atoms[0][21] if len(atoms[0]) > 21 else " "
    keep = [l for l in atoms if len(l) > 21 and l[21] == chain]
    return "\n".join(keep) + "\nEND\n", chain.strip() or None


_PDB_ID_RE = re.compile(r"^(\d[a-z0-9]{3})", re.IGNORECASE)


def _pdb_id_from_target(target: str) -> Optional[str]:
    """Foldseek PDB targets look like '1oag-assembly1.cif.gz_A' or '1oag_A'."""
    m = _PDB_ID_RE.match(target.strip())
    return m.group(1).lower() if m else None


_GO_PREFIX = {"molecular_function": "F", "biological_process": "P",
              "cellular_component": "C"}


def _go_terms_for(pdb_ids: "list[str]") -> "dict[str, list[str]]":
    """PDB id → ['F:peroxidase activity', ...] via PDBe SIFTS, cached on disk.
    Network failures leave a hit with no GO terms rather than failing."""
    cache_path = _models_dir() / "foldseek" / "go_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except Exception:
        cache = {}
    changed = False
    for pid in {p.lower() for p in pdb_ids}:
        if pid in cache:
            continue
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/go/{pid}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.load(r)
            terms = []
            entry = data.get(pid) or data.get(pid.upper()) or {}
            for _go_id, info in (entry.get("GO", {}) or {}).items():
                if not isinstance(info, dict):
                    continue
                cat = _GO_PREFIX.get(str(info.get("category", "")).lower().replace(" ", "_"), "?")
                label = info.get("name") or info.get("identifier") or _go_id
                terms.append(f"{cat}:{label}")
            cache[pid] = sorted(set(terms))
            changed = True
        except Exception as e:
            log("sys", f"foldseek: GO lookup for {pid} failed ({e})")
    if changed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
        except Exception:
            pass
    return cache


def aggregate_go_terms(prediction: Prediction, top_n: int = 8) -> "list[tuple[str, int]]":
    """Pull the GO frequency table out of a Foldseek prediction."""
    bag = (prediction.value or {}).get("go_aggregate", {})
    return sorted(bag.items(), key=lambda kv: -kv[1])[:top_n]
