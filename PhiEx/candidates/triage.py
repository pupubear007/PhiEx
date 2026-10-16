"""
PhiEx.candidates.triage — structure-first triage of a candidate gene list.

For every gene:

    1. classify the annotation          → track (dock / structure / unsuited),
                                           expected cofactor, ligand panel
    2. gene ID → UniProt accession       (STRING id, cache, or UniProt search)
    3. best available structure          experimental PDB  >  AlphaFill (when
                                           the site needs a cofactor)  >
                                           AlphaFold  >  SWISS-MODEL
    4. site                              cofactor / co-crystal ligand /
                                           catalytic triad / pocket (site.py)
    5. confidence                        mean pLDDT, and pLDDT of the site
                                           residues (AlphaFold models)
    6. (optional) dock the class panel   → best binder vs negative control

The output row keeps the manuscript columns (log2FC per time point, source
tables) next to the structural ones, so the table can be sorted either by
the biology or by how testable a candidate is.  Rows are cached on disk
(data/candidates/triage_cache.json) per gene and mode, so a long list can be
stopped and resumed.
"""

from __future__ import annotations
import csv
import io
import json
import math
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..ticker import log
from .classes import classify, CandidateClass
from .resolve import resolve_uniprot

TRACK_ORDER = {"dock": 0, "structure": 1, "unsuited": 2}


# ────────────────────────────────────────────────────────────────────────
# input
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    gene_id: str
    description: str = ""
    family: str = ""
    uniprot: str = ""
    organism: str = ""
    sources: str = ""
    log2fc: tuple = (None, None, None)       # 24, 48, 96 hpi
    extra: dict = field(default_factory=dict)


_GENE_RE = re.compile(r"(SS1G_\d+|Glyma\.\d\dG\d+|Sscle\d\dg\d+)", re.IGNORECASE)


def _norm_gene(g: str) -> str:
    m = re.match(r"^glyma\.(\d\d)g(\d+)$", g, re.IGNORECASE)
    if m:
        return f"Glyma.{m.group(1)}G{m.group(2)}"
    m = re.match(r"^ss1g_(\d+)$", g, re.IGNORECASE)
    if m:
        return f"SS1G_{m.group(1)}"
    return g


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace("−", "-").replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def parse_candidates(text: str) -> list[Candidate]:
    """A TSV/CSV with a header (gene_id / Gene ID column) or one gene ID per
    line, optionally followed by a description."""
    text = (text or "").strip()
    if not text:
        return []
    first = text.splitlines()[0]
    out: list[Candidate] = []
    if "\t" in first or ("," in first and re.search(r"gene", first, re.I)):
        delim = "\t" if "\t" in first else ","
        for r in csv.DictReader(io.StringIO(text), delimiter=delim):
            low = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            gid = next((low[k] for k in low if k.startswith("gene")), "")
            m = _GENE_RE.search(gid)
            if not m:
                continue
            gid = _norm_gene(m.group(1))
            out.append(Candidate(
                gene_id=gid, description=low.get("description", ""), family=low.get("family", ""),
                uniprot=low.get("uniprot", "") or low.get("string_id", ""),
                organism=low.get("organism", ""), sources=low.get("sources", ""),
                log2fc=tuple(_num(low.get(f"log2fc_{t}")) for t in ("24", "48", "96")),
                extra={k: v for k, v in low.items() if k in ("group", "clade", "sig_tp", "secreted")}))
        return out
    seen = set()
    for line in text.splitlines():
        m = _GENE_RE.search(line)
        if not m:
            continue
        gid = _norm_gene(m.group(1))
        if gid in seen:
            continue
        seen.add(gid)
        rest = line[m.end():].strip(" \t,;|-")
        out.append(Candidate(gene_id=gid, description=rest))
    return out


# ────────────────────────────────────────────────────────────────────────
# structure choice
# ────────────────────────────────────────────────────────────────────────

def choose_structure(acc: str, cls: CandidateClass) -> tuple[Optional[str], str]:
    """(fetch id, description) — experimental first; AlphaFill when the
    site needs a cofactor that a bare AlphaFold model would lack."""
    from ..adapters.pdb import list_pdbs_for_uniprot
    entries = list_pdbs_for_uniprot(acc)
    exp = [e for e in entries if ":" not in e["pdb_id"]]
    if exp:
        e = exp[0]
        res = f" {e['resolution']:.2f} Å" if e.get("resolution") else ""
        return e["pdb_id"], f"{e.get('method') or 'experimental'}{res}"
    by_prefix = {e["pdb_id"].split(":")[0]: e for e in entries if ":" in e["pdb_id"]}
    wants_cofactor = bool(cls.cofactor) and "none" not in cls.cofactor.lower()
    if wants_cofactor and "AFILL" in by_prefix:
        return by_prefix["AFILL"]["pdb_id"], "AlphaFill (AlphaFold + transplanted cofactors)"
    if "AF" in by_prefix:
        return by_prefix["AF"]["pdb_id"], "AlphaFold"
    if "SWISS" in by_prefix:
        return by_prefix["SWISS"]["pdb_id"], "SWISS-MODEL"
    if "AFILL" in by_prefix:
        return by_prefix["AFILL"]["pdb_id"], "AlphaFill"
    return None, "no structure or model found"


def plddt_stats(protein, residue_ids=()) -> tuple[Optional[float], Optional[float]]:
    """Mean pLDDT over Cα atoms (model B-factor column) and over the given
    residues.  Returns (None, None) for experimental structures."""
    if not (protein.name or "").upper().startswith(("AF:", "AFILL:")):
        return None, None            # experimental B-factors / SWISS-MODEL QMEAN are not pLDDT
    cas = [(r.index, r.ca.bfactor) for r in protein.residues if r.ca]
    if not cas:
        return None, None
    mean = statistics.fmean(b for _, b in cas)
    if mean <= 1.0:                       # some files store pLDDT as 0–1
        cas = [(i, b * 100) for i, b in cas]
        mean *= 100
    site = [b for i, b in cas if i in set(residue_ids)]
    return round(mean, 1), (round(statistics.fmean(site), 1) if site else None)


# ────────────────────────────────────────────────────────────────────────
# one gene
# ────────────────────────────────────────────────────────────────────────

def _max_abs(fcs) -> Optional[float]:
    vals = [v for v in fcs if v is not None]
    return max(vals, key=abs) if vals else None


def triage_one(c: Candidate, dock: bool = False, exhaustiveness: int = 8) -> dict:
    from ..adapters.pdb import fetch_protein
    from ..adapters.p2rank import P2RankAdapter
    from ..analysis.site import resolve_active_site, choose_active_pocket
    from ..core.state import ComplexState
    from ..pipeline.apx import resolve_working_structure

    cls = classify(c.description, c.family)
    peak = _max_abs(c.log2fc)
    row = {
        "gene_id": c.gene_id, "organism": c.organism or ("S. sclerotiorum" if c.gene_id.startswith("SS1G")
                                                         else "G. max" if c.gene_id.startswith("Glyma") else ""),
        "description": c.description, "family": c.family, "sources": c.sources,
        "log2fc_24": c.log2fc[0], "log2fc_48": c.log2fc[1], "log2fc_96": c.log2fc[2],
        "peak_log2fc": peak,
        # sign follows the contrast of the input table (e.g. treatment vs reference)
        "direction": None if peak is None else ("up" if peak > 0 else "down"),
        "class": cls.key, "class_title": cls.title, "track": cls.track,
        "cofactor_needed": cls.cofactor, "note": cls.note,
        "uniprot": None, "structure": None, "structure_kind": None, "n_residues": None,
        "mean_plddt": None, "site": None, "site_source": None, "site_confidence": None,
        "site_residues": [], "site_plddt": None, "hetero": "",
        "panel": [{"label": p.label, "value": p.value, "role": p.role} for p in cls.panel],
        "docked": False, "best_ligand": None, "best_role": None, "best_score": None,
        "best_efficiency": None, "control_score": None, "margin": None, "best_site_hits": [],
        "screen": None, "warnings": [], "error": None,
    }
    if cls.track == "unsuited":
        row["warnings"].append("not a small-molecule docking target")
    try:
        acc, how = resolve_uniprot(c.gene_id, c.uniprot)
        row["uniprot"] = acc
        if not acc:
            row["error"] = f"no UniProt accession ({how})"
            return row
        struct_id, kind = choose_structure(acc, cls)
        row["structure"], row["structure_kind"] = struct_id, kind
        if not struct_id:
            row["error"] = kind
            return row
        protein, cofactors, _ann = fetch_protein(struct_id, acc)
        if not protein.residues:
            row["error"] = f"{struct_id}: structure download failed"
            return row
        if protein.name and protein.name.upper() != struct_id.upper():
            row["warnings"].append(f"{struct_id} unavailable — used {protein.name} (no transplanted cofactor)")
            row["structure"], row["structure_kind"] = protein.name, "AlphaFold (AlphaFill unavailable)"
        protein, cofactors = resolve_working_structure(protein, cofactors, None)
        row["n_residues"] = len([r for r in protein.residues if r.chain == protein.residues[0].chain])
        het = {}
        for co in cofactors:
            k = f"{co.name}" + (f" ({co.metadata.get('role')})" if co.metadata.get("role") not in (None, "cofactor") else "")
            het[k] = het.get(k, 0) + 1
        row["hetero"] = ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in het.items())
        if "cofactor" in cls.cofactor.lower() or "use alphafill" in cls.cofactor.lower():
            if not any(co.metadata.get("role") in ("cofactor", "metal") for co in cofactors):
                row["warnings"].append(f"site cofactor absent from this structure ({cls.cofactor})")

        site = resolve_active_site(protein, cofactors, None)
        p2 = P2RankAdapter()
        state = ComplexState(protein=protein, cofactors=tuple(cofactors))
        pockets = p2.predict(state).value
        active, site, _ = choose_active_pocket(protein, pockets, site, real_detector=not p2.is_stub)
        row.update(site=site.label, site_source=site.source, site_confidence=site.confidence,
                   site_residues=list(active.nearby_residues) if active else [])
        mean_pl, site_pl = plddt_stats(protein, row["site_residues"])
        row["mean_plddt"], row["site_plddt"] = mean_pl, site_pl
        if site_pl is not None and site_pl < 70:
            row["warnings"].append(f"site pLDDT {site_pl:.0f} < 70: site geometry is low-confidence")
        if site.confidence == "low":
            row["warnings"].append("site from a fallback pocket — give an anchor or install P2Rank")

        if dock and cls.track == "dock" and cls.panel:
            from ..pipeline.screen import screen_ligands
            state = ComplexState(protein=protein, cofactors=tuple(cofactors),
                                 metadata={"site_center": site.center})
            roles = {p.label: p.role for p in cls.panel}
            out = screen_ligands(state, active, site, [(p.label, p.value) for p in cls.panel],
                                 exhaustiveness=exhaustiveness)
            for r in out["rows"]:
                r["role"] = roles.get(r.get("name"), roles.get(r.get("input"), ""))
            ok = [r for r in out["rows"] if not r["error"]]
            binders = [r for r in ok if r["role"] != "control"]
            controls = [r for r in ok if r["role"] == "control"]
            row["docked"] = bool(ok)
            row["screen"] = out["rows"]
            if binders:
                b = min(binders, key=lambda r: r["score"])
                row.update(best_ligand=b["name"], best_role=b["role"], best_score=b["score"],
                           best_efficiency=b["efficiency"], best_site_hits=b["site_hits"])
            if controls:
                row["control_score"] = min(r["score"] for r in controls)
            if row["best_score"] is not None and row["control_score"] is not None:
                row["margin"] = round(row["control_score"] - row["best_score"], 2)
            failed = [r for r in out["rows"] if r["error"]]
            if failed:
                row["warnings"].append(f"{len(failed)} panel ligand(s) failed: "
                                       + "; ".join(r["name"] for r in failed[:3]))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# ────────────────────────────────────────────────────────────────────────
# a whole list, in a background thread
# ────────────────────────────────────────────────────────────────────────

# bump when site/structure logic changes so cached rows are recomputed
TRIAGE_VERSION = 2


def _cache_file() -> Path:
    return Path("data/candidates/triage_cache.json")


def _load_cache() -> dict:
    try:
        return json.loads(_cache_file().read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _cache_file().parent.mkdir(parents=True, exist_ok=True)
        _cache_file().write_text(json.dumps(cache))
    except Exception:
        pass


@dataclass
class TriageJob:
    candidates: list
    dock: bool = False
    exhaustiveness: int = 8
    use_cache: bool = True
    rows: list = field(default_factory=list)
    done: int = 0
    running: bool = False
    stop_requested: bool = False
    started: float = 0.0
    finished: float = 0.0

    def run(self) -> None:
        self.running, self.started = True, time.time()
        cache = _load_cache() if self.use_cache else {}
        log("i", f"── candidate triage: {len(self.candidates)} gene(s), "
                 f"{'with' if self.dock else 'without'} docking ──")
        try:
            for c in self.candidates:
                if self.stop_requested:
                    log("sys", "candidate triage stopped by user")
                    break
                key = f"{c.gene_id}|dock={int(self.dock)}|ex={self.exhaustiveness}|v={TRIAGE_VERSION}"
                row = cache.get(key) if self.use_cache else None
                if row is None:
                    row = triage_one(c, dock=self.dock, exhaustiveness=self.exhaustiveness)
                    # keep the manuscript columns from *this* input even when cached
                    if not row.get("error"):
                        cache[key] = row
                        _save_cache(cache)
                else:
                    row = {**row, "log2fc_24": c.log2fc[0], "log2fc_48": c.log2fc[1],
                           "log2fc_96": c.log2fc[2], "peak_log2fc": _max_abs(c.log2fc),
                           "sources": c.sources or row.get("sources", "")}
                self.rows.append(row)
                self.done += 1
                log("s", f"triage {self.done}/{len(self.candidates)}: {c.gene_id} → {row['class']} "
                         f"[{row['track']}] {row.get('structure') or ''} "
                         f"{'site ' + row['site'] if row.get('site') else ''}"
                         f"{'  margin ' + format(row['margin'], '.2f') if row.get('margin') is not None else ''}"
                         f"{'  ERROR ' + row['error'] if row.get('error') else ''}")
        finally:
            self.running, self.finished = False, time.time()
            log("i", f"── triage done: {self.done} gene(s) in {self.finished - self.started:.0f} s ──")

    def start(self) -> None:
        threading.Thread(target=self.run, daemon=True).start()

    def status(self) -> dict:
        return {"running": self.running, "done": self.done, "total": len(self.candidates),
                "dock": self.dock, "elapsed": round((self.finished or time.time()) - self.started, 1)
                if self.started else 0.0, "rows": sort_rows(self.rows)}


def sort_rows(rows: list[dict]) -> list[dict]:
    """Testable first (dock track with a positive margin), then by |log2FC|."""
    def key(r):
        m = r.get("margin")
        return (TRACK_ORDER.get(r.get("track"), 3), r.get("error") is not None,
                -(m if m is not None else -99), -abs(r.get("peak_log2fc") or 0))
    return sorted(rows, key=key)


CSV_COLS = ["gene_id", "organism", "class_title", "track", "description", "family", "sources",
            "log2fc_24", "log2fc_48", "log2fc_96", "uniprot", "structure", "structure_kind",
            "n_residues", "mean_plddt", "site", "site_source", "site_confidence", "site_plddt",
            "hetero", "cofactor_needed", "best_ligand", "best_role", "best_score", "best_efficiency",
            "control_score", "margin", "best_site_hits", "warnings", "note", "error"]


def rows_to_csv(rows: list[dict]) -> str:
    def cell(v):
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v)
        s = "" if v is None else str(v)
        return '"' + s.replace('"', '""') + '"' if any(ch in s for ch in ',"\n') else s
    lines = [",".join(CSV_COLS)]
    lines += [",".join(cell(r.get(c)) for c in CSV_COLS) for r in rows]
    return "\n".join(lines) + "\n"
