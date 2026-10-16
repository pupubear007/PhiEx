"""
PhiEx.candidates.resolve — gene IDs from a transcriptomics table → UniProt.

    SS1G_10565          S. sclerotiorum 1980 locus tag  (UniProt organism 665079)
    Glyma.08G218600     soybean Wm82 locus              (UniProt ORF name GLYMA_08G218600, taxon 3847)
    A7EYZ9 / 665079.A7EYZ9   already an accession (e.g. a STRING id)

Lookups go to the UniProt REST API and are cached on disk
(data/candidates/uniprot_map.json) so a 700-gene list is resolved once.
Wm82.a6 models that were split or renumbered after the UniProt import
(IDs ending in a non-"00" suffix such as Glyma.06G302712) often have no
UniProt entry; they are reported, not guessed.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from ..ticker import log

_ACC_RE = re.compile(r"^(?:\d+\.)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_TAXA = {"SS1G": 665079, "Glyma": 3847}


def accession_from(text: str) -> Optional[str]:
    """'A7EYZ9' or '665079.A7EYZ9' → 'A7EYZ9'; anything else → None."""
    m = _ACC_RE.match((text or "").strip())
    return m.group(1) if m else None


def _cache_path() -> Path:
    return Path("data/candidates/uniprot_map.json")


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        _cache_path().parent.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(json.dumps(cache, indent=1))
    except Exception:
        pass


def _queries(gene_id: str) -> list[str]:
    g = gene_id.strip()
    if g.upper().startswith("SS1G_"):
        return [f"(gene:{g}) AND (organism_id:665079)", f"{g} AND (organism_id:665079)"]
    m = re.match(r"^Glyma\.(\d\d)[Gg](\d+)$", g)
    if m:
        orf = f"GLYMA_{m.group(1)}G{m.group(2)}"
        return [f"(gene:{orf}) AND (organism_id:3847)", f"{orf} AND (organism_id:3847)",
                f"\"{g}\" AND (organism_id:3847)"]
    return [g]


def resolve_uniprot(gene_id: str, hint: str = "") -> tuple[Optional[str], str]:
    """(accession or None, note).  `hint` may already be an accession."""
    acc = accession_from(hint) or accession_from(gene_id)
    if acc:
        return acc, "given"
    cache = _load_cache()
    if gene_id in cache:
        hit = cache[gene_id]
        return (hit or None), ("cached" if hit else "not in UniProt (cached)")
    try:
        import httpx
    except ImportError:
        return None, "httpx missing"
    for q in _queries(gene_id):
        try:
            r = httpx.get("https://rest.uniprot.org/uniprotkb/search",
                          params={"query": q, "fields": "accession,reviewed,protein_name,length",
                                  "format": "json", "size": 5},
                          timeout=15.0, headers={"User-Agent": "PhiEx/0.1"})
            if r.status_code != 200:
                continue
            results = r.json().get("results") or []
            if not results:
                continue
            # prefer reviewed (Swiss-Prot), then the longest entry (full-length model)
            results.sort(key=lambda e: (e.get("entryType", "").startswith("UniProtKB unreviewed"),
                                        -((e.get("sequence") or {}).get("length") or 0)))
            acc = results[0].get("primaryAccession")
            if acc:
                cache[gene_id] = acc
                _save_cache(cache)
                return acc, f"UniProt search: {q}"
        except Exception as e:
            log("sys", f"UniProt lookup for {gene_id} failed ({e})")
            return None, f"UniProt unreachable ({e})"
    cache[gene_id] = ""
    _save_cache(cache)
    return None, "not found in UniProt (model may be newer than the UniProt import)"
