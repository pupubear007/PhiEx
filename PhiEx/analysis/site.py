"""
PhiEx.analysis.site — where is the active site?

Every downstream stage (pocket choice, docking box, MACE region, residence
time, AL mutation panel, discovery flags) is organised around ONE point in
space: the active-site centre.  Getting it wrong makes everything after it
meaningless, so it is resolved explicitly, in priority order, and the
source + confidence travel with it into the UI:

    1. user anchor   — residues the user names, e.g. "S197,H351,E314"
    2. cofactor      — a real cofactor parsed from the structure (heme Fe,
                       FAD, NAD …); metal atom if present, else centroid
    3. catalytic     — geometric detection of a Ser/Cys–His–Asp/Glu triad
       triad           (or Cys–His dyad) hydrogen-bond network, boosted when
                       the Ser sits in a G-x-S-x-G nucleophile-elbow motif
    4. pocket        — the top-scoring P2Rank pocket (resolved later, in
                       stage 3, because it needs the pocket list)
    5. centroid      — last resort, flagged low confidence

Nothing is invented: unlike the old v0 path there is no synthetic heme.
A protein with no cofactor gets no cofactor.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..core.state import Atom, Cofactor, Pocket, Protein, Residue

_AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}
_AA1_TO_3 = {v: k for k, v in _AA3.items()}

# atom that best represents what a residue *does* in a site
_FUNCTIONAL_ATOMS = {
    "SER": ("OG",), "CYS": ("SG",), "THR": ("OG1",), "TYR": ("OH",),
    "HIS": ("NE2", "ND1"), "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "LYS": ("NZ",), "ARG": ("NH1", "NH2", "NE"), "ASN": ("OD1", "ND2"),
    "GLN": ("OE1", "NE2"), "TRP": ("NE1",), "MET": ("SD",),
}


@dataclass
class ActiveSite:
    center: tuple[float, float, float]
    source: str                      # anchor | cofactor | catalytic-triad | pocket | centroid
    label: str                       # human-readable, e.g. "Ser197–His351–Glu314 triad"
    confidence: str                  # user | high | medium | low
    kind: str = "unknown"            # metal-cofactor | cofactor | serine-hydrolase | cysteine-hydrolase | pocket | unknown
    residues: tuple[int, ...] = ()   # defining residues (resseq)
    chain: Optional[str] = None
    notes: list = field(default_factory=list)
    cofactor: Optional[str] = None   # e.g. "HEME" when the site is a cofactor

    def to_json(self) -> dict:
        return {"center": [round(c, 3) for c in self.center], "source": self.source,
                "label": self.label, "confidence": self.confidence, "kind": self.kind,
                "residues": list(self.residues), "chain": self.chain,
                "notes": list(self.notes), "cofactor": self.cofactor}


# ────────────────────────────────────────────────────────────────────────
# small geometry helpers
# ────────────────────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _mean(points: Iterable[tuple]) -> tuple[float, float, float]:
    pts = list(points)
    n = len(pts) or 1
    return (sum(p[0] for p in pts)/n, sum(p[1] for p in pts)/n, sum(p[2] for p in pts)/n)


def _atom(r: Residue, *names: str) -> Optional[Atom]:
    for n in names:
        for a in r.atoms:
            if a.name == n:
                return a
    return None


def functional_point(r: Residue) -> tuple[float, float, float]:
    names = _FUNCTIONAL_ATOMS.get(r.name.upper())
    if names:
        pts = [a.xyz for a in r.atoms if a.name in names]
        if pts:
            return _mean(pts)
    side = [a.xyz for a in r.atoms if a.name not in ("N", "C", "O", "CA")]
    if side:
        return _mean(side)
    return r.ca.xyz if r.ca else _mean(a.xyz for a in r.atoms)


def first_chain(protein: Protein) -> Optional[str]:
    return protein.residues[0].chain if protein.residues else None


def chain_residues(protein: Protein, chain: Optional[str] = None) -> list[Residue]:
    chain = chain if chain is not None else first_chain(protein)
    return [r for r in protein.residues if r.chain == chain]


def residue_label(r: Residue) -> str:
    return f"{r.name.title()}{r.index}"


def residues_near(protein: Protein, center, cutoff: float = 8.0,
                  chain: Optional[str] = None) -> tuple[int, ...]:
    """Residues (resseq) with any atom within `cutoff` Å of `center`."""
    out = []
    for r in protein.residues:
        if chain is not None and r.chain != chain:
            continue
        if any(_dist(a.xyz, center) <= cutoff for a in r.atoms):
            out.append(r.index)
    return tuple(sorted(set(out)))


# ────────────────────────────────────────────────────────────────────────
# 1. user anchor
# ────────────────────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<aa>[A-Za-z]{1,3})?(?P<num>-?\d+)$")


def parse_anchor(spec: str) -> list[tuple[Optional[str], Optional[str], int]]:
    """'S197, H351, A:E314, 42, Ser197' → [(chain, aa1, resseq), ...]."""
    out = []
    for tok in re.split(r"[,\s;+]+", (spec or "").strip()):
        if not tok:
            continue
        m = _ANCHOR_RE.match(tok)
        if not m:
            raise ValueError(f"cannot parse anchor residue {tok!r} "
                             f"(use e.g. S197, Ser197, A:197)")
        aa = m.group("aa")
        if aa:
            aa = aa.upper()
            aa = _AA3.get(aa, aa) if len(aa) == 3 else aa
            if len(aa) != 1 or aa not in _AA1_TO_3:
                raise ValueError(f"unknown amino acid in anchor {tok!r}")
        out.append((m.group("chain"), aa, int(m.group("num"))))
    if not out:
        raise ValueError("empty anchor")
    return out


def site_from_anchor(protein: Protein, spec: str,
                     cofactors: Optional[list] = None) -> ActiveSite:
    wanted = parse_anchor(spec)
    default_chain = first_chain(protein)
    found: list[Residue] = []
    notes: list[str] = []
    for chain, aa, num in wanted:
        ch = chain or default_chain
        r = next((x for x in protein.residues if x.chain == ch and x.index == num), None)
        if r is None:
            raise ValueError(f"anchor residue {num} not found in chain {ch}")
        actual = _AA3.get(r.name.upper(), "X")
        if aa and aa != actual:
            notes.append(f"anchor said {aa}{num} but the structure has "
                         f"{residue_label(r)} — using the structure")
        if r not in found:
            found.append(r)
    center = _mean(functional_point(r) for r in found)
    label = "anchor " + "–".join(residue_label(r) for r in found)
    kind, cof_name = _classify_anchor(found, center, cofactors or [])
    if kind != "unknown":
        notes.append(f"anchor residues look like a {kind} site"
                     + (f" ({cof_name})" if cof_name else ""))
    return ActiveSite(center=center, source="anchor", label=label,
                      confidence="user", kind=kind,
                      residues=tuple(r.index for r in found),
                      chain=found[0].chain, notes=notes, cofactor=cof_name)


def _classify_anchor(found: list, center, cofactors: list) -> "tuple[str, Optional[str]]":
    """What kind of site did the user point at?  A Ser/Cys hydrogen-bonded
    to a His among the anchor residues → hydrolase; a real cofactor within
    6 Å of the anchor centre → that cofactor's site; otherwise unknown."""
    hiss = [r for r in found if r.name == "HIS"]
    for nuc in (r for r in found if r.name in ("SER", "CYS")):
        nu = _atom(nuc, "OG", "SG")
        if nu is None:
            continue
        cutoff = _HB_MAX + (0.4 if nuc.name == "CYS" else 0.0)
        for his in hiss:
            if any(a.name in ("NE2", "ND1") and _dist(a.xyz, nu.xyz) <= cutoff
                   for a in his.atoms):
                return ("serine-hydrolase" if nuc.name == "SER"
                        else "cysteine-hydrolase"), None
    for c in cofactors:
        if c.metadata.get("stub") or not c.atoms:
            continue
        if min(_dist(a.xyz, center) for a in c.atoms) <= 6.0:
            return ("metal-cofactor" if c.metal_atom else "cofactor"), c.name
    return "unknown", None


# ────────────────────────────────────────────────────────────────────────
# 2. real cofactor
# ────────────────────────────────────────────────────────────────────────

def site_from_cofactors(protein: Protein, cofactors: list[Cofactor]) -> Optional[ActiveSite]:
    real = [c for c in cofactors if not c.metadata.get("stub") and c.atoms]
    if not real:
        return None
    chain = first_chain(protein)

    def _key(c: Cofactor) -> float:
        # prefer the copy bound to the first chain (homo-oligomers repeat it)
        p = c.metal_atom.xyz if c.metal_atom else _mean(a.xyz for a in c.atoms)
        own = chain_residues(protein, chain)
        return min((_dist(p, a.xyz) for r in own for a in r.atoms), default=1e9)

    c = sorted(real, key=_key)[0]
    if c.metal_atom is not None:
        center, kind = c.metal_atom.xyz, "metal-cofactor"
        label = f"{c.name} {c.metal_atom.element.title()}"
    else:
        center, kind = _mean(a.xyz for a in c.atoms), "cofactor"
        label = f"{c.name} centroid"
    return ActiveSite(center=center, source="cofactor", label=label,
                      confidence="high", kind=kind,
                      residues=residues_near(protein, center, 4.5, chain),
                      chain=chain, cofactor=c.name)


# ────────────────────────────────────────────────────────────────────────
# 3. catalytic triad / dyad detection
# ────────────────────────────────────────────────────────────────────────

_HB_MAX = 3.6     # Å, heavy-atom hydrogen-bond cutoff


@dataclass
class TriadHit:
    nucleophile: Residue
    histidine: Residue
    acid: Optional[Residue]
    motif: bool
    score: float

    @property
    def label(self) -> str:
        parts = [self.nucleophile, self.histidine] + ([self.acid] if self.acid else [])
        name = "–".join(residue_label(r) for r in parts)
        return f"{name} {'triad' if self.acid else 'dyad'}"


def detect_catalytic_sites(protein: Protein, chain: Optional[str] = None) -> list[TriadHit]:
    """Find Ser/Cys–His(–Asp/Glu) hydrogen-bond networks in one chain.

    Score (higher = more convincing):
        +3   acid partner present (triad, not dyad)
        +3   Ser in a G-x-S-x-G nucleophile-elbow motif (α/β-hydrolases)
        +1   nucleophile buried (closer to the chain centroid than average)
        +1   mean pLDDT/B-factor of the three residues ≥ 70 when the file
             is a prediction (AlphaFold writes pLDDT there)
    """
    res = chain_residues(protein, chain)
    if not res:
        return []
    seq = "".join(_AA3.get(r.name.upper(), "X") for r in res)
    pos = {id(r): i for i, r in enumerate(res)}
    cas = [r.ca.xyz for r in res if r.ca]
    centroid = _mean(cas) if cas else (0.0, 0.0, 0.0)
    mean_r = (sum(_dist(c, centroid) for c in cas) / len(cas)) if cas else 0.0

    nucs = [r for r in res if r.name in ("SER", "CYS")]
    hiss = [r for r in res if r.name == "HIS"]
    acids = [r for r in res if r.name in ("ASP", "GLU")]
    hits: list[TriadHit] = []
    for nuc in nucs:
        nu = _atom(nuc, "OG", "SG")
        if nu is None:
            continue
        for his in hiss:
            hn = [a for a in his.atoms if a.name in ("NE2", "ND1")]
            near = [a for a in hn if _dist(a.xyz, nu.xyz) <= _HB_MAX + (0.4 if nuc.name == "CYS" else 0)]
            if not near:
                continue
            other = [a for a in hn if a not in near]
            acid_hit = None
            if other:
                best = None
                for ac in acids:
                    for o in ac.atoms:
                        if o.name in ("OD1", "OD2", "OE1", "OE2"):
                            d = _dist(o.xyz, other[0].xyz)
                            if d <= _HB_MAX and (best is None or d < best[0]):
                                best = (d, ac)
                acid_hit = best[1] if best else None
            i = pos[id(nuc)]
            motif = (nuc.name == "SER" and i >= 2 and i + 2 < len(seq)
                     and seq[i-2] == "G" and seq[i+2] == "G")
            score = (3 if acid_hit else 0) + (3 if motif else 0)
            if _dist(nu.xyz, centroid) < mean_r:
                score += 1
            trio = [nuc, his] + ([acid_hit] if acid_hit else [])
            bf = [sum(a.bfactor for a in r.atoms) / max(1, len(r.atoms)) for r in trio]
            if protein.source != "pdb" and bf and sum(bf) / len(bf) >= 70:
                score += 1
            hits.append(TriadHit(nuc, his, acid_hit, motif, score))
    hits.sort(key=lambda h: -h.score)
    return hits


def site_from_catalytic(protein: Protein) -> Optional[ActiveSite]:
    hits = detect_catalytic_sites(protein)
    if not hits:
        return None
    h = hits[0]
    if h.acid is None and not h.motif:
        return None                     # a lone Ser/Cys–His H-bond is too common to trust
    trio = [h.nucleophile, h.histidine] + ([h.acid] if h.acid else [])
    center = _atom(h.nucleophile, "OG", "SG").xyz
    conf = "high" if (h.acid and h.motif) else "medium"
    kind = "serine-hydrolase" if h.nucleophile.name == "SER" else "cysteine-hydrolase"
    notes = []
    if h.motif:
        i = h.nucleophile.index
        notes.append(f"G-x-S-x-G motif around Ser{i} (nucleophile elbow)")
    if len(hits) > 1:
        notes.append("other candidates: " + "; ".join(x.label for x in hits[1:4]))
    return ActiveSite(center=center, source="catalytic-triad", label=h.label,
                      confidence=conf, kind=kind,
                      residues=tuple(r.index for r in trio),
                      chain=h.nucleophile.chain, notes=notes)


# ────────────────────────────────────────────────────────────────────────
# 4/5. pocket and centroid fallbacks
# ────────────────────────────────────────────────────────────────────────

def site_from_pocket(protein: Protein, pocket: Pocket, real_detector: bool) -> ActiveSite:
    return ActiveSite(center=tuple(pocket.center), source="pocket",
                      label=f"top pocket {pocket.id}",
                      confidence="medium" if real_detector else "low",
                      kind="pocket", residues=tuple(pocket.nearby_residues),
                      chain=first_chain(protein),
                      notes=[] if real_detector else
                      ["pocket from the geometric fallback, not P2Rank — install "
                       "P2Rank or give an anchor for a trustworthy site"])


def site_from_centroid(protein: Protein) -> ActiveSite:
    cas = [r.ca.xyz for r in chain_residues(protein) if r.ca]
    c = _mean(cas) if cas else (0.0, 0.0, 0.0)
    return ActiveSite(center=c, source="centroid", label="protein centroid",
                      confidence="low", chain=first_chain(protein),
                      notes=["no anchor, cofactor, catalytic motif or pocket found"])


# ────────────────────────────────────────────────────────────────────────
# entry point used by the pipeline and the API
# ────────────────────────────────────────────────────────────────────────

def resolve_active_site(protein: Protein, cofactors: list[Cofactor],
                        anchor: Optional[str] = None) -> Optional[ActiveSite]:
    """Steps 1–3.  Returns None when only a pocket can decide (step 4)."""
    if anchor:
        return site_from_anchor(protein, anchor, cofactors)
    return site_from_cofactors(protein, cofactors) or site_from_catalytic(protein)


def choose_active_pocket(protein: Protein, pockets: list[Pocket],
                         site: Optional[ActiveSite], real_detector: bool,
                         match_within: float = 8.0
                         ) -> tuple[Optional[Pocket], ActiveSite, list[Pocket]]:
    """Pick the pocket that IS the active site, and finalise the site.

    With a site: the nearest pocket within `match_within` Å, otherwise a
    pocket built around the site itself (a detector missing the catalytic
    cleft should not move the site).  Without a site: the top-scoring
    pocket defines it.  Returns (active_pocket, site, pockets_with_flags).
    """
    pockets = list(pockets)
    if site is None:
        if pockets:
            top = max(pockets, key=lambda p: p.score)
            site = site_from_pocket(protein, top, real_detector)
        else:
            site = site_from_centroid(protein)
    best, best_d = None, float("inf")
    # a geometric-fallback pocket must not move a site we already know
    candidates = pockets if (real_detector or site.source == "pocket") else []
    for p in candidates:
        d = _dist(p.center, site.center)
        if d < best_d:
            best, best_d = p, d
    if best is None or best_d > match_within:
        best = Pocket(id="SITE", center=tuple(site.center), score=0.0,
                      nearby_residues=residues_near(protein, site.center, 8.0, site.chain),
                      metadata={"from_site": site.source})
        pockets.append(best)
        best_d = 0.0
    flagged = [Pocket(id=p.id, center=p.center, score=p.score,
                      nearby_residues=p.nearby_residues,
                      is_active_site=(p is best),
                      metadata={**p.metadata,
                                "site_distance": round(best_d, 2) if p is best else None})
               for p in pockets]
    active = next(p for p in flagged if p.is_active_site)
    return active, site, flagged
