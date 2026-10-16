"""
PhiEx.pipeline.apx — the APX end-to-end example.

Stages, in framework vocabulary:

    1.  ϕ → ∃    structure acquisition
                 (a) PDB fetch                   (classical ϕ)
                 (b) ESMFold inference           (ML ϕ)
                 → compare  (RMSD + pLDDT)
    2.  s        function annotation
                 (a) UniProt GO terms            (classical)
                 (b) ESM-2 embeddings + Foldseek (ML)
                 → ϕ0 functional hypothesis
    3.  ∃        pocket detection
                 P2Rank with cofactor-aware active-site flag
    4.  s∃       ligand placement
                 Vina (or DiffDock stub)
    5.  φ → ∃    short ML/MM dynamics
                 OpenMM bulk + MACE active-site core
    6.  s∃ ⟳ tϕ  active-learning loop over a mutation panel
                 surrogate fits, UCB picks next, observed feeds back
    7.  s∃ ⟳ tϕ  cross-check trajectory residues vs. ESM-2 attention

Returns an APXPipelineResult that the FastAPI layer serialises to the UI.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from ..ticker import log
from ..core.state import ComplexState, Ligand, Pocket
from ..core.theory import THEORIES, FittedTheory
from ..adapters.pdb import (fetch_protein, APX_DEFAULT_PDB,
                             APX_DEFAULT_UNIPROT)
from ..adapters.esmfold import ESMFoldAdapter, ca_rmsd
from ..adapters.esm2 import ESM2Adapter, attention_residue_importance, mutation_llr
from ..adapters.foldseek import FoldseekAdapter, aggregate_go_terms
from ..adapters.p2rank import P2RankAdapter
from ..adapters.vina import (VinaAdapter, ASCORBATE, default_ligand_for_site,
                            resolve_ligand_input)
from ..adapters.diffdock import DiffDockAdapter
from ..adapters.openmm_calc import OpenMMSimulator
from ..adapters.mace import MACEAdapter, select_active_site_region
from ..adapters.surrogate import GBRSurrogate, encode_perturbation
from ..al import ActiveLearningLoop, UCBAcquisition, generate_mutation_panel
from ..analysis import (ligand_residue_contacts, contact_frequency,
                         rmsf_per_residue, residence_time,
                         frames_match_topology)
from ..analysis.site import (ActiveSite, resolve_active_site, choose_active_pocket,
                             chain_residues, functional_point)


# ────────────────────────────────────────────────────────────────────────
# result
# ────────────────────────────────────────────────────────────────────────

@dataclass
class APXPipelineResult:
    # stage 1
    experimental_pdb_id: str = APX_DEFAULT_PDB
    experimental_pdb_text: Optional[str] = None
    predicted_pdb_text: Optional[str] = None
    mean_plddt: Optional[float] = None
    plddt_sd: Optional[float] = None
    rmsd_pred_vs_exp: Optional[float] = None

    # stage 2
    annotations: dict = field(default_factory=dict)
    foldseek_hits: list = field(default_factory=list)
    go_aggregate: list = field(default_factory=list)
    esm2_embedding_sd: Optional[float] = None

    # stage 3
    pockets: list = field(default_factory=list)        # serialisable dicts
    active_site_id: Optional[str] = None
    site: dict = field(default_factory=dict)            # ActiveSite.to_json()

    # stage 4
    ligand: dict = field(default_factory=dict)          # {"name", "smiles", "default"}
    poses: list = field(default_factory=list)
    pose_contacts: list = field(default_factory=list)   # residues ≤ 4 Å of best pose
    docking_error: Optional[str] = None

    # stage 5
    n_md_frames: int = 0
    md_total_ps: float = 0.0
    energies_ev: list = field(default_factory=list)
    contacts: dict = field(default_factory=dict)
    top_contacts: list = field(default_factory=list)
    rmsf: dict = field(default_factory=dict)
    residence_ps_near_site: Optional[float] = None      # None = not computable
    residence_ps_near_heme_fe: Optional[float] = None   # legacy alias of the above
    mace_region_size: int = 0
    mace_energy_mean: Optional[float] = None
    mace_energy_sd: Optional[float] = None

    # stage 6
    al_history: list = field(default_factory=list)
    al_converged_residues: list = field(default_factory=list)
    al_scoring: str = ""                                # what "observed" means

    # stage 7
    esm2_per_residue_importance: list = field(default_factory=list)
    agreement_residues: list = field(default_factory=list)
    disagreement_residues: list = field(default_factory=list)   # trajectory-only
    attention_only_residues: list = field(default_factory=list)  # reported, not flagged
    cross_check_skipped: Optional[str] = None
    residue_labels: dict = field(default_factory=dict)  # resseq → "Ser197"

    # bookkeeping
    theories: list = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────
# main pipeline
# ────────────────────────────────────────────────────────────────────────

@dataclass
class APXPipeline:
    pdb_id: str = APX_DEFAULT_PDB
    uniprot: str = APX_DEFAULT_UNIPROT
    md_total_ps: float = 10.0
    al_iterations: int = 3
    use_diffdock: bool = False
    anchor: Optional[str] = None          # e.g. "S197,H351,E314"; None = auto
    ligand_smiles: Optional[str] = None   # None = default for the site type
    ligand_name: str = ""

    def run(self) -> APXPipelineResult:
        result = APXPipelineResult(experimental_pdb_id=self.pdb_id)

        # ── stage 1 ────────────────────────────────────────────────
        log("i", "── stage 1  ϕ → ∃  structure acquisition ──")
        protein, cofactors, annotations = fetch_protein(self.pdb_id, self.uniprot)
        result.experimental_pdb_text = protein.pdb_text
        result.annotations = annotations

        esmfold = ESMFoldAdapter()
        if protein.residues:
            esmfold.set_reference(protein)
        THEORIES.register(esmfold)

        if not protein.sequence:
            log("sys", "no sequence — skipping ESMFold step")
            predicted = None
        else:
            pred = esmfold.predict_chain(protein, name="APX-pred")
            predicted = pred.value
            result.predicted_pdb_text = predicted.pdb_text
            result.mean_plddt = pred.meta.get("mean_plddt")
            result.plddt_sd = pred.uncertainty
            if protein.residues and predicted.residues:
                rmsd = ca_rmsd(predicted, protein)
                result.rmsd_pred_vs_exp = rmsd
                log("phi", f"compared predicted vs experimental: Cα RMSD = {rmsd:.2f} Å"
                            if rmsd is not None else "RMSD: not computable")

        protein, cofactors = resolve_working_structure(protein, cofactors, predicted)

        # ── stage 2 ────────────────────────────────────────────────
        log("i", "── stage 2  s  function annotation ──")
        esm2 = ESM2Adapter()
        THEORIES.register(esm2)
        chain_res, chain_seq = working_chain(protein)
        emb_pred = esm2.predict(chain_seq) if chain_seq else None
        if emb_pred is not None:
            result.esm2_embedding_sd = float(emb_pred.uncertainty or 0.0)

        foldseek = FoldseekAdapter()
        THEORIES.register(foldseek)
        fs_pred = foldseek.predict(protein, k=5)
        result.foldseek_hits = fs_pred.value.get("hits", [])
        result.go_aggregate = aggregate_go_terms(fs_pred)
        log("s", f"ϕ0 hypothesis: top GO terms = "
                  f"{', '.join(g for g, _ in result.go_aggregate[:5])}")

        # ── stage 3 ────────────────────────────────────────────────
        log("i", "── stage 3  ∃  pocket detection ──")
        # form a ComplexState now that we have cofactors
        site = resolve_active_site(protein, cofactors, self.anchor)
        state = ComplexState(protein=protein, cofactors=tuple(cofactors))
        p2 = P2RankAdapter()
        THEORIES.register(p2)
        pocket_pred = p2.predict(state)
        active, site, pockets = choose_active_pocket(
            protein, pocket_pred.value, site, real_detector=not p2.is_stub)
        log_site(site, active)
        state = ComplexState(protein=protein, cofactors=tuple(cofactors),
                             metadata={"site_center": site.center})
        result.site = site.to_json()
        result.pockets = [_pocket_to_json(p) for p in pockets]
        result.active_site_id = active.id if active else None

        # ── stage 4 ────────────────────────────────────────────────
        log("i", "── stage 4  s∃  ligand placement ──")
        if active is None:
            log("sys", "no pocket — skipping docking")
            poses = []
        else:
            try:
                ligand = pick_ligand(site, self.ligand_smiles, self.ligand_name)
            except ValueError as e:
                log("sys", f"ligand: {e} — docking skipped")
                result.docking_error = str(e)
                ligand = None
            result.ligand = ({"name": ligand.name, "smiles": ligand.smiles,
                              "default": not self.ligand_smiles} if ligand else {})
            poses = []
            if ligand is not None:
                try:
                    if self.use_diffdock:
                        dd = DiffDockAdapter()
                        THEORIES.register(dd)
                        poses = dd.predict(state, ligand, active).value
                    else:
                        poses = VinaAdapter().dock(state, ligand, active)
                except RuntimeError as e:
                    result.docking_error = str(e)
                    log("sys", f"{e} — continuing without a docked ligand")
        result.poses = [_pose_to_json(p) for p in poses]
        if poses and not poses[0].ligand.metadata.get("stub", False):
            result.pose_contacts = pose_contact_residues(protein, poses[0], site.chain)
            if result.pose_contacts:
                log("s", f"best pose contacts (≤ 4 Å): {result.pose_contacts}")

        # form the docked complex from the best pose
        docked = state
        if poses:
            best = poses[0]
            docked = ComplexState(protein=protein, cofactors=tuple(cofactors),
                                  ligands=(best.ligand,),
                                  metadata={"site_center": site.center})

        # ── stage 5 ────────────────────────────────────────────────
        log("i", "── stage 5  ϕ → ∃  short ML/MM dynamics ──")
        sim = OpenMMSimulator(total_ps=self.md_total_ps)
        traj = sim.run(docked, label=f"{protein.name}-{result.ligand.get('name', 'apo')}")
        result.n_md_frames = len(traj.frames)
        result.md_total_ps = traj.frames[-1].time_ps if traj.frames else 0.0
        result.energies_ev = [f.energy_ev for f in traj.frames]

        # MACE on the active-site region
        mace = MACEAdapter()
        THEORIES.register(mace)
        region = select_active_site_region(docked, max_atoms=50)
        result.mace_region_size = len(region)
        if region:
            mace_pred = mace.predict(docked)
            result.mace_energy_mean = float(mace_pred.value)
            result.mace_energy_sd = float(mace_pred.uncertainty or 0.0)

        # contacts / RMSF / residence
        result.contacts = ligand_residue_contacts(traj)
        result.top_contacts = contact_frequency(result.contacts, top_k=10)
        result.rmsf = rmsf_per_residue(traj)
        ligand_simulated = traj.metadata.get("ligand_simulated", True)
        if not frames_match_topology(traj):
            log("sys", "trajectory frames do not match the complex's atom layout "
                       "(e.g. ligand absent from OpenMM system) — contacts, RMSF "
                       "and residence unavailable")
        elif not ligand_simulated:
            log("sys", "residence time not computed: the ligand has no MD parameters "
                       "in v0 and was held at its docked pose (protein-only MD)")
        else:
            res_ps = residence_time(traj, site.center, cutoff_a=6.0)
            result.residence_ps_near_site = res_ps
            result.residence_ps_near_heme_fe = res_ps
            if res_ps is not None:
                log("s", f"residence of {result.ligand.get('name', 'ligand')} within 6 Å of "
                         f"{site.label}: {res_ps:.2f} ps "
                         f"({100*res_ps/max(0.01, result.md_total_ps):.0f}% of trajectory)")

        # ── stage 6 ────────────────────────────────────────────────
        log("i", "── stage 6  s∃ ⟳ tϕ  active-learning loop ──")
        active_residues = rank_site_residues(protein, active, site, limit=10)
        residue_aa = {r.index: _AA1.get(r.name.upper(), "X") for r in chain_res}
        panel = generate_mutation_panel(active_residues, protein.sequence,
                                        substitutions=["A", "K"],
                                        residue_aa=residue_aa)
        log("t", f"AL panel size: {len(panel)} candidates over {len(active_residues)} "
                 f"site residues nearest {site.label}")
        if panel:
            surrogate = GBRSurrogate(sequence_length=max(1, len(chain_seq)))
            THEORIES.register(surrogate)
            evaluator, result.al_scoring = build_mutation_evaluator(
                emb_pred, chain_res, mace, docked)
            log("t", f"AL objective: {result.al_scoring}")

            loop = ActiveLearningLoop(
                surrogate=surrogate, evaluator=evaluator,
                acquisition=UCBAcquisition(kappa=1.5),
            )
            seed_n = min(5, max(2, len(panel)//4))
            loop.seed(panel[:seed_n])
            loop.set_candidates(panel[seed_n:])

            for _ in range(self.al_iterations):
                if not loop.candidates: break
                r = loop.iterate()
                result.al_history.append(r.to_json())
            # converged residues = those whose mutations had observed |Δ|
            # in the top quartile across the AL history
            magnitudes = sorted(
                ((h["observed"], h["chosen"].get("residue"))
                 for h in result.al_history if h["chosen"].get("residue")),
                reverse=True)
            cutoff = magnitudes[len(magnitudes)//4][0] if magnitudes else 0.0
            result.al_converged_residues = sorted({
                resid for mag, resid in magnitudes if mag >= cutoff
            })
            log("t", f"tϕ converged: residues flagged = "
                      f"{result.al_converged_residues}")

        # ── stage 7 ────────────────────────────────────────────────
        log("i", "── stage 7  s∃ ⟳ tϕ  cross-check ESM-2 attention ──")
        if emb_pred is not None and emb_pred.meta.get("stub"):
            # stub attention is synthetic; comparing against it would only
            # manufacture flags
            result.cross_check_skipped = "ESM-2 running as stub"
            log("sys", "cross-check skipped: ESM-2 is a stub, its attention is synthetic")
        elif emb_pred is not None and emb_pred.value:
            attention = emb_pred.value.get("attention")
            importance = attention_residue_importance(attention)
            result.esm2_per_residue_importance = importance
            top_attn = sorted(range(len(importance)), key=lambda i: -importance[i])[:15]
            attn_residues = {chain_res[i].index for i in top_attn if i < len(chain_res)}
            traj_residues = set(result.al_converged_residues) | \
                            {r for r, f in result.top_contacts[:10] if f > 0} | \
                            set(result.pose_contacts)
            # agreement: both lines of evidence point at the residue.
            # discovery flag: the simulation says it matters but sequence-level
            # attention does not — the interesting, testable disagreement.
            # Attention-only residues are reported but not flagged: top-15
            # attention always yields 15 residues, most unrelated to the site.
            result.agreement_residues = sorted(traj_residues & attn_residues)
            result.disagreement_residues = sorted(traj_residues - attn_residues)
            result.attention_only_residues = sorted(attn_residues - traj_residues)
            log("t", f"agreement (trajectory ∩ ESM-2 attention): {result.agreement_residues}")
            if result.disagreement_residues:
                log("s", f"discovery flag — trajectory-only: {result.disagreement_residues}")

        # bookkeeping
        result.residue_labels = {r.index: f"{r.name.title()}{r.index}" for r in chain_res}
        result.theories = [m.info() for m in THEORIES.all()]
        log("i", "── pipeline complete ──")
        return result


# ────────────────────────────────────────────────────────────────────────
# JSON helpers
# ────────────────────────────────────────────────────────────────────────

def _pocket_to_json(p) -> dict:
    return {
        "id": p.id, "center": list(p.center),
        "score": p.score, "is_active_site": p.is_active_site,
        "nearby_residues": list(p.nearby_residues),
    }


def _pose_to_json(p) -> dict:
    return {
        "pose_id": p.pose_id, "score": p.score,
        "stub": bool(p.ligand.metadata.get("stub")),
        "score_uncertainty": p.score_uncertainty,
        "rmsd_to_best": p.rmsd_to_best,
        "ligand": {"name": p.ligand.name, "smiles": p.ligand.smiles,
                   "atoms": [{"name": a.name, "element": a.element,
                              "xyz": list(a.xyz)} for a in p.ligand.atoms]},
    }


def run_apx_pipeline(**kwargs) -> APXPipelineResult:
    return APXPipeline(**kwargs).run()


# ────────────────────────────────────────────────────────────────────────
# working-structure resolution (shared by the pipeline and the per-stage API)
# ────────────────────────────────────────────────────────────────────────

def resolve_working_structure(protein, cofactors, predicted=None):
    """Pick the structure stages 3–7 operate on.

    If the experimental ϕ path returned a sequence-only stub (no residues),
    adopt the predicted ϕ as the working structure.  Cofactors are passed
    through untouched: a protein without a cofactor gets none.  (v0 used to
    synthesise a heme here, anchored at APX's His42, which put every
    downstream stage in the wrong place for any non-APX protein.  The
    active site is now resolved explicitly — see analysis/site.py.)

    Returns (protein, cofactors).  Both `APXPipeline.run` and the FastAPI
    per-stage routes call this so the two paths agree.
    """
    cofactors = [c for c in cofactors if not c.metadata.get("stub")]
    if predicted is not None and not protein.residues and predicted.residues:
        log("sys", "experimental structure absent — using ESMFold prediction as working ϕ")
        protein = predicted
    return protein, cofactors


# ────────────────────────────────────────────────────────────────────────
# shared stage helpers (pipeline + FastAPI routes)
# ────────────────────────────────────────────────────────────────────────

_AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}


def working_chain(protein) -> "tuple[list, str]":
    """Residues and one-letter sequence of the chain the analysis follows
    (the first chain).  Sequence-only proteins return ([], sequence)."""
    res = chain_residues(protein)
    if not res:
        return [], protein.sequence or ""
    return res, "".join(_AA1.get(r.name.upper(), "X") for r in res)


def log_site(site: ActiveSite, pocket) -> None:
    log("phi", f"active site: {site.label}  [source={site.source}, "
               f"confidence={site.confidence}] → pocket {pocket.id if pocket else '—'}")
    for n in site.notes:
        log("sys", f"site note: {n}")


def pick_ligand(site: Optional[ActiveSite], smiles: Optional[str],
                name: str = "") -> Optional[Ligand]:
    """The user's SMILES if given; else a default only when the site type
    implies one; else None (caller skips docking and says why)."""
    if smiles:
        return resolve_ligand_input(smiles, name)      # ValueError if unusable
    lig = default_ligand_for_site(site)
    kind = getattr(site, "kind", "unknown")
    if lig is None:
        log("sys", f"no ligand SMILES given and no default substrate for a {kind} "
                   f"site — docking skipped. Enter a SMILES in 'Ligand SMILES' to dock.")
    else:
        log("sys", f"no ligand SMILES given — default for a {kind} site: {lig.name} "
                   f"(enter a SMILES to dock your own ligand)")
    return lig


def pose_contact_residues(protein, pose, chain=None, cutoff: float = 4.0) -> list:
    atoms = [a.xyz for a in pose.ligand.atoms]
    out = set()
    for r in protein.residues:
        if chain is not None and r.chain != chain:
            continue
        if any(math.dist(a.xyz, l) <= cutoff for a in r.atoms for l in atoms):
            out.add(r.index)
    return sorted(out)


def rank_site_residues(protein, pocket, site: ActiveSite, limit: int = 10) -> list:
    """Pocket residues ordered by distance of their functional atoms to the
    site centre — the old path took the first 8 by residue *number*."""
    if pocket is None:
        return []
    by_idx = {r.index: r for r in chain_residues(protein, site.chain)}
    scored = []
    for i in pocket.nearby_residues:
        r = by_idx.get(i)
        if r is not None:
            scored.append((math.dist(functional_point(r), site.center), i))
    return [i for _, i in sorted(scored)[:limit]]


def build_mutation_evaluator(emb_pred, chain_res, mace, docked):
    """Return (evaluator, description).

    Real path: ESM-2 zero-shot mutation effect — observed = −LLR
    (log p(wt) − log p(mut)); larger = the language model finds the
    mutation more disruptive.  This is a published, cheap variant-effect
    score; it does NOT model ligand binding.
    Fallback (ESM-2 stub): the v0 BLOSUM heuristic, labelled synthetic.
    """
    from ..adapters.surrogate import _BLOSUM62
    pos_of = {r.index: i + 1 for i, r in enumerate(chain_res)}
    logprobs = None
    if emb_pred is not None and not emb_pred.meta.get("stub"):
        logprobs = (emb_pred.value or {}).get("aa_logprobs")

    if logprobs is not None:
        def evaluator(p: dict) -> "tuple[float, float]":
            pos = pos_of.get(p.get("residue", 0))
            llr = mutation_llr(logprobs, pos, p.get("from", "X"), p.get("to", "X")) if pos else None
            if llr is None:
                return 0.0, 1.0
            return -llr, 0.25
        return evaluator, "ESM-2 zero-shot mutation effect (−LLR, nats; larger = more disruptive)"

    base_pred = mace.predict(docked)
    base_sd = float(base_pred.uncertainty or 0.05)

    def evaluator(p: dict) -> "tuple[float, float]":
        resid = p.get("residue", 0)
        wt = p.get("from"); to = p.get("to")
        blosum = _BLOSUM62.get((wt, to), 0)
        delta = -0.3 * (5 - blosum) + 0.1 * math.sin(0.3 * resid)
        return abs(delta), base_sd
    return evaluator, "SYNTHETIC BLOSUM heuristic (ESM-2 is a stub) — not a real measurement"


# ────────────────────────────────────────────────────────────────────────
# legacy stub heme (kept for the offline tests; no longer used by the pipeline)
# ────────────────────────────────────────────────────────────────────────

def _synthesise_stub_heme(protein):
    """Place a stub heme Fe near the canonical APX catalytic histidine
    (His42, 1-indexed) when no experimental cofactor is available.  This
    is exactly the kind of heuristic the framework calls out as a stand-in
    for the real prosthetic-group placement; documented in README."""
    from ..core.state import Atom, Cofactor

    # Anchor the Fe near residue 42 (catalytic His in plant APX) if present,
    # otherwise the geometric centre of the structure.
    target_idx = 42
    anchor = None
    for r in protein.residues:
        if r.index == target_idx and r.ca:
            anchor = r.ca; break
    if anchor is None:
        cas = [r.ca for r in protein.residues if r.ca]
        if cas:
            cx = sum(a.xyz[0] for a in cas) / len(cas)
            cy = sum(a.xyz[1] for a in cas) / len(cas)
            cz = sum(a.xyz[2] for a in cas) / len(cas)
        else:
            cx = cy = cz = 0.0
    else:
        cx, cy, cz = anchor.xyz
        # nudge ~3 Å toward the centroid so Fe sits in a plausible pocket
        cas = [r.ca for r in protein.residues if r.ca]
        if cas:
            mx = sum(a.xyz[0] for a in cas) / len(cas)
            my = sum(a.xyz[1] for a in cas) / len(cas)
            mz = sum(a.xyz[2] for a in cas) / len(cas)
            dx, dy, dz = mx - cx, my - cy, mz - cz
            d = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
            cx += 3.0 * dx / d
            cy += 3.0 * dy / d
            cz += 3.0 * dz / d

    fe = Atom(index=0, name="FE", element="FE", xyz=(cx, cy, cz), bfactor=50.0)
    return Cofactor(name="HEME", resname="HEM", atoms=(fe,), metal_atom=fe,
                    smiles="", metadata={"stub": True})
