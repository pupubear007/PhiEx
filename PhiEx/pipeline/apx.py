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
from ..adapters.esm2 import ESM2Adapter, attention_residue_importance
from ..adapters.foldseek import FoldseekAdapter, aggregate_go_terms
from ..adapters.p2rank import P2RankAdapter
from ..adapters.vina import VinaAdapter, ASCORBATE
from ..adapters.diffdock import DiffDockAdapter
from ..adapters.openmm_calc import OpenMMSimulator
from ..adapters.mace import MACEAdapter, select_active_site_region
from ..adapters.surrogate import GBRSurrogate, encode_perturbation
from ..al import ActiveLearningLoop, UCBAcquisition, generate_mutation_panel
from ..analysis import (ligand_residue_contacts, contact_frequency,
                         rmsf_per_residue, residence_time)


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

    # stage 4
    poses: list = field(default_factory=list)

    # stage 5
    n_md_frames: int = 0
    md_total_ps: float = 0.0
    energies_ev: list = field(default_factory=list)
    contacts: dict = field(default_factory=dict)
    top_contacts: list = field(default_factory=list)
    rmsf: dict = field(default_factory=dict)
    residence_ps_near_heme_fe: float = 0.0
    mace_region_size: int = 0
    mace_energy_mean: Optional[float] = None
    mace_energy_sd: Optional[float] = None

    # stage 6
    al_history: list = field(default_factory=list)
    al_converged_residues: list = field(default_factory=list)

    # stage 7
    esm2_per_residue_importance: list = field(default_factory=list)
    agreement_residues: list = field(default_factory=list)
    disagreement_residues: list = field(default_factory=list)

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
            pred = esmfold.predict(protein.sequence, name="APX-pred")
            predicted = pred.value
            result.predicted_pdb_text = predicted.pdb_text
            result.mean_plddt = pred.meta.get("mean_plddt")
            result.plddt_sd = pred.uncertainty
            if protein.residues and predicted.residues:
                rmsd = ca_rmsd(predicted, protein)
                result.rmsd_pred_vs_exp = rmsd
                log("phi", f"compared predicted vs experimental: Cα RMSD = {rmsd:.2f} Å"
                            if rmsd is not None else "RMSD: not computable")

        # Architectural choice: if the experimental ϕ path returned a
        # sequence-only stub (no residues), adopt the predicted ϕ as the
        # working structure for stages 3–7.  README documents this.
        if predicted is not None and not protein.residues and predicted.residues:
            log("sys", "experimental structure absent — using ESMFold prediction as working ϕ")
            protein = predicted
        # If still no cofactor (offline first run, no PDB), synthesise a
        # stub heme placed near the geometric centre of the structure so
        # the cofactor-aware heuristics downstream have something to grip.
        if protein.residues and not cofactors:
            cofactors = [_synthesise_stub_heme(protein)]
            log("sys", f"synthesised STUB heme cofactor at "
                       f"({cofactors[0].metal_atom.xyz[0]:.1f}, "
                       f"{cofactors[0].metal_atom.xyz[1]:.1f}, "
                       f"{cofactors[0].metal_atom.xyz[2]:.1f})")

        # ── stage 2 ────────────────────────────────────────────────
        log("i", "── stage 2  s  function annotation ──")
        esm2 = ESM2Adapter()
        THEORIES.register(esm2)
        emb_pred = esm2.predict(protein.sequence) if protein.sequence else None
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
        state = ComplexState(protein=protein, cofactors=tuple(cofactors))
        p2 = P2RankAdapter()
        THEORIES.register(p2)
        pocket_pred = p2.predict(state)
        pockets: list[Pocket] = pocket_pred.value
        result.pockets = [_pocket_to_json(p) for p in pockets]
        active = next((p for p in pockets if p.is_active_site), pockets[0] if pockets else None)
        result.active_site_id = active.id if active else None

        # ── stage 4 ────────────────────────────────────────────────
        log("i", "── stage 4  s∃  ligand placement ──")
        if active is None:
            log("sys", "no pocket — skipping docking")
            poses = []
        else:
            ligand = ASCORBATE
            if self.use_diffdock:
                dd = DiffDockAdapter()
                THEORIES.register(dd)
                poses = dd.predict(state, ligand, active).value
            else:
                vina = VinaAdapter()
                poses = vina.dock(state, ligand, active)
        result.poses = [_pose_to_json(p) for p in poses]

        # form the docked complex from the best pose
        docked = state
        if poses:
            best = poses[0]
            docked = ComplexState(protein=protein, cofactors=tuple(cofactors),
                                  ligands=(best.ligand,))

        # ── stage 5 ────────────────────────────────────────────────
        log("i", "── stage 5  ϕ → ∃  short ML/MM dynamics ──")
        sim = OpenMMSimulator(total_ps=self.md_total_ps)
        traj = sim.run(docked, label="APX-ascorbate")
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
        metal = next((c.metal_atom for c in cofactors if c.metal_atom), None)
        if metal is not None:
            result.residence_ps_near_heme_fe = residence_time(traj, metal.xyz, cutoff_a=6.0)
            log("s", f"residence ascorbate near heme Fe: "
                      f"{result.residence_ps_near_heme_fe:.2f} ps "
                      f"({100*result.residence_ps_near_heme_fe/max(0.01, result.md_total_ps):.0f}% of trajectory)")

        # ── stage 6 ────────────────────────────────────────────────
        log("i", "── stage 6  s∃ ⟳ tϕ  active-learning loop ──")
        active_residues = list(active.nearby_residues) if active else []
        panel = generate_mutation_panel(active_residues[:8], protein.sequence,
                                        substitutions=["A", "K"])
        log("t", f"AL panel size: {len(panel)} candidates over {len(active_residues)} active-site residues")
        if panel:
            surrogate = GBRSurrogate(sequence_length=max(1, len(protein.sequence)))
            THEORIES.register(surrogate)

            # The v0 evaluator: a cheap proxy that uses MACE on the active-site
            # region energy *delta* between perturbed and unperturbed docked
            # complex.  We simulate this by perturbing a residue's local
            # geometry and re-evaluating MACE energy.  For headless tests
            # without MACE this still works because the MACE stub responds
            # deterministically.
            def evaluator(p: dict) -> tuple[float, float]:
                resid = p.get("residue", 0)
                if resid <= 0 or resid > len(protein.sequence):
                    return 0.0, 1.0
                # construct a perturbed state (BLOSUM-driven scalar; we don't
                # actually mutate the structure in v0 — heuristic, called
                # out in README)
                pred = mace.predict(docked)
                base_e = float(pred.value)
                # synthetic mutation effect: penalise mutations of conserved
                # residues by their BLOSUM score
                from ..adapters.surrogate import _BLOSUM62
                wt = p.get("from"); to = p.get("to")
                blosum = _BLOSUM62.get((wt, to), 0)
                delta = -0.3 * (5 - blosum) + 0.1 * math.sin(0.3 * resid)
                e = base_e + delta
                sd = float(pred.uncertainty or 0.05)
                return e - base_e, sd

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
                loop.candidates = [c for c in loop.candidates if c != r.chosen]
                result.al_history.append({
                    "iteration": r.iteration, "chosen": r.chosen,
                    "predicted": r.predicted, "predicted_sd": r.predicted_sd,
                    "observed": r.observed, "observed_sd": r.observed_sd,
                    "rmse": r.surrogate_rmse, "r2": r.surrogate_r2,
                })
            # converged residues = those whose mutations had observed magnitude
            # in the top quartile across the AL history
            magnitudes = sorted(
                ((abs(h["observed"]), h["chosen"].get("residue"))
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
        if emb_pred is not None and emb_pred.value:
            attention = emb_pred.value.get("attention")
            importance = attention_residue_importance(attention)
            result.esm2_per_residue_importance = importance
            top_attn = sorted(range(len(importance)), key=lambda i: -importance[i])[:15]
            top_attn_residues = sorted({i+1 for i in top_attn})  # 1-based
            traj_residues = set(result.al_converged_residues) | \
                            {r for r, _ in result.top_contacts[:10]}
            agree = sorted(traj_residues & set(top_attn_residues))
            disagree = sorted(traj_residues ^ set(top_attn_residues))
            result.agreement_residues = agree
            result.disagreement_residues = disagree
            log("t", f"agreement (trajectory ∩ ESM-2 attention): {agree}")
            if disagree:
                log("s", f"discovery flag — disagreement at: {disagree}")

        # bookkeeping
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
        "score_uncertainty": p.score_uncertainty,
        "rmsd_to_best": p.rmsd_to_best,
        "ligand": {"name": p.ligand.name, "smiles": p.ligand.smiles,
                   "atoms": [{"name": a.name, "element": a.element,
                              "xyz": list(a.xyz)} for a in p.ligand.atoms]},
    }


def run_apx_pipeline(**kwargs) -> APXPipelineResult:
    return APXPipeline(**kwargs).run()


# ────────────────────────────────────────────────────────────────────────
# stub heme synthesis (offline first-run fallback)
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
