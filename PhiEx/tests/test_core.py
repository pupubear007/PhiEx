"""
Fast unit tests — no ML weights, no network, no OpenMM required.

    make unit          # or:  python -m pytest PhiEx/tests -q
"""

from __future__ import annotations
import asyncio
import os
import subprocess
import sys
import threading

import pytest

from PhiEx.core.state import Atom, Residue, Protein, Cofactor, Ligand, ComplexState, Pocket
from PhiEx.core.learned import Prediction
from PhiEx.ticker import Ticker
from PhiEx.al import (ActiveLearningLoop, ThompsonAcquisition, UCBAcquisition,
                      make_acquisition, generate_mutation_panel)
from PhiEx.adapters.surrogate import GBRSurrogate, _BLOSUM62, encode_perturbation
from PhiEx.adapters.openmm_calc import OpenMMSimulator
from PhiEx.adapters.mace import MACEAdapter
from PhiEx.analysis import ligand_residue_contacts, residence_time
from PhiEx.runners.batch import synthetic_evaluator, run_batch_payload


# ────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────

def _toy_complex() -> ComplexState:
    """Three-residue protein, one-atom 'heme' Fe, two-atom ligand sitting
    right next to residue 2 and the Fe."""
    residues = tuple(
        Residue(index=i, name="ALA", chain="A", atoms=(
            Atom(0, "N", "N", (4.0 * i, 0.0, 0.0)),
            Atom(1, "CA", "C", (4.0 * i + 1.0, 0.0, 0.0)),
        ))
        for i in (1, 2, 3)
    )
    protein = Protein(name="toy", sequence="AHA", residues=residues)
    fe = Atom(0, "FE", "FE", (9.0, 3.0, 0.0))
    heme = Cofactor(name="HEME", resname="HEM", atoms=(fe,), metal_atom=fe)
    lig = Ligand(name="lig", smiles="CO", atoms=(
        Atom(0, "C1", "C", (9.0, 1.5, 0.0)),
        Atom(1, "O1", "O", (9.0, 2.5, 0.0)),
    ))
    return ComplexState(protein=protein, cofactors=(heme,), ligands=(lig,))


class _RecordingSurrogate:
    """Minimal surrogate double that records the feature vectors it sees."""
    def __init__(self, sequence_length: int) -> None:
        self.inner = GBRSurrogate(sequence_length=sequence_length)
        self.fit_X: list = []
        self.predict_X: list = []

    def encode(self, p):
        return self.inner.encode(p)

    def fit(self, X, y):
        self.fit_X = list(X)
        return self.inner.fit(X, y)

    def predict(self, p):
        self.predict_X.append(self.inner.encode(p))
        return self.inner.predict(p)


# ────────────────────────────────────────────────────────────────────────
# ticker
# ────────────────────────────────────────────────────────────────────────

def test_ticker_delivers_events_logged_from_worker_threads():
    ticker = Ticker()

    async def main():
        sub = ticker.subscribe(replay=False)
        first = asyncio.ensure_future(sub.__anext__())
        await asyncio.sleep(0)            # let the subscriber register
        t = threading.Thread(target=lambda: ticker.log("sys", "from thread"))
        t.start(); t.join()
        ev = await asyncio.wait_for(first, timeout=2.0)
        await sub.aclose()
        return ev

    ev = asyncio.run(main())
    assert ev.msg == "from thread"


# ────────────────────────────────────────────────────────────────────────
# acquisition
# ────────────────────────────────────────────────────────────────────────

def test_thompson_draws_independently_per_candidate():
    acq = ThompsonAcquisition(seed=0)
    pred = Prediction(value=0.0, uncertainty=1.0)
    draws = {acq.score(pred) for _ in range(5)}
    assert len(draws) == 5


def test_thompson_is_reproducible_for_a_seed():
    pred = Prediction(value=0.0, uncertainty=1.0)
    a, b = ThompsonAcquisition(seed=7), ThompsonAcquisition(seed=7)
    assert [a.score(pred) for _ in range(3)] == [b.score(pred) for _ in range(3)]


def test_make_acquisition():
    assert isinstance(make_acquisition("UCB"), UCBAcquisition)
    with pytest.raises(ValueError):
        make_acquisition("nope")


# ────────────────────────────────────────────────────────────────────────
# surrogate + AL loop
# ────────────────────────────────────────────────────────────────────────

def test_blosum62_is_full_and_symmetric():
    assert len(_BLOSUM62) == 400
    assert all(_BLOSUM62[a, b] == _BLOSUM62[b, a] for a, b in _BLOSUM62)
    assert _BLOSUM62["H", "A"] == -2 and _BLOSUM62["W", "W"] == 11


def test_loop_trains_on_the_same_features_it_predicts_with():
    seq = "A" * 120                      # deliberately != the encoder default (250)
    surrogate = _RecordingSurrogate(sequence_length=len(seq))
    panel = generate_mutation_panel(list(range(10, 20)), seq, ["K", "D"])
    loop = ActiveLearningLoop(surrogate=surrogate, evaluator=synthetic_evaluator)
    loop.seed(panel[:4])
    loop.set_candidates(panel[4:])
    loop.iterate()
    # every row the surrogate was fit on must be encodable at predict time
    expected = [surrogate.encode(p) for p in panel[:4]]
    assert surrogate.fit_X[:4] == expected
    assert surrogate.fit_X[0] != encode_perturbation(panel[0])   # 250-normalised


def test_loop_removes_evaluated_candidate_and_accumulates_history():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    panel = generate_mutation_panel(list(range(1, 12)), seq, ["A", "K"])
    loop = ActiveLearningLoop(surrogate=GBRSurrogate(sequence_length=len(seq)),
                              evaluator=synthetic_evaluator)
    loop.seed(panel[:4])
    loop.set_candidates(panel[4:])
    n0 = len(loop.candidates)
    chosen = [loop.iterate().chosen for _ in range(3)]
    assert len(loop.candidates) == n0 - 3
    assert all(c not in loop.candidates for c in chosen)
    assert len({tuple(sorted(c.items())) for c in chosen}) == 3   # no repeats


def test_surrogate_prediction_always_carries_uncertainty():
    s = GBRSurrogate(sequence_length=50)
    p = {"type": "mutation", "residue": 3, "from": "A", "to": "K"}
    assert s.predict(p).uncertainty is not None         # untrained
    X = [s.encode({**p, "residue": r}) for r in range(1, 6)]
    s.fit(X, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert s.predict(p).uncertainty is not None         # trained


# ────────────────────────────────────────────────────────────────────────
# runners
# ────────────────────────────────────────────────────────────────────────

def test_synthetic_evaluator_is_deterministic_across_processes():
    p = {"type": "mutation", "residue": 42, "from": "H", "to": "A"}
    code = ("from PhiEx.runners.batch import synthetic_evaluator as f;"
            f"print(repr(f({p!r})))")
    outs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.check_output([sys.executable, "-c", code], env=env,
                                         text=True).strip().splitlines()[-1])
    assert len(outs) == 1


def test_batch_payload_runs_with_each_acquisition():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    panel = generate_mutation_panel(list(range(1, 12)), seq, ["A", "K"])
    for acq in ("ucb", "ei", "thompson"):
        out = run_batch_payload({"candidates": panel, "n_iterations": 2,
                                 "acquisition": acq})
        assert out["n_iterations"] == 2


# ────────────────────────────────────────────────────────────────────────
# dynamics stub + analysis
# ────────────────────────────────────────────────────────────────────────

def test_stub_md_frames_cover_cofactor_and_ligand_atoms():
    state = _toy_complex()
    sim = OpenMMSimulator(total_ps=0.5)
    sim.is_stub = True                     # force the stub path even if OpenMM exists
    traj = sim.run(state)
    assert traj.frames
    assert all(len(f.positions) == state.n_atoms() for f in traj.frames)
    contacts = ligand_residue_contacts(traj, cutoff_a=4.0)
    assert contacts[2] > 0.5               # ligand sits next to residue 2
    fe = state.cofactors[0].metal_atom.xyz
    assert residence_time(traj, fe, cutoff_a=6.0) > 0.0


def test_mace_forces_match_region_size():
    state = _toy_complex()
    mace = MACEAdapter(device="cpu", max_atoms=5)
    forces = mace.forces(state)
    assert len(forces) == 5
    assert mace.predict(state).uncertainty is not None


# ────────────────────────────────────────────────────────────────────────
# surrogate quality is reported out-of-sample
# ────────────────────────────────────────────────────────────────────────

def test_al_history_reports_prospective_and_loo_metrics():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    panel = generate_mutation_panel(list(range(1, 12)), seq, ["A", "K"])
    loop = ActiveLearningLoop(surrogate=GBRSurrogate(sequence_length=len(seq)),
                              evaluator=synthetic_evaluator)
    loop.seed(panel[:4])
    loop.set_candidates(panel[4:])
    results = [loop.iterate() for _ in range(3)]
    last = results[-1].to_json()
    errs = [r.observed - r.predicted for r in results]
    assert last["prospective_rmse"] == pytest.approx(
        (sum(e * e for e in errs) / len(errs)) ** 0.5)
    assert 0.0 <= last["coverage_1sigma"] <= 1.0
    assert last["loo_r2"] is not None
    # the in-sample fit is optimistic; LOO must not be better than it
    assert last["loo_r2"] <= last["train_r2"] + 1e-9


# ────────────────────────────────────────────────────────────────────────
# shared working-structure fallback
# ────────────────────────────────────────────────────────────────────────

def test_resolve_working_structure_adopts_prediction_without_inventing_cofactors():
    from PhiEx.pipeline import resolve_working_structure
    predicted = _toy_complex().protein
    seq_only = Protein(name="exp", sequence=predicted.sequence)
    protein, cofactors = resolve_working_structure(seq_only, [], predicted)
    assert protein is predicted
    assert cofactors == []          # no synthetic heme any more


def test_resolve_working_structure_keeps_experimental():
    state = _toy_complex()
    from PhiEx.pipeline import resolve_working_structure
    protein, cofactors = resolve_working_structure(
        state.protein, list(state.cofactors), predicted=None)
    assert protein is state.protein
    assert cofactors == list(state.cofactors)


# ────────────────────────────────────────────────────────────────────────
# analysis refuses frames whose atom layout doesn't match the state
# ────────────────────────────────────────────────────────────────────────

def test_analysis_unavailable_when_frames_miss_ligand():
    from PhiEx.core.trajectory import Trajectory, Frame
    from PhiEx.analysis import rmsf_per_residue, frames_match_topology
    state = _toy_complex()
    protein_only = tuple(a.xyz for a in state.protein.all_atoms())
    traj = Trajectory(parent_state=state,
                      frames=[Frame(step=i, time_ps=0.1 * i, positions=protein_only)
                              for i in (1, 2)])
    assert not frames_match_topology(traj)
    assert ligand_residue_contacts(traj) == {}
    assert rmsf_per_residue(traj) == {}
    fe = state.cofactors[0].metal_atom.xyz
    assert residence_time(traj, fe) is None


# ────────────────────────────────────────────────────────────────────────
# ESM-2: a failing forward pass on the accelerator falls back, not 500s
# ────────────────────────────────────────────────────────────────────────

def test_esm2_falls_back_to_cpu_then_stub(monkeypatch):
    from PhiEx.adapters.esm2 import ESM2Adapter

    class _FakeModel:
        def to(self, device):
            self.device = device
            return self

    esm2 = ESM2Adapter(device="cpu")
    esm2.is_stub, esm2._model, esm2.device = False, _FakeModel(), "mps"
    calls = []

    def flaky(sequence, name):
        calls.append(esm2.device)
        if esm2.device == "mps":
            raise RuntimeError("MPS backend out of memory")
        return Prediction(value={"embedding": [], "attention": []}, uncertainty=0.1)

    monkeypatch.setattr(esm2, "_predict_real", flaky)
    assert esm2.predict("ACDE").uncertainty == 0.1
    assert calls == ["mps", "cpu"] and esm2.device == "cpu"

    monkeypatch.setattr(esm2, "_predict_real",
                        lambda s, n: (_ for _ in ()).throw(RuntimeError("boom")))
    pred = esm2.predict("ACDE")               # cpu fails too → stub
    assert pred.meta.get("stub") is True


def test_api_errors_are_reported_not_bare_500(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main

    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(main, "ESM2Adapter", _Broken)
    main.SESSION["protein"] = Protein(name="t", sequence="ACDE")
    try:
        with TestClient(main.app, raise_server_exceptions=False) as c:
            r = c.post("/api/annotate", json={})
    finally:
        main.SESSION.pop("protein", None)
    assert r.status_code == 500
    assert "simulated failure" in r.json()["error"]


# ────────────────────────────────────────────────────────────────────────
# ESMFold helpers: single-chain folding and superposed RMSD
# ────────────────────────────────────────────────────────────────────────

def _ca_res(i, chain, xyz, name="ALA"):
    return Residue(index=i, name=name, chain=chain,
                   atoms=(Atom(i, "CA", "C", xyz),))


def test_select_chain_sequence_picks_one_chain():
    from PhiEx.adapters.esmfold import select_chain_sequence
    res = tuple([_ca_res(1, "A", (0, 0, 0), "HIS"), _ca_res(2, "A", (1, 0, 0), "ARG"),
                 _ca_res(1, "B", (0, 5, 0), "TRP")])
    p = Protein(name="x", sequence="HRW", residues=res)
    assert select_chain_sequence(p) == ("A", "HR")
    assert select_chain_sequence(p, "B") == ("B", "W")
    assert select_chain_sequence(Protein(name="s", sequence="ACD")) == (None, "ACD")


def test_ca_rmsd_is_superposition_invariant():
    import math as _m
    from PhiEx.adapters.esmfold import ca_rmsd
    pts = [(0, 0, 0), (3.8, 0, 0), (3.8, 3.8, 0), (0, 3.8, 1.0), (1.0, 1.0, 4.0)]
    exp = Protein(name="e", residues=tuple(_ca_res(10 + i, "A", p) for i, p in enumerate(pts)))
    # rotate 90° about z and translate; renumber as ESMFold would (chain A, 1..N)
    rot = [(-y + 7, x - 3, z + 2) for x, y, z in pts]
    pred = Protein(name="p", residues=tuple(_ca_res(1 + i, "A", p) for i, p in enumerate(rot)),
                   metadata={"query_chain": "A"})
    assert ca_rmsd(pred, exp) < 1e-6


def test_esmfold_off_switch_keeps_stub(monkeypatch):
    from PhiEx.adapters.esmfold import ESMFoldAdapter
    monkeypatch.setenv("PHIEX_ESMFOLD", "off")
    a = ESMFoldAdapter(device="cpu")
    assert a.is_stub and "PHIEX_ESMFOLD=off" in a.backend


def test_foldseek_target_parsing_and_single_chain():
    from PhiEx.adapters.foldseek import _pdb_id_from_target, _single_chain_pdb
    assert _pdb_id_from_target("1oag-assembly1.cif.gz_A") == "1oag"
    assert _pdb_id_from_target("2CCA_B") == "2cca"
    assert _pdb_id_from_target("AF-Q9XXX-F1-model_v4") is None
    pdb = ("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 0.00           C\n"
           "ATOM      2  CA  ALA B   1       1.000   0.000   0.000  1.00 0.00           C\n")
    text, chain = _single_chain_pdb(Protein(name="t", pdb_text=pdb))
    assert chain == "A" and text.count("ATOM") == 1


def test_mace_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("PHIEX_MACE_DEVICE", raising=False)
    assert MACEAdapter().device == "cpu"


# ────────────────────────────────────────────────────────────────────────
# AlphaFold fetch: API first, no version guessing
# ────────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, code, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text
    def json(self):
        return self._p


class _FakeClient:
    routes: dict = {}
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get(self, url, headers=None):
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return _FakeResp(404)
    head = get


def test_alphafold_uses_api_url(monkeypatch, tmp_path):
    import httpx
    from PhiEx.adapters import pdb as pdbmod
    pdb_line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n"
    _FakeClient.routes = {
        "/api/prediction/Q00001": _FakeResp(200, [{
            "entryId": "AF-Q00001-F1", "uniprotAccession": "Q00001", "latestVersion": 9,
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-Q00001-F1-model_v9.pdb"}]),
        "model_v9.pdb": _FakeResp(200, text=pdb_line),
    }
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    text = pdbmod.fetch_alphafold_pdb("Q00001", cache_dir=tmp_path)
    assert text and text.startswith("ATOM")


def test_alphafold_absent_stops_without_guessing(monkeypatch, tmp_path):
    import httpx
    from PhiEx.adapters import pdb as pdbmod
    _FakeClient.routes = {"/api/prediction/": _FakeResp(404)}
    calls = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(pdbmod, "_fetch_text", lambda u, **k: calls.append(u))
    assert pdbmod.fetch_alphafold_pdb("Q00002", cache_dir=tmp_path) is None
    assert calls == []          # no model_vN probing once the API says "absent"


# ────────────────────────────────────────────────────────────────────────
# active-site resolution
# ────────────────────────────────────────────────────────────────────────

def _res(i, name, atoms, chain="A"):
    return Residue(index=i, name=name, chain=chain,
                   atoms=tuple(Atom(k, n, n[0], xyz) for k, (n, xyz) in enumerate(atoms)))


def _triad_protein(with_motif=True, with_acid=True):
    """Gly-x-Ser-x-Gly with Ser OG — His NE2 — Glu OE2 hydrogen bonds,
    plus filler residues far away."""
    res = []
    seq = ["GLY", "ALA", "SER", "ALA", "GLY"] if with_motif else ["ALA", "ALA", "SER", "ALA", "ALA"]
    for i, nm in enumerate(seq, start=10):
        atoms = [("N", (i*3.8, 0, 0)), ("CA", (i*3.8, 1, 0)), ("C", (i*3.8, 2, 0))]
        if nm == "SER":
            atoms.append(("OG", (0.0, 0.0, 0.0)))
        res.append(_res(i, nm, atoms))
    res.append(_res(40, "HIS", [("CA", (0, 5, 0)), ("ND1", (0, 4.6, 0)), ("NE2", (0, 2.8, 0))]))
    if with_acid:
        res.append(_res(60, "GLU", [("CA", (0, 9, 0)), ("OE1", (0, 8.0, 0)), ("OE2", (0, 7.4, 0))]))
    return Protein(name="toy", residues=tuple(res), source="pdb")


def test_triad_detected_with_motif_and_acid():
    from PhiEx.analysis.site import resolve_active_site
    site = resolve_active_site(_triad_protein(), [])
    assert site is not None and site.source == "catalytic-triad"
    assert site.residues == (12, 40, 60) and site.confidence == "high"
    assert site.kind == "serine-hydrolase"


def test_lone_ser_his_pair_is_not_trusted():
    from PhiEx.analysis.site import resolve_active_site
    assert resolve_active_site(_triad_protein(with_motif=False, with_acid=False), []) is None


def test_real_cofactor_beats_triad_and_stub_cofactor_is_ignored():
    from PhiEx.analysis.site import resolve_active_site
    p = _triad_protein()
    fe = Atom(0, "FE", "FE", (30.0, 0, 0))
    heme = Cofactor(name="HEME", resname="HEM", atoms=(fe,), metal_atom=fe)
    assert resolve_active_site(p, [heme]).source == "cofactor"
    stub = Cofactor(name="HEME", resname="HEM", atoms=(fe,), metal_atom=fe,
                    metadata={"stub": True})
    assert resolve_active_site(p, [stub]).source == "catalytic-triad"


def test_anchor_overrides_and_parses_formats():
    from PhiEx.analysis.site import resolve_active_site, parse_anchor
    assert parse_anchor("S12, Ser12 ,A:40,60") == [(None, "S", 12), (None, "S", 12),
                                                  ("A", None, 40), (None, None, 60)]
    site = resolve_active_site(_triad_protein(), [], anchor="S12,Ser12,H40")
    assert site.source == "anchor" and site.residues == (12, 40)
    mismatch = resolve_active_site(_triad_protein(), [], anchor="K12")
    assert mismatch.notes and "Ser12" in mismatch.notes[0]
    with pytest.raises(ValueError):
        resolve_active_site(_triad_protein(), [], anchor="999")


def test_fallback_pocket_does_not_move_a_known_site():
    from PhiEx.analysis.site import resolve_active_site, choose_active_pocket
    p = _triad_protein()
    site = resolve_active_site(p, [])
    near_but_off = Pocket(id="P1", center=(3.0, 3.0, 3.0), score=1.0)
    active, site2, pockets = choose_active_pocket(p, [near_but_off], site, real_detector=False)
    assert active.id == "SITE" and active.center == site.center
    active, _, _ = choose_active_pocket(p, [near_but_off], site, real_detector=True)
    assert active.id == "P1"           # a real P2Rank cavity within 8 Å is used


def test_no_site_uses_top_pocket_with_low_confidence():
    from PhiEx.analysis.site import choose_active_pocket
    p = _triad_protein(with_motif=False, with_acid=False)
    pk = [Pocket(id="P1", center=(1, 1, 1), score=0.4), Pocket(id="P2", center=(9, 9, 9), score=2.0)]
    active, site, _ = choose_active_pocket(p, pk, None, real_detector=False)
    assert active.id == "P2" and site.source == "pocket" and site.confidence == "low"


# ────────────────────────────────────────────────────────────────────────
# mutation scoring
# ────────────────────────────────────────────────────────────────────────

def test_mutation_llr_and_evaluator_use_structure_numbering():
    import numpy as np
    from PhiEx.adapters.esm2 import mutation_llr, AA20
    from PhiEx.pipeline.apx import build_mutation_evaluator
    lp = np.full((3, 20), -3.0)
    lp[1, AA20.index("S")] = -0.1          # position 2 strongly prefers Ser
    assert mutation_llr(lp, 2, "S", "A") == pytest.approx(-2.9)
    chain = [_res(101, "GLY", [("CA", (0, 0, 0))]), _res(102, "SER", [("CA", (1, 0, 0))]),
             _res(103, "GLY", [("CA", (2, 0, 0))])]
    pred = Prediction(value={"aa_logprobs": lp}, uncertainty=0.0, meta={})
    ev, desc = build_mutation_evaluator(pred, chain, None, None)
    y, sd = ev({"type": "mutation", "residue": 102, "from": "S", "to": "A"})
    assert y == pytest.approx(2.9) and "ESM-2" in desc


def test_mutation_panel_uses_residue_map_not_sequence_index():
    panel = generate_mutation_panel([102], "XXXX", ["A"], residue_aa={102: "S"})
    assert {"type": "mutation", "residue": 102, "from": "S", "to": "A"} in panel


def test_vina_pose_parser_keeps_heavy_atoms():
    from PhiEx.adapters.vina import _split_pdbqt_models
    txt = ("MODEL 1\n"
           "ATOM      1  C1  UNL     1       1.000   2.000   3.000  1.00  0.00     0.000 C \n"
           "ATOM      2  O1  UNL     1       1.500   2.000   3.000  1.00  0.00    -0.300 OA\n"
           "ATOM      3  H1  UNL     1       1.900   2.000   3.000  1.00  0.00     0.200 HD\n"
           "ENDMDL\n")
    models = _split_pdbqt_models(txt)
    assert len(models) == 1 and [a[1] for a in models[0]] == ["C", "O"]


def test_anchor_site_type_drives_default_ligand():
    from PhiEx.analysis.site import resolve_active_site
    from PhiEx.adapters.vina import default_ligand_for_site, PNP_ACETATE, ASCORBATE
    p = _triad_protein()
    # Ser12 hydrogen-bonded to His40 among the anchors → hydrolase → pNP-acetate
    s = resolve_active_site(p, [], anchor="S12,H40")
    assert s.kind == "serine-hydrolase" and default_ligand_for_site(s) is PNP_ACETATE
    # an anchor with no catalytic chemistry and no cofactor → no default at all
    s = resolve_active_site(p, [], anchor="G10")
    assert s.kind == "unknown" and default_ligand_for_site(s) is None
    # anchor next to a real heme → heme site → ascorbate
    fe = Atom(0, "FE", "FE", (38.0, 0, 0))
    heme = Cofactor(name="HEME", resname="HEM", atoms=(fe,), metal_atom=fe)
    s = resolve_active_site(p, [heme], anchor="G10")
    assert s.cofactor == "HEME" and default_ligand_for_site(s) is ASCORBATE


def test_dock_route_refuses_to_guess_a_ligand():
    from fastapi.testclient import TestClient
    import app.main as main
    from PhiEx.analysis.site import ActiveSite
    main.SESSION["protein"] = _triad_protein()
    main.SESSION["pockets"] = [Pocket(id="P1", center=(0, 0, 0), score=1.0, is_active_site=True)]
    main.SESSION["site"] = ActiveSite(center=(0, 0, 0), source="pocket", label="top pocket P1",
                                      confidence="low", kind="pocket")
    try:
        with TestClient(main.app, raise_server_exceptions=False) as c:
            r = c.post("/api/dock", json={"method": "vina"})
    finally:
        for k in ("protein", "pockets", "site"):
            main.SESSION.pop(k, None)
    assert r.status_code == 400 and "SMILES" in r.json()["error"]


# ────────────────────────────────────────────────────────────────────────
# ligand input: SMILES or name; failures are errors, not fake poses
# ────────────────────────────────────────────────────────────────────────

def test_ligand_input_accepts_smiles_and_names(monkeypatch):
    from PhiEx.adapters import vina as V
    lig = V.resolve_ligand_input("CCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]", "pNPB")
    assert lig.name == "pNPB" and lig.smiles.startswith("CCCC")
    monkeypatch.setattr(V, "_pubchem_smiles", lambda n: "CC(=O)O" if n == "acetic acid" else None)
    assert V.resolve_ligand_input("acetic acid").smiles == "CC(=O)O"
    with pytest.raises(ValueError, match="not a valid SMILES"):
        V.resolve_ligand_input("SMILES")


def test_real_docking_failure_raises_instead_of_fake_poses(monkeypatch):
    from PhiEx.adapters.vina import VinaAdapter
    a = VinaAdapter()
    a.is_stub = False
    monkeypatch.setattr(a, "_dock_real", lambda *x: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(RuntimeError, match="docking failed"):
        a.dock(_toy_complex(), Ligand(name="x", smiles="C"), Pocket(id="P", center=(0, 0, 0), score=1))


def test_dock_route_rejects_unparseable_ligand(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from PhiEx.adapters import vina as V
    from PhiEx.analysis.site import ActiveSite
    monkeypatch.setattr(V, "_pubchem_smiles", lambda n: None)
    main.SESSION["protein"] = _triad_protein()
    main.SESSION["pockets"] = [Pocket(id="P1", center=(0, 0, 0), score=1.0, is_active_site=True)]
    main.SESSION["site"] = ActiveSite(center=(0, 0, 0), source="pocket", label="p", confidence="low")
    try:
        with TestClient(main.app, raise_server_exceptions=False) as c:
            r = c.post("/api/dock", json={"method": "vina", "smiles": "SMILES"})
    finally:
        for k in ("protein", "pockets", "site"):
            main.SESSION.pop(k, None)
    assert r.status_code == 400 and "not a valid SMILES" in r.json()["error"]


# ────────────────────────────────────────────────────────────────────────
# hetero-group roles
# ────────────────────────────────────────────────────────────────────────

def test_hetero_groups_are_classified_by_role():
    from PhiEx.adapters.pdb import classify_het
    one = lambda el: [Atom(0, el, el, (0, 0, 0))]
    many = lambda n: [Atom(i, f"C{i}", "C", (i, 0, 0)) for i in range(n)]
    assert classify_het("HEM", many(43)) == ("HEME", "cofactor")
    assert classify_het("PLP", many(15)) == ("PLP", "cofactor")
    assert classify_het("ZN", one("ZN")) == ("Zn", "metal")
    assert classify_het("CA", one("CA")) == ("Ca", "ion")
    assert classify_het("NA", one("NA")) is None              # buffer ion
    assert classify_het("GOL", many(6)) is None                # cryoprotectant
    assert classify_het("NAG", many(14)) is None               # glycan
    assert classify_het("STI", many(37)) == ("STI", "ligand")  # inhibitor
    assert classify_het("ACP", many(31)) == ("AMP-PCP", "ligand")   # nucleotide analog: substrate site
    assert classify_het("ADP", many(27)) == ("ADP", "ligand")
    modified = [Atom(0, "N", "N", (0, 0, 0)), Atom(1, "CA", "C", (1, 0, 0)),
                Atom(2, "C", "C", (2, 0, 0))] + many(5)
    assert classify_het("CAS", modified) is None               # modified residue


def test_selenomethionine_reads_as_methionine():
    from PhiEx.adapters.pdb import parse_pdb_text
    mse = ("HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N\n"
           "HETATM    2  CA  MSE A   1       1.000   0.000   0.000  1.00  0.00           C\n"
           "HETATM    3 SE   MSE A   1       2.000   0.000   0.000  1.00  0.00          SE\n")
    p, cof = parse_pdb_text(mse, "x")
    assert [r.name for r in p.residues] == ["MET"] and cof == []
    assert ("SD", "S") in [(a.name, a.element) for a in p.residues[0].atoms]


def test_site_priority_cofactor_then_ligand_then_metal_never_ion():
    from PhiEx.analysis.site import resolve_active_site
    p = _triad_protein()
    def cof(name, role, xyz, n=1):
        atoms = tuple(Atom(i, f"X{i}", "C" if n > 1 else name.upper(), (xyz[0] + i, xyz[1], xyz[2]))
                      for i in range(n))
        metal = atoms[0] if role in ("metal", "ion") else None
        return Cofactor(name=name, resname=name.upper()[:3], atoms=atoms, metal_atom=metal,
                        metadata={"role": role})
    ca = cof("Ca", "ion", (5, 0, 0)); zn = cof("Zn", "metal", (6, 0, 0))
    lig = cof("STI", "ligand", (7, 0, 0), n=8); fad = cof("FAD", "cofactor", (8, 0, 0), n=10)
    assert resolve_active_site(p, [ca]).source == "catalytic-triad"     # Ca never defines it
    assert resolve_active_site(p, [ca, zn]).label == "Zn ion"
    s = resolve_active_site(p, [ca, zn, lig])
    assert s.source == "bound-ligand" and s.bound_ligand == "STI" and len(s.bound_ligand_xyz) == 8
    assert resolve_active_site(p, [ca, zn, lig, fad]).cofactor == "FAD"


def test_crystal_pose_rmsd_matches_atoms_by_element():
    from PhiEx.adapters.vina import crystal_pose_rmsd
    ref = [("C", (0.0, 0, 0)), ("O", (1.2, 0, 0)), ("C", (2.4, 0, 0))]
    shuffled = [Atom(0, "C2", "C", (2.4, 0, 0)), Atom(1, "C1", "C", (0.0, 0, 0)),
                Atom(2, "O1", "O", (1.2, 0, 0))]
    assert crystal_pose_rmsd(shuffled, ref) == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────────
# ligand screen
# ────────────────────────────────────────────────────────────────────────

def test_parse_ligand_list_formats():
    from PhiEx.pipeline.screen import parse_ligand_list
    got = parse_ligand_list("CCO\n# comment\n\ntributyrin; C8: CCCCCCCC(=O)O\n[CH3:1]O")
    assert got == [("", "CCO"), ("", "tributyrin"), ("C8", "CCCCCCCC(=O)O"), ("", "[CH3:1]O")]


def test_screen_ranks_and_reports_failures(monkeypatch):
    from PhiEx.pipeline import screen as S
    from PhiEx.core.state import DockingPose
    from PhiEx.analysis.site import resolve_active_site
    p = _triad_protein()
    site = resolve_active_site(p, [])
    scores = {"CCO": -3.0, "CCCCO": -5.0}

    class FakeVina:
        is_stub = False; backend = "fake"
        def __init__(self, **k): pass
        def dock(self, state, lig, pocket):
            atoms = (Atom(0, "C1", "C", site.center),)
            return [DockingPose(ligand=Ligand(name=lig.name, smiles=lig.smiles, atoms=atoms),
                                score=scores[lig.smiles], score_uncertainty=0.0, pose_id=0)]
    monkeypatch.setattr(S, "VinaAdapter", FakeVina)
    monkeypatch.setattr("PhiEx.adapters.vina._pubchem_smiles", lambda n: None)
    out = S.screen_ligands(ComplexState(protein=p), Pocket(id="SITE", center=site.center, score=0),
                           site, [("", "CCO"), ("butanol", "CCCCO"), ("", "notamolecule")])
    ok = [r for r in out["rows"] if not r["error"]]
    assert [r["name"] for r in ok] == ["butanol", "CCO"] and ok[0]["rank"] == 1
    assert ok[0]["site_hits"] == [12, 40] and ok[0]["efficiency"] == pytest.approx(1.0)
    assert out["n_failed"] == 1 and "not a valid SMILES" in out["rows"][-1]["error"]
    csv = S.rows_to_csv(out["rows"])
    assert csv.splitlines()[0].startswith("rank,name,score") and "butanol" in csv


# ────────────────────────────────────────────────────────────────────────
# candidate-gene triage
# ────────────────────────────────────────────────────────────────────────

def test_every_panel_smiles_parses_or_is_a_name():
    from rdkit import Chem, RDLogger
    from PhiEx.candidates.classes import all_panel_smiles
    RDLogger.DisableLog("rdApp.*")
    for key, value in all_panel_smiles():
        looks_like_smiles = any(ch in value for ch in "()=[]#@/\\") and " " not in value
        if looks_like_smiles:
            assert Chem.MolFromSmiles(value) is not None, (key, value)


def test_every_dock_class_has_a_negative_control():
    from PhiEx.candidates.classes import RULES
    for r in RULES:
        if r.track == "dock" and r.key not in ("nlr", "kinase", "gtpase"):
            assert any(p.role == "control" for p in r.panel), r.key


def test_classify_annotation_examples():
    from PhiEx.candidates.classes import classify
    cases = {
        ("Laccase; copper ion binding; hydroquinone:oxygen oxidoreductase activity", "AA1_3"): "laccase",
        ("WSC domain | Predicted effector (EffectorP; Apoplastic effector); CAZyme (GH128)", "GH128"): "b13_glucanase",
        ("WSC domain-containing protein", "AA5_1"): "cro",
        ("Prolyl oligopeptidase (peptidase S9); serine-type peptidase activity; triacylglycerol lipase", ""): "peptidase",
        ("Pectinesterase; Involved in maceration and soft-rotting", "CE8"): "pectinase",
        ("Methyltransferase activity; sterol 24-C-methyltransferase activity", ""): "sterol_mt",
        ("Lectin (Ssp1/fucose-specific type)", ""): "lectin_fucose",
        ("Lectin; Agglutinin", ""): "lectin_gal",
        ("Auxin responsive GH3 gene family (GH3); Indole-3-acetic acid-amido synthetase GH3.6", ""): "gh3_amido",
        ("L-type lectin-domain containing receptor kinase IX.1", ""): "kinase",
        ("WRKY transcription factor 40-related", ""): "tf",
        ("Nucleic acid binding; ribonuclease H activity; Reverse transcriptase", ""): "transposon",
        ("Uncharacterized | Predicted effector (EffectorP; Apoplastic effector)", "NA"): "effector_unknown",
        ("Hydrolase activity, hydrolyzing O-glycosyl compounds", "GH43_1"): "hemicellulase",
    }
    for (desc, fam), want in cases.items():
        assert classify(desc, fam).key == want, (desc, classify(desc, fam).key)


def test_parse_candidates_tsv_and_plain_list():
    from PhiEx.candidates.triage import parse_candidates
    tsv = ("gene_id\torganism\tuniprot\tdescription\tfamily\tsources\tlog2fc_24\tlog2fc_48\tlog2fc_96\n"
           "SS1G_00001\tS. sclerotiorum\tA7EYZ9\tAlcohol dehydrogenase (NADP(+))\t\tT1\t-1.5\t3.0\t0.5\n")
    c = parse_candidates(tsv)[0]
    assert c.gene_id == "SS1G_00001" and c.uniprot == "A7EYZ9" and c.log2fc == (-1.5, 3.0, 0.5)
    plain = parse_candidates("SS1G_00002 LysM protein\nglyma.01g000100\nSS1G_00002 duplicate\nnot a gene")
    assert [x.gene_id for x in plain] == ["SS1G_00002", "Glyma.01G000100"]


def test_accession_parsing():
    from PhiEx.candidates.resolve import accession_from
    assert accession_from("665079.A7EYZ9") == "A7EYZ9"
    assert accession_from("A0A0R0HLQ4") == "A0A0R0HLQ4"
    assert accession_from("SS1G_10565") is None


def _patch_triage_offline(monkeypatch, pdb_text):
    from PhiEx.candidates import triage as T
    from PhiEx.adapters import pdb as P
    monkeypatch.setattr(T, "resolve_uniprot", lambda g, h="": ("A7F474", "test"))
    monkeypatch.setattr(T, "choose_structure", lambda acc, cls: ("AF:A7F474", "AlphaFold"))
    monkeypatch.setattr(P, "fetch_protein", lambda pid, up: (*P.parse_pdb_text(pdb_text, name=pid), {}))


def test_triage_one_structure_only_on_alphafold_model(monkeypatch):
    from pathlib import Path
    from PhiEx.candidates.triage import triage_one, Candidate
    pdb_text = (Path(__file__).parent / "data" / "AF_A7F474.pdb").read_text()
    _patch_triage_offline(monkeypatch, pdb_text)
    row = triage_one(Candidate(gene_id="SS1G_00003", description="Alpha/beta hydrolase fold-3 domain-containing protein",
                               log2fc=(1.0, -2.5, None)), dock=False)
    assert row["error"] is None and row["class"] == "cutinase_esterase" and row["track"] == "dock"
    assert "Ser197" in row["site"] and row["site_confidence"] == "high"
    assert row["mean_plddt"] > 85 and row["site_plddt"] > 85
    assert row["peak_log2fc"] == -2.5 and row["direction"].startswith("down")
    assert any(p["role"] == "control" for p in row["panel"])


def test_triage_one_reports_margin_against_control(monkeypatch):
    from pathlib import Path
    from PhiEx.candidates import triage as T
    from PhiEx.pipeline import screen as S
    pdb_text = (Path(__file__).parent / "data" / "AF_A7F474.pdb").read_text()
    _patch_triage_offline(monkeypatch, pdb_text)
    def fake_screen(state, pocket, site, entries, exhaustiveness=8):
        rows = []
        for i, (label, value) in enumerate(entries):
            score = -4.0 if "control" in label else -5.0 - i * 0.1
            rows.append({"name": label, "input": value, "score": score, "efficiency": 0.3,
                         "site_hits": [197, 351], "error": None})
        return {"rows": rows}
    monkeypatch.setattr(S, "screen_ligands", fake_screen)
    row = T.triage_one(T.Candidate(gene_id="SS1G_00004", description="CAZyme (CE5); Cutinase"), dock=True)
    assert row["docked"] and row["best_role"] != "control"
    assert row["control_score"] == -4.0 and row["margin"] == pytest.approx(row["control_score"] - row["best_score"])


def test_unsuited_genes_are_not_docked(monkeypatch):
    from pathlib import Path
    from PhiEx.candidates.triage import triage_one, Candidate, sort_rows, rows_to_csv
    pdb_text = (Path(__file__).parent / "data" / "AF_A7F474.pdb").read_text()
    _patch_triage_offline(monkeypatch, pdb_text)
    tf = triage_one(Candidate(gene_id="Glyma.01G000200", description="WRKY transcription factor 40-related"), dock=True)
    assert tf["track"] == "unsuited" and not tf["docked"] and tf["warnings"]
    enz = triage_one(Candidate(gene_id="SS1G_00003", description="Alpha/beta hydrolase fold-3"), dock=False)
    assert [r["gene_id"] for r in sort_rows([tf, enz])] == ["SS1G_00003", "Glyma.01G000200"]
    assert rows_to_csv([enz]).splitlines()[0].startswith("gene_id,organism,class_title")
