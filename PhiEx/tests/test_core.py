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

from PhiEx.core.state import Atom, Residue, Protein, Cofactor, Ligand, ComplexState
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

def test_resolve_working_structure_adopts_prediction_and_stub_heme():
    from PhiEx.pipeline import resolve_working_structure
    predicted = _toy_complex().protein
    seq_only = Protein(name="exp", sequence=predicted.sequence)
    protein, cofactors = resolve_working_structure(seq_only, [], predicted)
    assert protein is predicted
    assert len(cofactors) == 1 and cofactors[0].metadata.get("stub")


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
