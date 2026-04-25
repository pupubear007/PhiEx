# PhiEx Sandbox v0

A web-based workbench for real-time visual simulation of in planta and
in vitro biomolecular systems, framed in the φ → ∃ / s∃ ⟳ tφ vocabulary.
v0 is the minimum viable vertical slice — every stage
of the real pipeline runs end-to-end for a single example (plant
ascorbate peroxidase, APX), with ML integrated from stage 1 and not
bolted on later.

This README is framed in the framework's terms, not only in engineering
terms.

```
   ϕ        a theory       — a Calculator OR a LearnedModel
   ∃        an outcome     — a State (atoms, sequence, embedding)
   s∃       study          — a Trajectory, a sampled run
   tϕ       refined theory — a FittedTheory + the live LearnedModel registry
   i        iteration      — one step of the active-learning loop
   ↑↓       polarity       — deductive ϕ → ∃  vs. inductive s∃ ⟳ tϕ
```

The repo's directory layout, the `Calculator` and `LearnedModel`
protocols, and the reasoning ticker UI are all literal mirrors of those
five symbols. The point is not metaphor; the point is that any new tool
or model that gets added has a single, framework-determined place to go.

## ML as tφ

ML models *are* the refined theory tφ. Not "models we use for prediction"
— theory in the framework sense. Three properties make this work:

1. **They are first-class citizens.** Every ML model implements
   `LearnedModel`, the same protocol as the classical `Calculator`. The
   live session carries `theories: list[LearnedModel]`. ESMFold, ESM-2,
   P2Rank, MACE-OFF23, the surrogate, and (when the stub is replaced)
   DiffDock are all in there together.

2. **They expose uncertainty.** `predict()` returns
   `Prediction(value, uncertainty, meta)`. Uncertainty is required, not
   optional. The UI cannot render an ML readout without an error bar
   beside it. This is enforced by code shape (no `Optional` on
   `uncertainty`) and by the right-hand panel's CSS — every readout is
   `<value> ± <uncertainty>`.

3. **They are updated by iteration `i`.** The active-learning loop
   (`PhiEx/al/loop.py`) is what advances the theory. Each `i ← i+1`
   takes the current surrogate (one entry in `theories`), uses an
   acquisition function to pick the next perturbation, runs an
   evaluator, and refits. The UI's "run next iteration" button is
   literally that step.

The Bayesian active-learning loop is iteration `i`. UCB by default
(`acquisition.UCBAcquisition`); EI and Thompson sampling are
swap-targets at the same interface. The loop's history is what produces
the per-residue acquisition heatmap on the right-hand panel.

## Mapping pipeline stages to the framework

| Stage | Framework symbol | What runs | Where |
|---|---|---|---|
| 1. Structure acquisition | ϕ → ∃ (parallel paths) | RCSB PDB fetch (classical) + ESMFold (ML) | `adapters/pdb.py`, `adapters/esmfold.py` |
| 2. Function annotation | s | UniProt GO terms (classical) + ESM-2 embeddings + Foldseek aggregation (ML) → ϕ₀ | `adapters/esm2.py`, `adapters/foldseek.py` |
| 3. Pocket detection | ∃ | P2Rank (ML) + cofactor-aware active-site flag (heuristic) | `adapters/p2rank.py` |
| 4. Ligand placement | s∃ | AutoDock Vina (classical) or DiffDock (ML, stubbed in v0) | `adapters/vina.py`, `adapters/diffdock.py` |
| 5. Dynamics | ϕ → ∃ at molecular scale | OpenMM-AMBER bulk (classical) + MACE-OFF23 active-site core (ML) | `adapters/openmm_calc.py`, `adapters/mace.py` |
| 6. Active-learning loop | s∃ ⟳ tϕ, iteration i | UCB acquisition + GBR surrogate refit | `al/acquisition.py`, `al/loop.py`, `adapters/surrogate.py` |
| 7. Cross-check | s∃ ⟳ tϕ | trajectory-flagged residues vs. ESM-2 attention | `pipeline/apx.py` (final stage) |
| Visualisation | — | Mol* + reasoning ticker, prototype's visual vocabulary preserved | `static/index.html`, `static/js/app.js` |

## The example session (APX)

Run the test target after `make env && make weights` and you get the
worked APX session described in the spec:

```
make test
```

What happens, in framework terms:

1. **ϕ → ∃ structure acquisition.** Fetch 1APX from RCSB (classical) and
   run ESMFold on the same sequence (ML). Both end up in the session as
   `Protein` objects with `source = "pdb"` and `source = "esmfold"`
   respectively. The UI overlays them and reports Cα RMSD plus mean
   pLDDT ± SD.

2. **s function annotation.** Pull GO terms for UniProt P48534. Run
   ESM-2 over the sequence and Foldseek against AlphaFoldDB. Aggregate
   the structural neighbours' GO terms into ϕ₀, the starting functional
   hypothesis. Display in the left panel.

3. **∃ pocket detection.** P2Rank produces pockets with confidence
   scores. The cofactor-aware heuristic flags the pocket closest to the
   heme Fe atom as the active site. The UI marks it with ★.

4. **s∃ ligand placement.** Vina docks ascorbate into the active site
   pocket. We get 9 poses with scores and rank-order; DiffDock (ML
   alternative) is stubbed at the matching interface.

5. **ϕ → ∃ short ML/MM dynamics.** OpenMM runs 10 ps of implicit-solvent
   MD on the protein bulk. MACE-OFF23 evaluates the active-site region
   (≤ 50 atoms — heme + ligand + nearest residues by Cα distance to
   Fe). The UI plots energy(t) and reports MACE energy ± ensemble σ.

6. **s∃ ⟳ tϕ active learning over a mutation panel.** Generate ~10–20
   single-point mutations of active-site residues. Seed the surrogate
   with five cheap evaluations. Then three iterations of "rank by UCB →
   evaluate the top → refit". The UI's heatmap fills in. The loop
   converges on a small set of residues whose mutations produced the
   largest observed Δ.

7. **Cross-check.** Compare those residues against the residues
   highlighted by ESM-2's last-layer attention (column-wise mean
   importance — a documented heuristic). Where they agree, confidence
   ↑; where they disagree, the residue is flagged in red as a discovery
   candidate.

## Architecture

The most important design decision in this repo is the **adapter
layer**. Every external tool — RCSB API, ESMFold, ESM-2, Foldseek,
P2Rank, AutoDock Vina, DiffDock, OpenMM, MACE-OFF23, scikit-learn —
wears one of two coats:

```python
class Calculator(Protocol):                    # the classical coat
    def energy(self, state: ComplexState) -> float: ...
    def forces(self, state: ComplexState)  -> list[tuple[float, float, float]]: ...

class LearnedModel(Protocol):                  # the ML coat
    name: str; device: str; is_stub: bool
    def predict(self, *args, **kwargs) -> Prediction: ...
    def info(self) -> dict: ...
```

ML potentials (MACE-OFF23) wear *both*. Tool-specific types
(`Bio.PDB.Structure`, `openmm.app.Topology`, `esm.ESM2`) never escape
their adapter file. The core dataclasses are
`Protein`, `Cofactor`, `Ligand`, `Pocket`, `DockingPose`, `ComplexState`,
`Trajectory`, `Frame`, `FittedTheory`, `Prediction` — and that is the
full surface area that flows between adapters.

> A beginner call into a tool without thought is never acceptable; every
> adapter is the minimum code that translates between the tool's API and
> the core data model. This is the single most important architectural
> constraint.

The FastAPI backend wires routes one-per-stage; long-running stages
stream events via SSE so the UI ticker updates live. Heavy jobs
(`runners/docker.py`, `runners/simulator.py`, `runners/batch.py`) are
callable both in-process and as standalone scripts that read JSON from
stdin and write JSON to stdout — Slurm-ready by construction.

## Installation

### Prerequisites

* `mamba` (via [miniforge](https://github.com/conda-forge/miniforge)) —
  plain conda will not solve this stack in reasonable time.
* `make` — the entry point.
* For Mac M-series: nothing extra. PyTorch's MPS backend ships in the
  conda environment.
* For x86_64 Linux (MSI): nothing extra; v0 deliberately runs on CPU.
  CUDA is opt-in via `environment-cuda.yml` (not in v0).

### Apple Silicon (local M-series)

```
git clone <repo> && cd <repo>
make env       # creates conda env "PhiEx", ~20 min, ~5 GB
make weights   # ~5 GB download for ESMFold + ESM-2 + MACE-OFF23
make run       # http://localhost:8000
```

### MSI (x86_64 Linux)

```
git clone <repo> && cd <repo>
mkdir -p /scratch.global/$USER/PhiEx   # group-space env recommended
mamba env create -f environment.yml -p /scratch.global/$USER/PhiEx
mamba activate /scratch.global/$USER/PhiEx
make weights
PHIEX_DEVICE=cpu make run
```

The same `environment.yml` works on both platforms; `PhiEx.device`
picks the right runtime device. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
by `make run` so unsupported MPS ops fall through to CPU instead of
raising.

### Running it

```
make run            # FastAPI at http://localhost:8000, UI in browser
make test           # apx_end_to_end sanity check
```

## Known simplifications and stubs (called out explicitly)

These are the v0 corners. Each is a localised swap; the repo's adapters
do not bake them in elsewhere.

* **DiffDock** is a stub. The interface is correct — the file
  `adapters/diffdock.py` has the exact signature the real model will
  fill. The stub generates plausible poses with a confidence head so
  the rest of the pipeline can run.
* **Foldseek** is a stub for the APX example (canned plant-peroxidase
  hits). Replace `_predict_real` with a real `foldseek easy-search`
  call once the database is local.
* **OpenMM heme parametrization** is the gap that most often forces the
  stub MD path on real hardware. Production heme parametrization
  requires GAFF or a custom XML; v0 falls back to a synthetic harmonic
  trajectory if `ForceField.createSystem` fails on the cofactor. Logged
  as `[STUB]` in the ticker.
* **MACE region selection** uses the explicit heuristic
  *"heme atoms first, ligand atoms second, nearest residues by
  Cα-distance-to-Fe third, capped at 50 atoms"*. Documented in
  `adapters/mace.py::select_active_site_region`.
* **Surrogate perturbation encoding** concatenates a small set of
  hand-picked features (one-hot type, residue position, BLOSUM62 score,
  temperature offset, Morgan-256 fingerprint). Documented in
  `adapters/surrogate.py::encode_perturbation`. Replace with a learned
  encoder once the AL panel grows.
* **ESM-2 attention probing** is a column-wise mean of the last-layer,
  mean-over-heads attention. Real attention probing per-head is
  post-v0.
* **AL evaluator** in v0 is a cheap proxy: a MACE-region energy delta
  perturbed by a BLOSUM-driven scalar. The interface
  (`evaluator: dict -> (y, sigma)`) is the correct shape; production
  swaps in a real residence-time MD evaluator.

## Swap recipes

Each of these should be a small, localised change. If it isn't, the
adapter layer is wrong and a refactor is owed.

### Swap the PDB fetcher for ESMFold-only (genuinely novel proteins)

In `pipeline/apx.py::APXPipeline.run`, the conditional already exists:
when `protein.residues` is empty (no PDB hit), the rest of the pipeline
runs off the predicted structure alone. To force this path, pass an
unknown PDB id to `fetch_protein` — it returns `stub_apx_protein()`
which has only sequence, and ESMFold provides residues from there.

### Swap Vina for DiffDock once compute allows

In `app/main.py::api_dock` the `method` field already accepts
`"diffdock"`. Replace the body of
`adapters/diffdock.py::DiffDockAdapter._predict_real` (currently a stub)
with a real call to `diffdock.inference.run`. The pipeline does not
change.

### Swap MACE-OFF23 for ANI-2x or AIMNet2

Add `adapters/ani.py` (or `aimnet2.py`) that wears `Calculator` and
`LearnedModel` like `mace.py`. Replace `MACEAdapter()` with the new
class in two places: `pipeline/apx.py` (stage 5 & AL evaluator) and
`app/main.py::api_dynamics`.

### Swap the surrogate gradient-boosting model for a GNN

Drop `adapters/gnn_surrogate.py` next to `surrogate.py` with the same
`(fit, predict)` shape. The AL loop's only requirement is the
`encode_perturbation` interface — make the GNN consume the same dict
and you're done.

### Swap UCB for Thompson sampling or expected improvement

`acquisition.py` already exposes all three. Pass
`acquisition=ThompsonAcquisition(seed=0)` (or
`ExpectedImprovementAcquisition(incumbent=...)`) to
`ActiveLearningLoop`.

### Add a `ReaDDyAdapter` for the cell-scale layer

Create `adapters/readdy.py` wearing `Calculator`. ReaDDy's particles
become a new field on `ComplexState` (e.g. `particles: tuple[Particle, ...]`).
The simulator-shaped runner is `runners/readdy_simulator.py` mirroring
`runners/simulator.py`.

### Add a CUDA device branch in `device.py`

```python
def select_device(override=None):
    forced = override or os.environ.get("PHIEX_DEVICE")
    if forced == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PHIEX_DEVICE=cuda but no CUDA")
        return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

Add a separate `environment-cuda.yml` pinning `pytorch-cuda` to the
target node's driver. v0 deliberately does not select CUDA
automatically.

## Reasoning ticker

Every pipeline event is logged with one of five tags so the bottom
strip of the UI is a continuous trace:

* **ϕ → ∃** — deductive theory step (force calc, ESMFold inference,
  P2Rank pocket detection)
* **s∃** — sampled existence (a single trajectory event, a docking
  pose found)
* **tϕ** — refined theory (surrogate fit, AL update, ML model load)
* **i** — iteration boundary
* **···** — system messages

The ticker is the producer side; the FastAPI SSE endpoint
(`/events`) forwards to the browser.

## Repository layout

```
PhiEx/
├── environment.yml       authoritative conda spec (Mac MPS + Linux CPU)
├── Makefile              `make env`, `make weights`, `make run`, `make test`
├── device.py             single source of truth for device selection
├── engine.py             scaffolding skeleton (the toy ϕ → ∃ demo)
├── index.html            scaffolding prototype (visual vocabulary)
├── README.md
├── ROADMAP.md
├── app/
│   ├── __init__.py
│   └── main.py           FastAPI backend with SSE streaming
├── PhiEx/
│   ├── __init__.py
│   ├── device.py         re-export of root device.py
│   ├── ticker.py         reasoning-ticker pub/sub
│   ├── core/             State, Calculator, LearnedModel, Trajectory, FittedTheory
│   ├── adapters/         pdb / esmfold / esm2 / foldseek / p2rank / vina /
│   │                     diffdock / openmm_calc / mace / surrogate
│   ├── analysis/         contacts, RMSF, residence
│   ├── al/               UCB / EI / Thompson, the AL outer loop
│   ├── runners/          docker, simulator, batch  (in-process AND JSON-stdio)
│   ├── pipeline/         apx.py — the v0 example
│   └── tests/            apx_end_to_end.py
├── scripts/
│   └── download_weights.py
├── static/               frontend (Mol*, prototype's aesthetic preserved)
│   ├── index.html
│   └── js/app.js
└── data/                 caches (pdb, uniprot, scratch)
```

## What this is not

* Not a new simulation engine — every physics call is delegated to
  OpenMM, Vina, MACE-OFF23, etc.
* Not a benchmark — RMSD comparisons and AL evaluators in v0 are
  simplified; production runs need real MM/PB(GB)SA, real
  residence-time integration, real heme parametrization.
* Not a closed system — the adapter layer is the API. Adding a new
  tool means one new file and zero changes elsewhere.

## Pointers

* The framework essay (Wang 2025) is the conceptual scaffold; this
  README is the engineering surface.
* The prototype `index.html` at the repo root is the visual reference
  the new frontend extends.
* The toy `engine.py` is the architectural reference the core
  protocols extend.
