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
   `Prediction(value, uncertainty, meta)`. Every adapter is expected to
   fill `uncertainty`; the UI cannot render an ML readout without an
   error bar beside it — every readout in the right-hand panel is
   `<value> ± <uncertainty>`. (The dataclass field itself still defaults
   to `None`, so this is a convention checked by the unit tests, not by
   the type system.)

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
   evaluate the top → refit". The objective is effect magnitude |Δ|, for
   both acquisition and the final ranking. The UI's heatmap fills in. The
   loop converges on a small set of residues whose mutations produced the
   largest observed |Δ|. Surrogate quality is reported out-of-sample:
   prospective RMSE, leave-one-out R², and ±1σ coverage.

7. **Cross-check.** Compare those residues (plus residues in contact
   with the ligand) against the residues highlighted by ESM-2's
   last-layer attention (column-wise mean importance — a documented
   heuristic). Where they agree, confidence ↑. A residue the trajectory
   flags but attention does not is marked in red as a discovery
   candidate; attention-only residues are listed but not flagged. The
   cross-check is skipped when ESM-2 runs as a stub.

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

Optional — real Foldseek and P2Rank instead of their stubs (any machine;
the M4 Pro with 48 GB is the comfortable one for the Foldseek PDB database):

```
cd ~/PhiEx && conda activate PhiEx
make tools     # foldseek (bioconda, or the official static binary) +
               # P2Rank 2.4.2 + openjdk 17, all inside the PhiEx env / ./tools
make weights   # now also builds the Foldseek PDB database under models/foldseek
make run       # registry panel: foldseek / p2rank without the STUB tag
```

When a candidate asks for AlphaFill (cofactor transplant) and AlphaFill has
no model for that accession, PhiEx loads the plain AlphaFold model instead,
says so in the log and the triage warnings, and finds the cofactor site from
sequence motifs (row 3b above).

Check by hand with `foldseek version` and `prank -v`.  A different P2Rank
release: `P2RANK_VERSION=2.5 make tools`.

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
make unit           # fast unit tests (pytest, no ML weights or network)
```

## Using it on your own protein

PhiEx is no longer tied to APX. For any structure (`1ABC`, or `AF:<UniProt>`
for an AlphaFold model) it decides **where the active site is** before
anything else, and every later stage — pocket, docking box, MD analysis,
MACE region, mutation panel, discovery flags — is built around that site.

**Site resolution, in priority order** (`PhiEx/analysis/site.py`):

| # | source | when | confidence |
|---|--------|------|------------|
| 1 | **your anchor** — the *Site anchor* box / `anchor` API field, e.g. `S197,H351,E314`, `Ser197`, `A:197` | always wins | `user` |
| 2 | **real cofactor** in the structure (heme Fe, FAD, NAD …) | crystal structures with HETATM cofactors | `high` |
| 3 | **catalytic triad/dyad** found geometrically: Ser/Cys–His–Asp/Glu hydrogen-bond network (≤ 3.6 Å), boosted by a G‑x‑S‑x‑G nucleophile-elbow motif | serine / cysteine hydrolases, esterases, lipases, proteases | `high` with motif + acid, else `medium` |
| 3b | **cofactor-binding motif** in an apo model (AlphaFold without the cofactor): class-I SAM methyltransferase motif I `hh(D/E)hGxGxG`, Rossmann dinucleotide loop `GxGxxG` (NAD(P)/FAD); motif must be ordered (pLDDT ≥ 70). A medium-confidence triad without the G-x-S-x-G motif ranks below this | SAM methyltransferases (ERG6 …), FAD/NAD(P) oxidoreductases when AlphaFill has no model; default probe for a SAM site is SAM itself | `medium` |
| 4 | **top P2Rank pocket** | nothing above found | `medium` (`low` with the geometric fallback) |
| 5 | protein centroid | nothing at all | `low` |

No cofactor is ever invented: the v0 code used to place a synthetic heme
next to residue 42 (APX's catalytic His) on *every* protein, which put the
whole downstream analysis in the wrong place for anything that is not APX.

**Ligand:** give a SMILES in the *Ligand SMILES* box (`ligand_smiles` in
the API). Without one, a default is chosen from the site type: hydrolase
sites get p‑nitrophenyl acetate (the standard esterase assay substrate),
everything else ascorbate.

**What each stage now computes, and how real it is**

| stage | real when | otherwise |
|-------|-----------|-----------|
| docking | `vina` + `meeko` + `rdkit` installed — receptor H added by meeko, 22 Å box on the site, cofactors near the site kept as rigid receptor atoms | labelled `[STUB]` poses |
| pose contacts | real docking ran — residues ≤ 4 Å of the best pose | — |
| MD | OpenMM: protein-only, chain holding the site, H added (pdbfixer rebuilds missing side-chain atoms), OBC2 implicit solvent, 2 fs; frames mapped back by (chain, residue, atom) so RMSF and contacts work. Uses the OpenCL GPU platform on Apple Silicon when available | stub harmonic trajectory, labelled |
| residence time | — the ligand is **not** simulated (no small-molecule force field in v0); reported as *n/a* instead of a fake number | — |
| mutation scores (AL objective) | ESM-2 loaded: zero-shot mutation effect, −LLR = log p(wt) − log p(mut) at that position (Meier et al. 2021). Larger = more disruptive. It does **not** model ligand binding | BLOSUM heuristic, labelled `SYNTHETIC` |
| discovery flags | residues supported by AL / MD contacts / docked-pose contacts but not by ESM-2 attention | skipped when ESM-2 is a stub |

**Hetero groups in the structure** (`classify_het` in `adapters/pdb.py`) are
sorted by role: known **cofactors** (heme, FAD/FMN, NAD(P)(H), PLP, TPP, CoA,
SAM/SAH, ATP/ADP/GTP/GDP, Fe–S clusters, molybdopterin, cobalamin, biotin,
chlorophylls, …) and **catalytic metals** (Zn, Fe, Cu, Mn, Ni, Co, Mo, …)
stay in the receptor and can define the site; **structural ions** (Ca, Mg)
stay in the receptor but never define it; **co-crystallized ligands**
(inhibitors, substrate analogs) define the site but are removed from the
receptor — leave *Ligand* empty and PhiEx fetches the ligand's SMILES from
RCSB and **re-docks it**, reporting the pose RMSD to the crystal (≤ 2 Å means
docking reproduces the experimental binding mode). Waters, buffers,
cryoprotectants, glycans and modified residues are ignored; MSE is read as
Met. For AlphaFold models use the `AFILL:<UniProt>` entry — AlphaFill
transplants cofactors and ligands from homologous structures.

**Ligand screen** (stage 4b, `POST /api/dock/batch`, CSV at
`/api/dock/batch.csv`): paste up to 50 ligands, one per line — a SMILES, a
compound name (PubChem), or `label: SMILES-or-name`. Each is docked into the
same site with the same settings and the table ranks them by Vina score with
ligand efficiency (−score / heavy atoms), the site residues each pose touches,
and distance to the site centre. Include a known substrate and a negative
control: Vina scores carry ~±2 kcal/mol error, so trust the ranking and the
contacts, not the absolute numbers.

**How to read the discovery flags now:** they are site residues that the
structure-based evidence (docked-pose contacts, MD contacts, mutation
sensitivity) points to but sequence-level attention does not — candidates
for mutagenesis around a site you trust. They are only as good as the
site: check the *site* line in the Pockets panel (source + confidence)
before using them.

## Sequence viewer (AlphaFold-DB style)

The strip under the two 3D panels (**▸ sequence**) shows the working chain
the way the AlphaFold DB entry page does, with PhiEx's own results added:

| track | source |
|---|---|
| sequence, coloured by pLDDT | AlphaFold / AlphaFill model B-factors (dark blue > 90, light blue 70–90, yellow 50–70, orange < 50); grey for experimental structures |
| UniProt topology / domains / sites / PTM | UniProt entry features (signal peptide, TM, domains, active and binding sites, disulfides, glycosylation); mapped by alignment if the structure is not numbered like UniProt |
| PhiEx site / pocket | the resolved active site (stage 3) and the residues lining the active pocket |
| catalytic motifs | G-x-S-x-G nucleophile elbow, HGG/HGA oxyanion hole, detected Ser/Cys–His–Asp/Glu triads |
| docked-ligand contacts | residues ≤ 4 Å from the best Vina pose (stage 4) |
| ESM-2 constraint | per-position mean −LLR of the 19 substitutions (stage 2); bright = the language model rejects changes there |
| isolate variants | reserved for the RNA-seq variant step (not yet filled) |

* hover → residue, pLDDT and every feature covering it
* click or drag → those residues drawn as yellow sticks in the lower 3D panel, camera centred on them; **clear 3D** removes them
* **PAE** → AlphaFold predicted-aligned-error map (green = confident relative placement); drag a box on it to highlight both residue ranges in 3D — a quick way to see whether a domain or loop is placed reliably relative to the active site
* **↻ tracks** rebuilds after stages 2–4 (also done automatically)

Endpoints: `GET /api/sequence-track`, `GET /api/pae`.

## Candidate-gene triage (transcriptomics → structure → ligands)

For a list of candidate genes from an expression study (e.g. DESeq2 hits,
network hubs, predicted effectors/CAZymes), PhiEx asks, gene by gene, *what
can structure-based modelling actually say about this protein?*

1. **Class and track** (`PhiEx/candidates/classes.py`): the annotation and
   CAZy family are matched against explicit rules. Each class has a track —
   **dock** (enzyme or binding protein with known small-molecule partners),
   **structure** (no known ligand: use structure, pockets and Foldseek
   neighbours to build a function hypothesis) or **unsuited** (transcription
   factors, transposon proteins, membrane transporters, protein–protein
   binders, housekeeping machinery) — plus the cofactor its site needs and a
   ligand panel with roles (substrate / product / inhibitor / cofactor /
   **negative control**).
2. **UniProt**: from the STRING id when given, else the UniProt REST search
   (`SS1G_xxxxx` → organism 665079; `Glyma.xxGxxxxxx` → ORF name
   `GLYMA_xxGxxxxxx`, taxon 3847). Cached in `data/candidates/`.
3. **Structure**: experimental PDB if one exists, else **AlphaFill** when the
   class needs a cofactor (heme, Cu, FAD, NAD(P), PLP, Fe, Mn …), else
   AlphaFold, else SWISS-MODEL.
4. **Site and confidence**: site resolution as above; mean pLDDT and the
   pLDDT of the site residues (site pLDDT < 70 is flagged).
5. **Optional docking**: the class panel is docked into the site and the row
   reports the best binder and its **margin over the negative control**
   (kcal/mol). The margin, not the raw Vina score, is what to rank on.

In the UI: *CANDIDATE GENES* (left) — load a TSV with a `gene_id` column
(optional `uniprot`, `description`, `family`, `sources`, `log2fc_24/48/96`)
or paste gene IDs; filter by organism, source table, track; tick *dock
panels* for docking (about a minute per gene). Results stream into
*CANDIDATE TRIAGE* (right); click a row to open its structure and site, with
its ligand panel loaded into the screen list. CSV: `/api/candidates.csv`.
Results are cached per gene, so a long list can be stopped and resumed.

Candidate files and caches live in `data/candidates/`, which is git-ignored:
unpublished candidate lists never leave the machine through the repository.

## Known simplifications and stubs (called out explicitly)

These are the v0 corners. Each is a localised swap; the repo's adapters
do not bake them in elsewhere.

* **DiffDock** is a stub. The interface is correct — the file
  `adapters/diffdock.py` has the exact signature the real model will
  fill. The stub generates plausible poses with a confidence head so
  the rest of the pipeline can run.
* **Foldseek** runs for real when `foldseek` is on PATH and a database
  exists at `models/foldseek/pdb` (`make weights` downloads the PDB
  database) or `$PHIEX_FOLDSEEK_DB`. It searches chain A only, drops the
  query's own entry, and maps hits to GO terms via the PDBe SIFTS API
  (cached in `models/foldseek/go_cache.json`). Otherwise it uses canned
  APX neighbours.
* **ESMFold** loads fair-esm if `openfold` is installed, otherwise the
  Hugging Face `facebook/esmfold_v1` port (no openfold). It needs ≥ 32 GB
  RAM (ESM-2 3B inside); below that it stays a stub unless
  `PHIEX_ESMFOLD=force`. `PHIEX_ESMFOLD=off` disables it. Only one chain
  is folded (`PHIEX_ESMFOLD_CHAIN`, default first chain), and Cα RMSD
  is computed after Kabsch superposition.
* **MACE-OFF23** runs on CPU by default (`PHIEX_MACE_DEVICE` to change):
  the checkpoint is float64, which MPS cannot hold during loading.
* **MD is protein-only.** The docked ligand and cofactors are held at
  their input coordinates; residence time is therefore not computed.
  Parametrizing the ligand (GAFF/OpenFF via `openmmforcefields`) is the
  next step for ligand dynamics.
* **MACE region selection**: ligand atoms first, cofactor atoms within
  6 Å of the site second, nearest residues by Cα distance to the site
  centre third, capped at 50 atoms; elements outside MACE-OFF23's
  training set (e.g. Fe) are dropped. The region has no hydrogens or
  caps, so its absolute energy is not physically meaningful — compare
  values only between runs of the same system.
* **Surrogate perturbation encoding** concatenates a small set of
  hand-picked features (one-hot type, residue position, BLOSUM62 score,
  temperature offset, Morgan-256 fingerprint). Documented in
  `adapters/surrogate.py::encode_perturbation`. Replace with a learned
  encoder once the AL panel grows.
* **ESM-2 attention probing** is a column-wise mean of the last-layer,
  mean-over-heads attention. Real attention probing per-head is
  post-v0.
* **AL evaluator** in v0 is a cheap proxy: a BLOSUM-driven |Δ| per
  mutation, with the MACE-region ensemble σ as observation noise. The
  structure is not mutated, so AL results reflect the proxy, not physics. The interface
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

### Add a CUDA device branch in `PhiEx/device.py`

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
├── device.py             shim re-exporting PhiEx/device.py
├── engine.py             scaffolding skeleton (the toy ϕ → ∃ demo)
├── index.html            scaffolding prototype (visual vocabulary)
├── README.md
├── ROADMAP.md
├── app/
│   ├── __init__.py
│   └── main.py           FastAPI backend with SSE streaming
├── PhiEx/
│   ├── __init__.py
│   ├── device.py         single source of truth for device selection
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
