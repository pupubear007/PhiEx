# Getting Started — Phytologue Sandbox

A friendly walkthrough for someone opening this folder for the first time.

The full `README.md` is the engineering reference; it speaks in the
framework vocabulary (φ, ∃, tφ, …).  This file is the gentle on-ramp:
what to install, what to click, and what every panel on the screen
actually means.

---

## 1.  What is this thing?

Phytologue is a small web app that simulates a plant enzyme called
**ascorbate peroxidase (APX)** all the way from "we have a sequence of
amino acids" to "we have predicted which residues control the catalytic
chemistry."  It runs in your browser, talks to a Python backend on your
laptop, and shows the work as it happens.

You don't need a wet lab.  You don't need a GPU.  You don't need to
download 3 GB of weights to see something useful — every heavy step has
a "stub" that runs offline and produces the same shape of output.

If everything works on the first try, you'll see:

* a 3-D protein structure rendered in the middle of the screen,
* a stream of one-line log messages on the bottom (the "ticker"),
* a list of mutations the active-learning loop is trying, and
* eventually a ranked list of residues the system thinks matter.

---

## 2.  Install (Mac or Linux, ~5 minutes)

You need:

* **conda** or **mamba** (Anaconda or Miniforge work fine).
* **make** (Mac: ships with Xcode CLI tools; Linux: `apt install build-essential`).
* About 5 GB of free disk space if you want the real ML weights;
  ~500 MB if you stick to stubs.

In a terminal, from the folder this file is in:

```bash
make env       # creates a conda env called `phytologue`
make weights   # OPTIONAL — downloads ESMFold, ESM-2, MACE.  Skip for stub-only.
make run       # starts the web app at  http://localhost:8000
```

That's it.  Open your browser to <http://localhost:8000>.

> **First time?**  Skip `make weights` and run the app in stub mode.
> Everything will still work — the ticker will say `[STUB]` next to the
> ML steps so you know which numbers are real and which are synthetic.
> When you're ready, run `make weights` and restart.

If `make env` fails, the most common reasons are:

* No conda on PATH — install [Miniforge](https://github.com/conda-forge/miniforge).
* Mac without Apple Silicon — works fine, just slower; everything
  falls back to CPU automatically.

---

## 3.  Your first run, step by step

Once `make run` is going and you see something like
`Uvicorn running on http://0.0.0.0:8000`, open the browser tab.

You'll see three columns and a ticker at the bottom:

```
┌─────────────────┬────────────────────────────┬──────────────────┐
│  LEFT           │  CENTER                    │  RIGHT           │
│  controls (ϕ)   │  3-D viewer (∃)            │  readouts (tϕ)   │
│  blue tones     │  protein + ligand          │  red tones       │
│                 │                            │                  │
│  buttons:       │  Mol* renders the          │  GO terms        │
│  ▷ Run all      │  predicted structure       │  pockets         │
│  ▷ Per stage    │  with pLDDT colouring      │  AL history      │
└─────────────────┴────────────────────────────┴──────────────────┘
│  reasoning ticker — one line per pipeline event, live          │
└────────────────────────────────────────────────────────────────┘
```

**Click "Run pipeline" once.**  The ticker will start scrolling.  Each
line is tagged with one of these symbols:

| Tag | Meaning                                     | Colour |
|-----|---------------------------------------------|--------|
| ϕ   | a *theory* / model is making a prediction   | blue   |
| ∃   | an *outcome* / measurement                  | grey   |
| s   | a *trajectory* — many outcomes over time    | red    |
| t   | a refined theory has been *fit* from s∃     | yellow |
| i   | iteration boundary in the active-learning loop | dim |
| sys | plumbing / fallback messages                | grey   |

So a typical sequence looks like:

```
[i  ] ── stage 1  ϕ → ∃  structure acquisition ──
[ϕ  ] ESMFold ∃: predicted structure  pLDDT = 78 ± 6
[i  ] ── stage 2  s  function annotation ──
[s  ] ϕ0 hypothesis: top GO terms = peroxidase activity, heme binding, …
[i  ] ── stage 3  ∃  pocket detection ──
[ϕ  ] P2Rank ∃: 1 pocket(s); active-site flagged at P1
[i  ] ── stage 4  s∃  ligand placement ──
[i  ] ── stage 5  ϕ → ∃  short ML/MM dynamics ──
[ϕ  ] OpenMM step 1000/2000  t=1.00 ps  E=-299 eV
[i  ] ── stage 6  s∃ ⟳ tϕ  active-learning loop ──
[t  ] tϕ suggests: mutate H42 → A, predicted Δ = -1.5 ± 0.1 (acq=ucb)
[s  ] observed: y = -1.5 ± 0.1
[i  ] ── pipeline complete ──
```

The pipeline takes ~15 seconds in stub mode and ~3 minutes if you
downloaded all the real weights.

---

## 4.  What the seven stages do, in plain English

1. **Structure acquisition.**  Two parallel paths.  The classical path
   tries to download the experimental crystal structure (PDB code
   `1APX`).  The ML path predicts a structure from the amino-acid
   sequence using ESMFold.  We show you both and report how different
   they are (Cα RMSD).
2. **Function annotation.**  What does this protein *do*?  We pull GO
   terms from UniProt (classical), then independently embed the
   sequence with ESM-2 and find structurally-similar proteins via
   Foldseek (ML).  The two pieces of evidence are aggregated into a
   ranked list of likely functions.
3. **Pocket detection.**  Where on the surface could a small molecule
   bind?  P2Rank scores candidate pockets.  Among them, we flag the
   one closest to the heme iron as the **active site** — APX is a
   heme-iron enzyme, so this cofactor-aware step is built in.
4. **Ligand placement.**  Where does the substrate (ascorbate) sit
   inside the active site?  AutoDock Vina docks the molecule and
   returns up to ~9 candidate poses ranked by score.
5. **Short dynamics.**  Run a few picoseconds of molecular dynamics so
   the system can wiggle.  We use OpenMM with the AMBER force field
   for the bulk and (in real-weights mode) a MACE-OFF23 ML potential
   for the active-site core — that's the v0 stand-in for QM/MM.
6. **Active-learning loop.**  Now the framework's spine.  We pick a
   small panel of mutations (e.g. H42→A, W41→K, …), predict the effect
   of each with a fast surrogate model, run the most informative one
   for real, observe the answer, and update the surrogate.  Repeat 3
   times.  This is the **s∃ ⟳ tφ** loop.
7. **Cross-check.**  Compare the residues the simulation says matter
   against the residues ESM-2's attention map says matter.  Agreement
   = confirmation.  Disagreement = a discovery flag.

If everything goes right, the AL loop converges on **Trp41 / His42** —
the canonical distal-side catalytic pair of plant peroxidases.  The
system has rediscovered known biology, blind, from a sequence.

---

## 5.  Reading the right-hand panel

* **Predicted vs experimental** — pLDDT score (0–100, higher is more
  confident) and a Cα RMSD if both structures exist.
* **GO terms** — top function calls.  In APX the top three should be
  *peroxidase activity*, *heme binding*, *response to oxidative stress*.
* **Pockets** — pocket id, score, and whether it was flagged as the
  active site (look for the yellow `★`).
* **AL history** — one row per iteration showing the chosen mutation,
  the predicted Δ-energy ± uncertainty, and the observed value.  If
  predictions and observations track each other, the surrogate is
  learning the right thing.
* **Discovery flags** — residues where the simulation and ESM-2
  disagree.  These are the most interesting candidates for follow-up.

---

## 6.  Common things that go wrong

| Symptom                                   | Fix                                                    |
|-------------------------------------------|--------------------------------------------------------|
| `make env` fails with "conda: not found"  | Install Miniforge, then re-run.                        |
| Browser shows "connection refused"        | The server didn't start.  Check the terminal for errors. |
| All ticker lines say `[STUB]`             | You skipped `make weights`.  That's fine for a tour.   |
| ESMFold takes forever                     | Real weights are 3 GB; first inference is slow on CPU. |
| 3-D viewer is blank                       | Hard-reload (Cmd-Shift-R / Ctrl-Shift-R).              |
| AL history is empty                       | The pipeline didn't find an active site.  Check stage 3 ticker. |

---

## 7.  Where to go next

* **Try a different protein.**  Edit `phytologue/adapters/pdb.py` to
  change `APX_DEFAULT_PDB` and `APX_DEFAULT_UNIPROT`.  Anything with a
  cofactor and a small-molecule substrate will work; without a
  cofactor, the active-site heuristic just won't fire.
* **Tune the run.**  In the UI, change MD length (default 10 ps) and
  AL iterations (default 3).  Longer MD = more residence-time signal;
  more AL iterations = better surrogate.
* **Read the engineering README.**  Once the framework vocabulary
  feels familiar, `README.md` is a much denser tour of the same
  material.
* **Read `ROADMAP.md`.**  Each item is a small extension that unlocks
  new science — generative inverse design, real QM/MM, plant-cell
  membrane context, and so on.

---

## 8.  One-paragraph mental model

Think of every prediction as a **theory ϕ** and every observation as an
**outcome ∃**.  When you run dynamics or active learning, you produce a
**trajectory of outcomes s∃**, and you fit those outcomes back into a
better theory **tφ**.  The whole app is one big loop that keeps doing
this, with ML models filling in for tφ wherever a closed-form theory
would be too expensive.  The visual vocabulary on screen — blue for
deductive, red for inductive, yellow for refined-theory — exists to
make that loop legible.
