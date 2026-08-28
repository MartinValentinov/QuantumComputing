# Learning Quantum Computing

## Overview
This repository documents my journey into **Quantum Computing** — a field that combines physics, mathematics, and computer science to solve problems in fundamentally new ways.

The goal of this repo is to track my progress as I move from basic quantum concepts to building and simulating quantum algorithms.

---

## What This Repository Is

This is a **learning lab** where I:

- Study how quantum computers work
- Write quantum programs and simulations
- Implement and test quantum algorithms
- Explore how quantum information differs from classical computing

---

## Topics Covered

- Qubits and quantum states  
- Superposition and entanglement  
- Quantum gates (X, Y, Z, H, CNOT, etc.)  
- Quantum circuits  
- Measurement and probability  
- Bloch sphere representation  
- Quantum teleportation and Bell states  
- Quantum algorithms:
  - Deutsch–Jozsa
  - Grover’s search
  - Quantum Fourier Transform (QFT)
  - Shor’s algorithm (conceptual)

---

## Tools & Technologies

- **Python**
- **Qiskit** (IBM Quantum SDK)
- **NumPy & SciPy** (for matrix math)
- **Jupyter Notebooks** (experiments and visualization)
- Quantum simulators (local + IBM Quantum cloud)

---

## What I’m Learning

- How quantum states are represented as vectors  
- How gates act as matrices on qubits  
- How entanglement enables non-classical behavior  
- How quantum speedups are achieved  
- How noise and decoherence affect real hardware  

---

## Why This Repo Exists

Quantum computing is one of the most advanced areas of modern computing.  
This repository exists to:

- Build a **strong mathematical and physical foundation**
- Gain **hands-on experience** with real quantum frameworks
- Understand how **future computers will solve problems differently**

---

## Notes

- This is not a finished product — it is a **learning journey**
- Code may evolve or be rewritten as my understanding improves
- Mistakes are part of the process

---

# The codebase

The front door is **QDE** — pick a blood sample, get a matched compound and
what it is made of. The detailed research tools live behind it at `/full`.

The learning journey above is what this repo started as. It has since grown a
working toolchain — three packages under `src/`, built as independent systems
that share a repo rather than a dependency graph.

| Package | What it does | Status |
|---|---|---|
| `qc_research` | Pulls biomedical databases into one queryable knowledge graph | **working** |
| `qc_quantum` | Quantum vs classical electronic-structure benchmarking | **working** |
| `qc_design` | Hybrid agent: an LLM that reasons over the graph and calls quantum tools | **tool layer working** |
| `qc_ui` | Local web UI over the graph — stdlib only, no build step | **working** |
| `qc_model` | A molecular potency model built from first principles | **working** |
| `qc_sample` | Simulated blood panels matched to disease signatures | **working** |
| `qc_draw` | SMILES → 2D structure, functional-group detection | **working** |

## qc_research — the research graph

Stdlib only. No install, no API keys, no accounts.

```bash
PYTHONPATH=src python3 -m qc_research ingest "pancreatic ductal adenocarcinoma"
```

That single command:

1. resolves the disease name to an ontology ID (**Open Targets**)
2. pulls the top associated targets, with evidence breakdown and tractability
3. maps each target to **ChEMBL** via its UniProt accession
4. pulls measured potencies (IC50/Ki/Kd/EC50) and curated mechanisms of action
5. pulls compound structures and physicochemical properties
6. pulls **ClinicalTrials.gov** records *including terminated and withdrawn ones*
7. pulls review literature per target from **PubMed**

into a single SQLite file with full provenance on every row.

### Then ask it things

```bash
PYTHONPATH=src python3 -m qc_research landscape "pancreatic ductal"
PYTHONPATH=src python3 -m qc_research gaps      "pancreatic ductal"
PYTHONPATH=src python3 -m qc_research failures  "pancreatic ductal"
PYTHONPATH=src python3 -m qc_research compounds KRAS
```

`landscape` is the triage table — association score crossed with how much
chemical matter exists:

```
symbol      assoc  actives  best_pX  drugs  phase  SM AB  name
--------------------------------------------------------------
KRAS        0.512      121     9.00      2      4  y  y   KRas proto-oncogene, GTPase
TP53        0.477       14     8.70      2      3  y  -   tumor protein p53
TOP1        0.461      176     9.53     24      4  y  -   DNA topoisomerase I
SMAD4       0.427        0     0.00      0      0  y  -   SMAD family member 4
```

`gaps` is the query the whole package exists for: targets where the biology is
well evidenced, the target is small-molecule tractable, **and no drug has ever
reached the clinic**.

`failures` buckets the free-text `whyStopped` field across every trial in the
indication. Modelling only successes teaches you nothing about risk.


### Target deep-dive

```bash
PYTHONPATH=src python3 -m qc_research dossier KRAS --disease "pancreatic ductal adenocarcinoma"
```

A dossier answers the question a project team actually asks before committing
people to a target: *should we work on this, and what will kill us?* Four things
decide that, and they come from four different sources:

| Question | Source | Fields |
|---|---|---|
| Is the biology real? | Open Targets | association score, evidence-type breakdown |
| Can we drug it? | Open Targets + ChEMBL | tractability, chemical probes, measured actives |
| Will it be toxic? | Open Targets | `isEssential`, LOEUF constraint, safety liabilities, mouse KO |
| Has anyone tried? | ChEMBL + ClinicalTrials.gov | curated mechanisms, max phase, trial outcomes |

The assessment at the end is **rule-based, not generated**. Every flag traces to
a named field, so you can argue with it. Three real examples:

```
KRAS   precedented -- competition, not feasibility, is the question
       + 2 approved drugs (sotorasib, adagrasib), 69 compounds at pChEMBL>=7
       - pan-essential in DepMap; LOEUF bin 0; 2404 disease associations

SMAD4  tractable in principle, nothing made yet -- highest risk, highest novelty
       + small-molecule tractable
       - zero measured chemical matter, no probe, no clinical history

LRRK2  chemically enabled but not yet approved, best phase 2
       + 338 compounds at pChEMBL>=7, 5 high-quality probes
       - nothing approved yet
```

Flags `--markdown`, `--json`, `--out FILE`, `--refresh`, `--no-literature`.
Targets not yet in the graph are fetched on demand — `dossier` works standalone.


### The investigation loop

```bash
PYTHONPATH=src python3 -m qc_research investigate "pancreatic ductal adenocarcinoma" --strategy novel
```

Three passes, because that is honestly how the work goes:

| Pass | Cost | What it does |
|---|---|---|
| 1 · Survey | cheap | Score every target in the graph from data already held |
| 2 · Deep dive | expensive | Full dossier for the top N only — **safety data first appears here** |
| 3 · Re-rank | cheap | Re-score with the new evidence and report what moved |

Pass 3 exists because the deep dive routinely disqualifies what the survey
liked. *What changed* is the finding, so it is reported explicitly:

```
WHAT CHANGED AFTER THE DEEP DIVE
  ^ RRM2       rank 5 -> 4   (pan-essential, LOEUF bin 0)
  v RRM1       rank 4 -> 5   (pan-essential, LOEUF bin 0, 3 safety liabilities)
```

**Ranking is an explicit weighted rubric, not a model.** Strategies differ only
in how they weight the same five components — which is the honest way to
express portfolio strategy:

| Strategy | Weights | Finds |
|---|---|---|
| `novel` | evidence .30, novelty .30, tractability .20 | untaken opportunities |
| `fast` | chemistry .35, precedent .25, evidence .20 | shortest path to a molecule |
| `balanced` | evidence .30, tractability .20, chemistry .20, safety .20 | no strong prior |

They genuinely diverge. Same disease, same graph:

```
novel  ->  POLD1, TUBA1C, SMAD4, RRM2, RRM1     (all undrugged)
fast   ->  ERBB4, ACVR1, CD274, KRAS, TOP1      (all clinically precedented)
```

There is no LLM anywhere in this loop. Every component is 0–1 and traces to a
named database field, so you can read the rubric, tune it, and disagree with
it. The output is structured so a language model can write narrative over it
later — but the judgement stays in code, where it can be tested.

Flags: `--strategy`, `--top N`, `--markdown`, `--out FILE`, `--no-literature`.
Auto-ingests the disease first if the graph does not have it yet.

### Design rules

- **Provenance is not optional.** Every row carries `source` and `fetched_at`.
- **No LLM in the ingest layer.** The agent reasons *over* this graph; it does
  not hallucinate it.
- **Cache everything.** HTTP responses land in `data/cache/`, writes are
  `INSERT OR REPLACE`. Re-running is cheap and idempotent.
- **SQLite is deliberate.** At 10^5–10^6 rows it is entirely adequate, and
  moving to DuckDB or Postgres later is a driver change, not a rewrite.

Set `NCBI_API_KEY` to raise the PubMed rate limit from 3/s to 10/s. Optional.

---

## qc_quantum — benchmarking, not magic

```bash
PYTHONPATH=src qc/qiskit-modern/bin/python -m qc_quantum.benchmark H2
```

```
H2O  (3 atoms, sto3g, 10 qubits, 551 Pauli terms)
method                  E (Ha)    err vs ref  chem acc      sec
HF                -74.96440482     +0.051024        no     0.01
MP2               -75.00091684     +0.014512        no     0.03
CCSD              -75.01530778     +0.000121       yes     0.01
CCSD(T)           -75.01537643     +0.000052       yes     0.00
FCI               -75.01542879     +0.000000       yes     0.02
VQE/uccsd         -75.01524959     +0.000179       yes  4044.22
```

VQE reaches chemical accuracy. It takes **4,044 seconds** to do it, on a
noiseless simulator, for three atoms in a minimal basis. FCI — exact in this
basis — takes 0.02s, and CCSD(T) gets a *better* answer in under 0.01s.
That ratio is the finding, and it is why this package is scoped as a benchmark
harness rather than a design engine.

Runs VQE *and* the classical ladder (HF → MP2 → CCSD → CCSD(T) → FCI) on the
same molecule, reports error against an exact reference and wall time for each.

Two deliberate changes from the original notebook code:

- **UCCSD replaces `TwoLocal`.** A hardware-efficient ansatz ignores chemistry
  and its gradients vanish exponentially with qubit count (*barren plateaus*).
  It gives plausible answers for H2 and then silently stops working.
- **Parity mapping replaces Jordan-Wigner.** Two-qubit reduction saves 2 qubits
  on closed-shell systems, and qubit count is the binding constraint.

### Honest scope

There is no quantum advantage for drug design today, and there will not be one
before fault-tolerant hardware. Measurement cost scales as O(N⁴) Pauli terms;
classical CCSD(T) is both faster and more accurate on every molecule you can
currently run. The canonical "quantum will solve this" target — FeMoco — has
resource estimates in the millions of physical qubits.

So this package is scoped as what it honestly is: **a benchmark harness that
measures where the classical/quantum crossover actually sits.** That is a real,
defensible contribution. Claiming it designs drugs is not.

Next step here: replace VQE with SQD (sample-based quantum diagonalization),
which survives hardware noise in a way VQE does not.

---

## Layout

Each directory has one job.

| Path | What lives here | Why it is separate |
|---|---|---|
| `src/qc_research/` | The research graph. Stdlib only. | The core product. No install, no keys, runs anywhere. |
| `src/qc_research/sources/` | One adapter per public API — `opentargets`, `chembl`, `clinicaltrials`, `pubmed`. | Each returns plain dicts matching the DB schema. When an API changes, exactly one file changes. No orchestration here. |
| `src/qc_research/db.py` | Schema, migrations, provenance. | Every row carries `source` + `fetched_at`. The schema is the contract everything else agrees on. |
| `src/qc_research/http.py` | Caching, throttling, retries. | Politeness and caching belong in one place, not scattered across four adapters. |
| `src/qc_research/pipeline.py` | Orchestration — the order the sources are called in. | Sequencing logic, separate from parsing logic. Resumable and idempotent. |
| `src/qc_research/queries.py` | Cross-source questions (`landscape`, `gaps`, `failures`). | If a question needs only one API call, it does not belong here. This file is the proof the integration was worth doing. |
| `src/qc_research/dossier.py` | Per-target assembly + rule-based assessment. | Judgement lives here, isolated so it can be argued with and tested. |
| `src/qc_research/investigate.py` | The three-pass loop and the scoring rubric. | Strategy weights live in one dict at the top of the file. Tuning the rubric never means touching data code. |
| `src/qc_research/render.py` | Terminal and Markdown output. | Presentation only, zero logic. Swap it for HTML without touching anything else. |
| `src/qc_research/cli.py` | Argument parsing and dispatch. | Thin. Every command is a few lines calling into the modules above. |
| `src/qc_quantum/` | Quantum chemistry benchmarking. Needs the qiskit venv. | Totally independent of `qc_research`. Different deps, different purpose, different maturity. |
| `src/qc_quantum/solvers/` | `vqe.py` (quantum) and `classical.py` (HF→CCSD(T)→FCI). | Same `EnergyResult` interface, so the benchmark treats them identically. That symmetry is the whole point. |
| `src/qc_design/` | Phase 2: multi-objective molecule optimisation. | Empty. Will consume `qc_research` output; will not depend on `qc_quantum`. |
| `notebooks/learning/` | Bloch spheres, Bell states, teaching material. | Learning artifacts, not library code. Kept out of `src/` so they never become an import dependency. |
| `scripts/` | One-off and legacy scripts. | `_legacy_3Dmolecule.py` is the original monolith, kept for reference until the port is trusted. |
| `data/` | SQLite graph + HTTP cache. **Gitignored.** | Fully regenerable from the APIs. Never commit it. |

```
src/qc_research/     research graph  (stdlib only)
  sources/           one adapter per public API
  db.py              schema + provenance
  http.py            cache, throttle, retry
  pipeline.py        orchestration
  queries.py         cross-source questions
  dossier.py         per-target deep dive + assessment
  investigate.py     survey -> deep dive -> re-rank loop
  render.py          text / markdown output
src/qc_quantum/      quantum chemistry
  solvers/           vqe.py, classical.py
  benchmark.py
src/qc_design/       (empty -- phase 2)
notebooks/learning/  Bloch sphere, Bell states, teaching material
data/                gitignored: cache + sqlite
```

## Setup

`qc_research` needs nothing. For `qc_quantum`, the existing venv works:

```bash
source qc/qiskit-modern/bin/activate
```

---

## qc_design — the hybrid agent

An LLM that plans; tools that measure. The division of labour is the point:

| Component | Supplies | Never does |
|---|---|---|
| Claude | what to look at, in what order, when to stop | state a fact it did not read from a tool |
| `qc_research` | evidence — scores, potencies, trial outcomes | guess |
| `qc_quantum` | physics — energies, qubit counts | pretend to scale to drug-sized molecules |

```bash
PYTHONPATH=src python3 -m qc_design tools      # the 10-tool surface
PYTHONPATH=src python3 -m qc_design dry-run    # exercise tools, no LLM, no API key
PYTHONPATH=src python3 -m qc_design ask "Which undrugged target in pancreatic cancer is the best bet?"
```

### Why the tool layer is the real deliverable

`dry-run` executes a fixed tool sequence with no model in the loop. That makes a
bad run bisectable into *the tools are wrong* versus *the model chose badly* —
two very different problems that look identical from the outside.

```
tool                      status       sec    bytes
graph_stats               ok          0.00      151
target_landscape          ok          0.01     1497
opportunity_gaps          ok          0.01     1489
trial_failures            ok          0.00     2307
estimate_quantum_cost     no-deps     0.00      263
```

### Three rules the tool layer follows

- **Errors are return values, not exceptions.** A tool that raises kills the
  loop; a tool that returns `{"error": ..., "remedy": ...}` lets the model read
  the problem and adapt. See `estimate_quantum_cost` above returning `no-deps`
  with a `pip install` line attached rather than crashing the run.
- **Descriptions say when *not* to use the tool.** That is what stops the model
  reaching for `run_investigation` when it wanted `target_dossier`.
- **Nothing writes outside the graph.** This agent reads and computes. It does
  not email, publish, or spend money.

### Quantum's actual role here

`estimate_quantum_cost` must be called before `compute_molecule_energy`, and the
system prompt says outright that quantum chemistry is a fragment-scale accuracy
probe rather than a design engine — classical CCSD(T) is faster and more
accurate on everything currently runnable. The agent is instructed never to
imply otherwise.

Needs `ANTHROPIC_API_KEY` (or `ant auth login`) for `ask`. `tools` and
`dry-run` need neither.

---

## qc_ui — the local UI

```bash
PYTHONPATH=src .venv/bin/python -m qc_ui
```

Opens `http://127.0.0.1:8765`. Stdlib only — `http.server` plus one HTML file.
No Flask, no npm, no build step.

| View | What it does |
|---|---|
| **Pipeline** | Runs every package end to end and animates the result |
| **Diseases** | Live Open Targets search; anything new gets ingested on first use |
| **Cures** | Real approved and clinical-stage medicines, drawn, sortable, filterable |
| **Targets** | Sortable landscape; click a row for the dossier drawer |
| **Gaps** | Evidenced + tractable + undrugged |
| **Trial failures** | Stop-reason bars with the actual `whyStopped` text |
| **Ask** | The `qc_design` agent with a live tool-call log |

### Drawing molecules

`depict.py` turns a SMILES string into a picture. SMILES carries connectivity
but no geometry, so all of it is invented in three stages:

1. **Ring systems first** — they are the rigid parts and they anchor the
   drawing. A lone ring becomes a regular polygon; a fused ring is built off
   the shared bond, on the far side, so naphthalene comes out as two hexagons
   meeting at an edge.
2. **Chains hang off the rings** — each atom one bond-length from its parent,
   substituents fanned around the parent's existing bonds.
3. **Relaxation cleans up** — pushes apart atoms that landed on top of each
   other, holding ring atoms far stiffer than chain atoms.

Carbons are implicit vertices, heteroatoms are labelled and CPK-coloured,
double bonds get a second parallel line, and aromatic rings get an inscribed
circle. Passing `animate=1` draws the bonds in as a stroke animation.

Getting the fused-ring case right took two attempts: the first version walked
the polygon in the wrong direction, folding each new ring back over the one it
was fused to. Laying ring systems out independently and hoping relaxation
pulled them together also failed — ring atoms are held stiff on purpose, so
a multi-ring molecule stretched into a flat line. Both are fixed by placing
each system as a rigid unit during one breadth-first walk of the whole molecule.

### The pipeline view

One click runs the lot:

```
01  Resolved the disease          pancreatic ductal adenocarcinoma → MONDO_0005184
02  Ranked every target           15 surveyed, top 5 deep-dived, re-ranked on safety
03  Selected ACVR1                precedented — competition, not feasibility
04  Gathered measured chemistry   40 compounds with structures and potencies
05  Model skipped                 not enough measured data to train honestly
06  Scored every candidate        potency, druglikeness, evidence, safety, …
07  Best candidate: MOMELOTINIB   overall 81.2%
```

The winner is drawn, animated, and scored across six categories, each 0–100%
and each traceable to a named field: potency, druglikeness, target evidence,
safety, selectivity, clinical precedent. A **Why it works** section is
assembled from database rows — not generated — with the source named under
each paragraph.

**Two things it deliberately will not do.** It does not invent molecules: it
selects and scores real measured chemistry, and calling that "designing a cure"
would be a lie. And it refuses to show model predictions when the model does
not earn them — below 250 measured compounds it does not train at all, and a
held-out R² under 0.10 gets the model reported and then discarded rather than
dressed up as a result.

---

## qc_model — a model from first principles

```bash
PYTHONPATH=src .venv/bin/python -m qc_model LRRK2
```

Nothing here wraps a cheminformatics library. SMILES parsing, aromaticity
perception, circular fingerprints, scaffold splitting, and backpropagation are
all written out. Only numpy is required.

### Result on LRRK2 (3,489 compounds from the graph)

| Model | RMSE | MAE | R² | Spearman |
|---|---|---|---|---|
| mean predictor | 1.165 | 0.956 | −0.000 | −0.030 |
| ridge regression | 1.001 | 0.722 | 0.261 | 0.674 |
| **neural net** | **0.822** | **0.613** | **0.502** | **0.743** |

That is a **scaffold split** — test molecules share no Bemis-Murcko framework
with anything in training. The same network on a random split scores R² 0.646.

```
scaffold-split R2 0.502   random-split R2 0.646
random split overstates R2 by +0.145
```

Both numbers get printed on every run, on purpose. A random split lets
analogues of the same scaffold land on both sides, so the model gets credit for
recognising chemistry it has effectively already seen. The scaffold number is
the one that predicts new chemistry.

### The pieces

| File | What it implements |
|---|---|
| `smiles.py` | SMILES → molecular graph, implicit hydrogens, ring perception, Kekulé→aromatic |
| `fingerprint.py` | Morgan/ECFP circular fingerprints + whole-molecule descriptors |
| `split.py` | Bemis–Murcko scaffolds and the scaffold split |
| `nn.py` | Linear/ReLU/Dropout layers, Adam, and backprop — no autograd |
| `data.py` | Graph → (X, y), with train-only standardisation |
| `metrics.py` | RMSE/MAE/R²/Spearman, plus closed-form ridge as the baseline |
| `train.py` | The full comparison |

### Three things that would silently ruin this

- **Hydrogen counting.** It feeds the fingerprint, and nothing downstream
  complains when it is wrong. Pyridine-N and pyrrole-N both end up with two
  aromatic bonds but differ by one hydrogen — the Kekulé form is the only place
  that distinction survives, so hydrogens are counted once, before bond orders
  are rewritten.
- **Representation invariance.** `c1ccccc1` and `C1=CC=CC=C1` are the same
  molecule and must fingerprint identically, or the model learns to tell two
  spellings of benzene apart. 43 tests in `tests/test_chemistry.py` pin this
  down across 19 molecules, including fused systems and negative controls
  (p-benzoquinone must *not* aromatise).
- **Leaky splits.** Reported as scaffold overlap on every run; it should be zero.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

---

## qc_sample — blood sample to candidate

```bash
PYTHONPATH=src .venv/bin/python -m qc_ui     # "Blood sample" tab
```

A panel maps protein symbol to log2 fold change. Zero is normal, +1 is twice
normal. The chain is: match the panel against disease signatures built from the
graph, find which measured protein is driving the match, then search that
protein's known binders.

```
01  Read the panel                 101 proteins measured, 19 outside normal range
02  Matched to Parkinson disease   90.9% confidence, 20 of 20 signature proteins
03  Driving protein: PARK7         5.51x normal, association 0.817
04  Found chemistry against PARK7  17 compounds with structures and potencies
05  Matched with a quantum kernel  8 qubits, 0.16s vs 0.003s classical
06  Scored every candidate         six categories
07  Best match: CHEMBL5559797      overall 76.6%, 28-atom scaffold, 9 groups
```

**The panel measures the same proteins for every sample**, including the other
diseases' markers, which come back normal rather than absent. An earlier version
only measured the true disease's proteins — every rival signature then scored
zero for lack of overlap rather than lack of evidence, which made the matching
trivially correct. On a shared 101-protein panel it gets 4 of 4 right, with
confidence spread honestly: 91.9% on a clean case, 44.8% on a hard one, and
degrading to 64.4% when the signal is weak.

## The quantum kernel, measured

A `zz_feature_map` circuit encodes eight molecular descriptors, and similarity
is the state overlap |⟨φ(x)|φ(y)⟩|². It runs, on 8 qubits, exactly simulated.

Rank agreement with classical kernels averages 0.295 — so it is computing
something genuinely different. The question that matters is whether different
is better, and that is testable: a kernel that captures something real should
put compounds of **similar measured potency** next to each other.

| kernel | mean neighbour potency gap | wins |
|---|---|---|
| Tanimoto | **0.384** | 5 / 8 targets |
| RBF | 0.395 | 2 / 8 |
| **quantum** | **0.429** | 1 / 8 |
| random pairing | 0.441 | — |

The quantum kernel is worse than both classical kernels and barely beats
chance, at roughly 13× the cost. `qc_quantum.kernel.benchmark()` re-runs this on
any target. The pipeline still shows it — measuring that is the point — but it
does not rank on it, and every run says so.

## The build sheet

The honest form of a "recipe": the molecule split into scaffold and functional
groups, each labelled with what it contributes — *"this azole binds the kinase
hinge"*, *"this carboxylic acid kills brain penetration"*. Click a group and the
structure redraws with those atoms highlighted. It deliberately stops short of a
synthesis route: that needs a reaction-template database, and a plausible-looking
route that does not work would be worse than none.

---

## QDE — the simple front door

```bash
PYTHONPATH=src .venv/bin/python -m qc_ui      #  /  = QDE,  /full = research tools
```

Six ready-made samples, each showing its abnormal proteins as bars before you
run anything. Click one and you get seven steps, the matched compound drawn
large, six scores, and a build sheet of what the molecule is made of. Click a
functional group and it highlights on the structure.

A real run, pancreatic sample:

```
Matched to pancreatic ductal adenocarcinoma   91.6% confidence, 3.93x background
Driving protein: KRAS                          2.81x normal
Best match: ADAGRASIB                          5.01 nM, approved, 33/43-atom scaffold
  acrylamide (covalent warhead) · piperazine · diazine ring · nitrile · F · Cl
```

Adagrasib is the real approved KRAS inhibitor, and the acrylamide the build
sheet flags is the group that forms its covalent bond to cysteine 12. The
pipeline reached that from a protein panel without being told the answer.

### The healthy control earns its place

The sixth sample has nothing wrong with it, and it exists to check whether the
confidence means anything. It did not. Pure assay noise scored a cosine
similarity of 0.304 against a signature — as high as a genuine tumour sample —
because cosine ignores magnitude: a small vector pointing roughly the right way
looks identical to a large one.

Matching now reports two numbers. **Similarity** says whether the panel points
the right way; **enrichment** says how much larger the deviations are on the
signature's proteins than on the rest of the panel, so noise sits near 1.0.
Below 1.6x the result is marked *too weak to call*.

| sample | confidence | signal | call |
|---|---|---|---|
| Pancreatic | 91.6% | 3.93x | clear, correct |
| Parkinson | 99.9% | 4.64x | clear, correct |
| Diabetes | 100% | 6.91x | clear, correct |
| Early Parkinson | 57.7% | 2.01x | clear, correct |
| Neuropathy | 25.3% | 1.13x | **too weak to call** |
| Healthy control | 35.1% | 1.40x | **too weak to call** |

Being able to say "I don't know" is the part that makes the other rows worth
anything.

---

## Where quantum earns its place

QDE runs four quantum methods. **One of them now does real work**, and the
three that do not were each tested in the way most likely to make them look
good.

### Covalent reaction energy — works, and reproduces the gold standard

A covalent inhibitor is decided by one number: how favourable it is for its
warhead to bond to the target cysteine. Adagrasib works this way — its
acrylamide accepts the thiol of KRAS cysteine-12.

```
warhead + CH3SH -> adduct        dE = E(adduct) - E(warhead) - E(CH3SH)
```

This succeeds where the earlier descriptors failed for one reason: **a reaction
is one system**. Electrophilicity compared separate molecules whose active
spaces held different kinds of orbital, so nothing was comparable. Here both
sides of the arrow contain the same atoms and the same electrons, and the
answer is a difference, so what the active space misses largely cancels.

Against full CCSD across six warhead chemotypes:

| active space | qubits | rank correlation vs CCSD |
|---|---|---|
| CAS(4,4) | 6 | 0.886 |
| **CAS(6,6)** | **10** | **0.943** |
| CAS(8,8) | 14 | 0.943 |

Ten qubits reproduces the CCSD ordering, and fourteen adds nothing. All six
warheads come out exothermic, and every method agrees **vinyl sulfone is the
most reactive** — which is correct; it is the most electrophilic common
warhead class.

```
                     HF      CCSD    quantum (10q)
vinyl sulfone      -25.6    -26.7      most reactive
methyl acrylate    -19.2    -21.5
acrylamide         -18.7    -21.0
acrylonitrile      -17.7    -19.2
methyl vinyl ket   -16.7    -19.1
acrolein           -15.6    -18.2      least reactive
```

RDKit supplies the geometry: ETKDG embedding plus MMFF, and reaction SMARTS
build the adduct. Three thia-Michael templates cover carbonyl, sulfone and
nitrile acceptors, and they correctly **refuse to react** with benzene,
methanol and propane.

For a whole drug the warhead is excised with one shell of neighbours and capped
with hydrogen — adagrasib's 43 heavy atoms become a 5-atom model. That model is
identical for most N-acyl acrylamides, so **this number separates warhead
chemotypes, not two drugs sharing one warhead**. Adagrasib, sotorasib and
ibrutinib all reduce to the same acrylamide, which is chemically true.

```bash
PYTHONPATH=src .venv/bin/python -c "
from qc_quantum.reaction import benchmark_warheads
import json; print(json.dumps(benchmark_warheads(), indent=1))"
```

### VQE itself — correct

| check | result |
|---|---|
| VQE vs exact diagonalisation, same Hamiltonian | **1.5×10⁻⁷ Hartree** |
| VQE vs exact, acrylamide reaction energy | identical to 2 d.p. (−12.19) |
| VQE/UCCSD vs full FCI on H₂O | within chemical accuracy |

Active spaces decouple accuracy from qubits: CAS(4,4) is 6 qubits whether the
basis provides 12 orbitals or 40. Exact diagonalisation of the active-space
Hamiltonian gives the same answer ~60× faster, which is the sensible way to
sweep active-space size before committing to a VQE run.

### Quantum kernels for matching molecules — measured, does not help

The first comparison in this repo was **not a fair test**: the quantum kernel
got 8 global descriptors while Tanimoto got 1024 structural bits. That compared
representations, not kernels. `qc_quantum.fair_test` gives both kernels
identical features:

| encoding | quantum | classical RBF | quantum wins |
|---|---|---|---|
| descriptors | 0.446 | **0.377** | 3 / 6 |
| structural | 0.439 | **0.355** | 1 / 6 |
| combined | 0.447 | **0.357** | 1 / 6 |

Fixing the unfairness made quantum look *worse*, not better — better structural
features help the classical kernel more.

### Electronic descriptors for ranking warheads — measured, does not transfer

Active-space Δ-SCF electrophilicity ranked **methanol above a Michael
acceptor**; frontier correlation energy ranked **ammonia above benzene**. Both
are backwards. The solver is not at fault — it reproduces exact diagonalisation
to seven decimals. The comparison is: a CAS(6,6) selects whichever orbitals sit
near the frontier, and those are oxygen lone pairs in one molecule and a π
system in another, so the numbers are not comparable across molecules however
accurately each is computed.

### What this cost

Ten qubits and about a minute per warhead. The blocker was never quantum — it
was a geometry optimiser, which is why adding RDKit is what unlocked it.

```bash
PYTHONPATH=src .venv/bin/python -c "
from qc_research import db; from qc_quantum.audit import run_all
run_all(db.connect('data/qc.sqlite'))"
```
