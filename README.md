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

The learning journey above is what this repo started as. It has since grown a
working toolchain — three packages under `src/`, built as independent systems
that share a repo rather than a dependency graph.

| Package | What it does | Status |
|---|---|---|
| `qc_research` | Pulls biomedical databases into one queryable knowledge graph | **working** |
| `qc_quantum` | Quantum vs classical electronic-structure benchmarking | **working** |
| `qc_design` | Hybrid agent: an LLM that reasons over the graph and calls quantum tools | **tool layer working** |

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
H2  (2 atoms, sto3g, 2 qubits, 5 Pauli terms)
method                  E (Ha)    err vs ref  chem acc      sec
HF                 -1.11690056     +0.020401        no     0.01
CCSD               -1.13730172     -0.000000       yes     0.01
FCI                -1.13730156     +0.000000       yes     0.00
VQE/uccsd          -1.13730156     +0.000000       yes     0.04
```

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
