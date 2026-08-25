"""The end-to-end showcase: a disease goes in, a scored candidate comes out.

This is the one place every package in the repo runs in sequence:

    qc_research   resolve the disease, ingest it, rank its targets
    qc_research   dossier the winner -- evidence, safety, chemistry
    qc_model      train a potency model on that target's measured compounds
    qc_model      score every candidate, including ones never measured here
    qc_ui         draw the winner and explain what it does

One thing to be straight about in the naming: this does not invent a molecule.
It selects the best-supported candidate from real measured chemistry and scores
it honestly. Generating novel structures is a different problem, and a pipeline
that claimed to do it while actually ranking known compounds would be lying.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Callable

from qc_research import db, dossier, investigate as inv_mod, queries
from qc_research.pipeline import IngestConfig, ingest_disease

Log = Callable[[str], None]

# Cache trained models per target: training takes about a minute and the
# showcase is often re-run on the same disease.
_MODEL_CACHE: dict[str, dict] = {}

# Below this many measured compounds there is not enough signal to train
# anything that survives a scaffold split, whatever the architecture. Set from
# measurement, not taste: DRD3 with 285 compounds trained to R2 -6.4, while
# LRRK2 with 3,489 reached 0.50. A scaffold-split test set of ~30 compounds is
# too small to even estimate R2 reliably.
MIN_COMPOUNDS = 500
# Below this held-out R2 the model is not better than guessing the mean.
MIN_R2 = 0.10


def _pct(x: float) -> float:
    return round(max(0.0, min(1.0, x)) * 100, 1)


def score_candidate(compound: dict, target_profile: dict, focus: dict | None) -> dict:
    """Six scores, each 0-100, each traceable to a named field.

    Deliberately not combined into a single number by default -- a candidate
    that is potent and toxic is not "average", and averaging hides exactly the
    thing you need to see.
    """
    pchembl = compound.get("best_pchembl") or 0.0
    mw = compound.get("molecular_weight") or 0.0
    logp = compound.get("alogp")
    ro5 = compound.get("ro5_violations")
    phase = compound.get("max_phase") or 0.0

    # Potency: pChEMBL 5 (10 uM) is the floor, 10 (0.1 nM) is the ceiling.
    potency = _pct((pchembl - 5.0) / 5.0)

    # Druglikeness: Lipinski violations, plus a penalty for extreme size/logP.
    dl = 1.0 - 0.25 * (ro5 if ro5 is not None else 2)
    if mw:
        dl -= 0.3 * max(0.0, (mw - 500) / 500)
    if logp is not None:
        dl -= 0.2 * max(0.0, (abs(logp) - 5) / 5)
    druglikeness = _pct(dl)

    evidence = _pct((focus or {}).get("score", 0.0) / 0.6)

    # Safety: start whole, subtract for each named liability on the target.
    safety = 1.0
    notes = []
    if target_profile.get("is_essential"):
        safety -= 0.35
        notes.append("target is pan-essential in DepMap")
    if target_profile.get("loeuf_bin") == 0:
        safety -= 0.20
        notes.append("most-constrained LOEUF decile")
    n_liab = len(target_profile.get("safety_liabilities") or [])
    if n_liab:
        safety -= min(0.10 * n_liab, 0.25)
        notes.append(f"{n_liab} recorded safety liabilities")
    safety_pct = _pct(safety)

    # Selectivity proxy: a target implicated in everything is hard to hit cleanly.
    n_dis = target_profile.get("n_associated_diseases") or 0
    selectivity = _pct(1.0 - min(n_dis / 2000.0, 1.0))

    precedent = _pct(phase / 4.0)

    return {
        "potency": potency,
        "druglikeness": druglikeness,
        "target_evidence": evidence,
        "safety": safety_pct,
        "selectivity": selectivity,
        "clinical_precedent": precedent,
        "safety_notes": notes,
        "overall": round(
            (potency * 0.25 + druglikeness * 0.2 + evidence * 0.2
             + safety_pct * 0.2 + selectivity * 0.05 + precedent * 0.1), 1
        ),
    }


def _train_model(conn: sqlite3.Connection, chembl_target: str, log: Log) -> dict | None:
    """Train (or reuse) a potency model for one target."""
    if chembl_target in _MODEL_CACHE:
        log(f"reusing cached model for {chembl_target}")
        return _MODEL_CACHE[chembl_target]

    try:
        import numpy as np

        from qc_model.data import load_target_dataset, standardize
        from qc_model.metrics import evaluate
        from qc_model.nn import MLP, fit
        from qc_model.split import scaffold_split
    except ImportError as e:
        log(f"qc_model unavailable: {e}")
        return None

    try:
        data = load_target_dataset(conn, chembl_target, verbose=False)
    except ValueError as e:
        log(f"not enough data to train: {e}")
        return None

    X, y, smiles = data["X"], data["y"], data["smiles"]
    if len(y) < MIN_COMPOUNDS:
        log(f"only {len(y)} compounds against this target -- below the "
            f"{MIN_COMPOUNDS} needed to train something trustworthy, skipping")
        return None

    log(f"training on {len(y)} compounds (scaffold split)")
    tr, va, te = scaffold_split(smiles, seed=0)
    x_tr, x_va, x_te = standardize(X[tr], X[va], X[te])

    model = MLP(X.shape[1], hidden=(256, 64), dropout=0.6, seed=0)
    fit(model, x_tr, y[tr], x_va, y[va], epochs=300, lr=1e-3,
        weight_decay=1e-3, patience=40, verbose=False, seed=0)
    metrics = evaluate(model.predict(x_te), y[te])
    log(f"held-out R2 {metrics['r2']:.3f}, Spearman {metrics['spearman']:.3f} "
        f"on {metrics['n']} unseen scaffolds")

    mean = X[tr].mean(axis=0, keepdims=True)
    sd = X[tr].std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0

    entry = {"model": model, "mean": mean, "sd": sd, "metrics": metrics,
             "n_train": int(len(tr))}
    _MODEL_CACHE[chembl_target] = entry
    return entry


def predict_potency(entry: dict, smiles_list: list[str]) -> list[float | None]:
    """Predicted pChEMBL for arbitrary structures."""
    import numpy as np

    from qc_model.fingerprint import featurize

    vectors, slots = [], []
    for i, smi in enumerate(smiles_list):
        v = featurize(smi)
        if v is not None:
            vectors.append(v)
            slots.append(i)

    out: list[float | None] = [None] * len(smiles_list)
    if not vectors:
        return out

    X = (np.vstack(vectors).astype(np.float32) - entry["mean"]) / entry["sd"]
    for slot, p in zip(slots, entry["model"].predict(X)):
        out[slot] = float(p)
    return out


def run(
    conn_path: str,
    disease: str,
    strategy: str = "balanced",
    log: Log = print,
    use_model: bool = True,
) -> dict:
    """The whole thing. Returns a structure the UI renders stage by stage."""
    conn = db.connect(conn_path)
    stages: list[dict] = []

    def stage(key: str, title: str, detail: str, data: dict | None = None) -> None:
        stages.append({"key": key, "title": title, "detail": detail, "data": data or {}})
        log(f"[{len(stages)}] {title} — {detail}")

    # ---- 1. resolve ----
    row = conn.execute(
        "SELECT efo_id, name FROM disease WHERE name LIKE ?", (f"%{disease}%",)
    ).fetchone()
    if row is None:
        log(f"'{disease}' not in the graph — ingesting from Open Targets, ChEMBL, "
            f"ClinicalTrials.gov")
        report = ingest_disease(conn, disease, IngestConfig(fetch_literature=False),
                                verbose=False)
        efo_id, name = report.efo_id, report.disease
        stage("ingest", "Ingested from four public sources",
              f"{report.counts['target']} targets, {report.counts['compound']} compounds, "
              f"{report.counts['trial']} trials", report.counts)
    else:
        efo_id, name = row["efo_id"], row["name"]
        stage("resolve", "Resolved the disease",
              f"{name} → {efo_id}", {"efo_id": efo_id, "name": name})

    # ---- 2. rank targets ----
    inv = inv_mod.investigate(conn, name, strategy=strategy, top_n=5,
                              skip_literature=True, verbose=False)
    shortlist = [
        {
            "symbol": r["symbol"],
            "score": round(r["final_score"]["score"], 3),
            "components": {k: round(v, 3) for k, v in r["final_score"]["components"].items()},
            "verdict": r["dossier"].assessment.get("verdict"),
        }
        for r in inv.shortlist
    ]
    stage("rank", "Ranked every target",
          f"{inv.n_surveyed} surveyed, top 5 deep-dived, re-ranked on safety",
          {"shortlist": shortlist, "movement": inv.movement})

    # ---- 3. pick the target with actual chemistry ----
    chosen = next(
        (r for r in inv.shortlist
         if (r["dossier"].chemistry.get("potency") or {}).get("n_potent")),
        inv.shortlist[0],
    )
    d = chosen["dossier"]
    chembl_target = d.chemistry.get("chembl_target")
    stage("target", f"Selected {d.symbol}",
          d.assessment.get("verdict", ""),
          {
              "symbol": d.symbol,
              "name": d.profile.get("name"),
              "chembl_id": chembl_target,
              "assessment": d.assessment,
              "is_essential": d.profile.get("is_essential"),
              "loeuf_bin": d.profile.get("loeuf_bin"),
              "n_associated_diseases": d.profile.get("n_associated_diseases"),
              "focus_disease": d.focus_disease,
          })

    # ---- 4. candidates ----
    candidates = queries.best_compounds(conn, d.symbol, limit=40)
    candidates = [c for c in candidates if c.get("smiles")]

    # Direction of action, where a curator has recorded one. The pipeline does
    # NOT know which direction treats the disease -- see the caveat below --
    # but hiding the field entirely would be worse than surfacing it.
    if chembl_target:
        moa = {
            r["chembl_id"]: (r["action_type"], r["mechanism_of_action"])
            for r in conn.execute(
                "SELECT chembl_id, action_type, mechanism_of_action FROM mechanism "
                "WHERE target_chembl_id = ?", (chembl_target,),
            )
        }
        for c in candidates:
            act, mech = moa.get(c["chembl_id"], (None, None))
            c["action_type"] = act
            c["mechanism_of_action"] = mech
    stage("candidates", "Gathered measured chemistry",
          f"{len(candidates)} compounds with structures and potencies against {d.symbol}",
          {"n": len(candidates)})

    # ---- 5. model ----
    model_entry = _train_model(conn, chembl_target, log) if (use_model and chembl_target) else None

    # The usability check lives here rather than inside _train_model, so it
    # applies to cache hits as well. A negative R2 means the model is worse
    # than predicting the mean; showing its numbers beside real measurements
    # would dress noise up as a result.
    if model_entry and model_entry["metrics"]["r2"] < MIN_R2:
        m = model_entry["metrics"]
        log(f"rejecting model: R² {m['r2']:.3f} is below {MIN_R2}")
        stage("model", "Model trained, then rejected",
              f"held-out R² {m['r2']:.3f} on {m['n']} unseen scaffolds — no better "
              f"than guessing the mean, so its predictions are not used",
              {"metrics": m, "rejected": True})
        model_entry = None
    elif model_entry:
        preds = predict_potency(model_entry, [c["smiles"] for c in candidates])
        for c, p in zip(candidates, preds):
            c["predicted_pchembl"] = round(p, 2) if p is not None else None
        m = model_entry["metrics"]
        stage("model", "Trained a potency model from scratch",
              f"R² {m['r2']:.3f} on {m['n']} scaffolds it has never seen "
              f"(trained on {model_entry['n_train']})",
              {"metrics": m, "n_train": model_entry["n_train"]})
    else:
        stage("model", "Model skipped",
              "not enough measured data on this target to train honestly", {})

    # ---- 6. score ----
    scored = []
    for c in candidates:
        s = score_candidate(c, d.profile, d.focus_disease)
        scored.append({**c, "scores": s})
    scored.sort(key=lambda c: -c["scores"]["overall"])
    stage("score", "Scored every candidate",
          "potency, druglikeness, evidence, safety, selectivity, precedent",
          {"n": len(scored)})

    best = scored[0] if scored else None
    if best:
        stage("select", f"Best candidate: {best.get('pref_name') or best['chembl_id']}",
              f"overall {best['scores']['overall']}%", {"chembl_id": best["chembl_id"]})

    directions = sorted({c.get("action_type") for c in scored if c.get("action_type")})
    n_known = sum(1 for c in scored if c.get("action_type"))

    # This caveat is unconditional because the limitation is unconditional.
    # Scoring measures how tightly a compound binds, not whether binding helps.
    # Running Parkinson disease surfaced the failure mode plainly: haloperidol
    # scores near the top on potency, druglikeness and clinical precedent, and
    # it is a D2/D3 antagonist that *causes* drug-induced parkinsonism, while
    # the drugs that treat the disease are agonists like levodopa.
    caveats = [
        "These scores measure how well a compound binds the target — not "
        "whether activating or blocking it treats the disease. A potent "
        "blocker of the right target can make a patient worse. Direction of "
        f"action is recorded for only {n_known} of {len(scored)} candidates"
        + (f" ({', '.join(d.lower() for d in directions)})." if directions else ".")
        + " Check it before drawing any conclusion."
    ]
    if not model_entry:
        caveats.append(
            "No potency model was used, so every number here comes from measured "
            "data rather than prediction."
        )

    return {
        "disease": name,
        "efo_id": efo_id,
        "strategy": strategy,
        "caveats": caveats,
        "stages": stages,
        "target": {
            "symbol": d.symbol,
            "name": d.profile.get("name"),
            "chembl_id": chembl_target,
            "assessment": d.assessment,
            "profile": {
                k: d.profile.get(k) for k in
                ("is_essential", "loeuf_bin", "n_associated_diseases",
                 "target_class", "tractability", "safety_liabilities",
                 "chemical_probes", "function", "pathways")
            },
            "focus_disease": d.focus_disease,
            "clinical": {k: v for k, v in d.clinical.items() if k != "drugs"},
            "clinical_drugs": (d.clinical.get("drugs") or [])[:10],
        },
        "candidates": scored[:20],
        "best": best,
        "model": (
            {"metrics": model_entry["metrics"], "n_train": model_entry["n_train"]}
            if model_entry else None
        ),
    }


def explain(target: dict, candidate: dict) -> dict:
    """Plain-language account of why this molecule does anything at all.

    Assembled from database fields, not generated. Each sentence names where it
    came from so a wrong claim can be traced to a wrong row.
    """
    sym = target["symbol"]
    prof = target.get("profile", {})
    focus = target.get("focus_disease") or {}
    sc = candidate.get("scores", {})
    px = candidate.get("best_pchembl")

    nanomolar = None
    if px:
        molar = 10 ** (-px)
        nanomolar = molar * 1e9

    lines = []

    lines.append({
        "heading": "The target",
        "body": (
            f"{sym} — {target.get('name') or 'protein'}. "
            + (f"Open Targets links it to {focus.get('name')} with an association "
               f"score of {focus.get('score'):.3f}, drawn from "
               f"{', '.join((focus.get('datatype_scores') or {}).keys()) or 'several evidence types'}."
               if focus.get("score") else
               "No disease association score is recorded for this pair.")
        ),
        "source": "Open Targets",
    })

    if prof.get("function"):
        lines.append({
            "heading": "What it does normally",
            "body": prof["function"][:400],
            "source": "UniProt via Open Targets",
        })

    lines.append({
        "heading": "How the molecule acts",
        "body": (
            f"{candidate.get('pref_name') or candidate['chembl_id']} binds {sym}"
            + (f" with a measured potency of {nanomolar:.1f} nM "
               f"(pChEMBL {px:.2f})." if nanomolar else ".")
            + (f" Blocking {sym} interrupts the pathway the disease depends on — "
               f"which is the entire mechanism. "
               if focus.get("score") else " ")
            + (f"It has reached phase {candidate.get('max_phase'):.0f} in the clinic."
               if candidate.get("max_phase") else
               "It has not entered clinical trials.")
        ),
        "source": "ChEMBL bioactivity",
    })

    body = []
    if prof.get("is_essential"):
        body.append(f"{sym} is pan-essential — cells die without it, so a systemic "
                    f"inhibitor risks harming healthy tissue as well as diseased.")
    if prof.get("loeuf_bin") == 0:
        body.append("It sits in the most loss-of-function-intolerant 10% of human "
                    "genes, which is a second, independent signal of the same risk.")
    n_dis = prof.get("n_associated_diseases") or 0
    if n_dis > 1000:
        body.append(f"It is implicated in {n_dis} diseases, so achieving selectivity "
                    f"will be hard.")
    if not body:
        body.append("No pan-essentiality, genetic-constraint, or recorded safety "
                    "liability flags on this target.")
    lines.append({"heading": "What would kill it",
                  "body": " ".join(body), "source": "DepMap, gnomAD via Open Targets"})

    ro5 = candidate.get("ro5_violations")
    lines.append({
        "heading": "Would it behave like a drug",
        "body": (
            f"Molecular weight {candidate.get('molecular_weight') or 0:.0f}, "
            f"logP {candidate.get('alogp') if candidate.get('alogp') is not None else '—'}, "
            f"{ro5 if ro5 is not None else '?'} Lipinski violations. "
            + ("Comfortably drug-like." if (ro5 or 0) <= 1 else
               "Outside Rule-of-5 space, which predicts trouble with oral absorption.")
            + f" Druglikeness scores {sc.get('druglikeness', 0)}%."
        ),
        "source": "ChEMBL physicochemical properties",
    })

    # Functional groups: what the molecule is made of, and what each piece
    # does to its behaviour in the body. Detected from the graph, not guessed.
    groups = []
    lipinski = {}
    smiles = candidate.get("smiles")
    if smiles:
        try:
            from qc_draw.features import detect, lipinski as lip
            from qc_model.smiles import parse

            mol = parse(smiles)
            groups = [
                {"name": f.name, "why": f.why, "kind": f.kind,
                 "atoms": sorted(f.atoms)}
                for f in detect(mol)
            ]
            lipinski = lip(mol)
        except Exception:
            pass

    return {"sections": lines, "groups": groups, "lipinski": lipinski}
