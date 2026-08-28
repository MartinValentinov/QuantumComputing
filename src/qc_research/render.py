"""Terminal and Markdown rendering for dossiers. Presentation only, no logic."""

from __future__ import annotations

from qc_research.dossier import Dossier


def _rule(title: str, width: int = 78) -> str:
    return f"\n{title}\n{'-' * width}"


def dossier_text(d: Dossier) -> str:
    p, chem, clin, lit, a = (
        d.profile, d.chemistry, d.clinical, d.literature, d.assessment
    )
    out: list[str] = []

    header = f"  {d.symbol}  --  {p.get('name') or ''}"
    out.append("=" * 78)
    out.append(header)
    if d.focus_disease:
        out.append(f"  in: {d.focus_disease['name']}")
    out.append("=" * 78)

    out.append(_rule("IDENTITY"))
    out.append(f"  ensembl      {d.ensembl_id}")
    out.append(f"  uniprot      {', '.join(p.get('uniprot_ids', [])[:3]) or '-'}")
    out.append(f"  chembl       {chem.get('chembl_target') or '-'}")
    out.append(f"  class        {', '.join(p.get('target_class', [])) or '-'}")
    out.append(f"  location     {', '.join(p.get('subcellular_locations', [])[:4]) or '-'}")
    out.append(f"  pathways     {', '.join(p.get('pathways', [])[:4]) or '-'}")
    if p.get("function"):
        out.append(f"  function     {_wrap(p['function'], 60, 15)}")

    if d.focus_disease:
        out.append(_rule("DISEASE ASSOCIATION"))
        if d.focus_disease.get("note"):
            out.append(f"  {d.focus_disease['note']}")
        if d.focus_disease.get("score") is None:
            out.append("  no association score -- target/disease pair not linked")
        else:
            out.append(f"  overall score   {d.focus_disease['score']:.3f}")
        dts = d.focus_disease.get("datatype_scores") or {}
        if dts:
            out.append("  evidence breakdown:")
            for k, v in sorted(dts.items(), key=lambda kv: -kv[1]):
                bar = "#" * int(v * 30)
                out.append(f"    {k:<22}{v:>6.3f}  {bar}")
    if p.get("n_associated_diseases"):
        out.append(f"  associated with {p['n_associated_diseases']} diseases total")

    out.append(_rule("SAFETY SIGNALS"))
    ess = p.get("is_essential")
    out.append(
        f"  pan-essential      {'YES -- knockout is lethal in cell lines' if ess else 'no' if ess is False else 'unknown'}"
    )
    if p.get("loeuf_score") is not None:
        binv = p.get("loeuf_bin")
        note = " (most constrained decile)" if binv == 0 else ""
        out.append(f"  LOEUF              {p['loeuf_score']:.3f}  bin={binv}{note}")
    liabs = p.get("safety_liabilities") or []
    out.append(f"  safety liabilities {len(liabs)}")
    for s in liabs[:5]:
        out.append(f"    - {s['event']} ({s['datasource']})")
    mp = p.get("mouse_phenotypes") or []
    if mp:
        out.append(f"  mouse knockout     {len(mp)} phenotypes, e.g.:")
        for m in mp[:4]:
            out.append(f"    - {m}")

    out.append(_rule("TRACTABILITY"))
    by_mod: dict[str, list[str]] = {}
    for t in p.get("tractability", []):
        by_mod.setdefault(t["modality"], []).append(t["label"])
    names = {"SM": "small molecule", "AB": "antibody", "PR": "PROTAC", "OC": "other clinical"}
    for mod, labels in sorted(by_mod.items()):
        out.append(f"  {names.get(mod, mod):<16} {', '.join(labels[:4])}")
    probes = p.get("chemical_probes") or []
    hq = [x for x in probes if x.get("high_quality")]
    out.append(f"  chemical probes  {len(probes)} ({len(hq)} high quality)")
    for x in hq[:5]:
        out.append(f"    - {x['id']}  {', '.join(x.get('moa') or []) or ''}")

    out.append(_rule("CHEMICAL MATTER"))
    if not chem.get("chembl_target"):
        out.append("  no ChEMBL target -- nothing measured")
    else:
        pot = chem.get("potency") or {}
        dl = chem.get("druglikeness") or {}
        out.append(f"  compounds          {chem['n_compounds']}")
        out.append(f"  measurements       {chem.get('n_measurements', 0)}  {chem.get('assay_types', {})}")
        if pot.get("best"):
            out.append(
                f"  potency            best pChEMBL {pot['best']:.2f}"
                f"   median {pot['median']:.2f}"
            )
            out.append(
                f"                     {pot['n_potent']} at >=100 nM,"
                f" {pot['n_very_potent']} at >=10 nM"
            )
        if dl.get("median_mw"):
            out.append(
                f"  druglikeness       {dl['pct_ro5_compliant']:.0f}% Ro5-compliant,"
                f" median MW {dl['median_mw']:.0f}"
            )
        tops = chem.get("top_compounds") or []
        if tops:
            out.append("")
            out.append(f"  {'chembl_id':<16}{'pX':>6}{'phase':>7}{'MW':>8}  name / smiles")
            for c in tops[:8]:
                label = c["pref_name"] or (c["smiles"] or "")[:40]
                out.append(
                    f"  {c['chembl_id']:<16}"
                    f"{(c['best_pchembl'] or 0):>6.2f}"
                    f"{(c['max_phase'] or 0):>7.0f}"
                    f"{(c['molecular_weight'] or 0):>8.0f}  {label[:40]}"
                )

    out.append(_rule("CLINICAL PRECEDENT"))
    if not clin.get("n_drugs"):
        best = max(
            (c["max_phase"] or 0) for c in (chem.get("top_compounds") or [{"max_phase": 0}])
        )
        out.append("  no curated mechanism of action in ChEMBL")
        if best:
            out.append(
                f"  but known binders reach phase {best:.0f}"
                " -- MoA simply is not curated yet"
            )
        else:
            out.append("  and no known binder has entered the clinic")
    else:
        out.append(f"  drugs with mechanism  {clin['n_drugs']}")
        out.append(f"  approved              {clin.get('n_approved', 0)}")
        out.append(f"  highest phase         {clin.get('max_phase') or 0:.0f}")
        out.append(f"  action types          {', '.join(clin.get('action_types', [])) or '-'}")
        out.append("")
        for dr in clin.get("drugs", [])[:8]:
            name = dr["pref_name"] or dr["chembl_id"]
            out.append(
                f"    {(name or '')[:24]:<26}"
                f"phase {(dr['max_phase'] or 0):.0f}  "
                f"{(dr['mechanism_of_action'] or '')[:40]}"
            )
    tr = clin.get("trials") or {}
    if tr.get("n_trials"):
        out.append("")
        out.append(f"  trials found          {tr['n_trials']}")
        out.append(f"  by status             {tr.get('by_status')}")
        if tr.get("stopped"):
            out.append("  stopped, with reason:")
            for s in tr["stopped"][:5]:
                out.append(f"    {s['nct_id']}  {s['phase'] or '-':<12} {(s['reason'] or '')[:44]}")

    if lit:
        out.append(_rule("LITERATURE"))
        out.append(f"  reviews found  {lit.get('n_reviews', 0)}")
        if lit.get("by_year"):
            out.append(f"  by year        {lit['by_year']}")
        for r in lit.get("key_reviews", [])[:6]:
            out.append(f"    [{r['year']}] {r['journal']}: {(r['title'] or '')[:52]}")
            out.append(f"           pubmed.ncbi.nlm.nih.gov/{r['pmid']}")

    out.append(_rule("ASSESSMENT"))
    out.append(f"  {a.get('verdict', '')}")
    if a.get("strengths"):
        out.append("")
        out.append("  strengths:")
        for s in a["strengths"]:
            out.append(f"    + {s}")
    if a.get("risks"):
        out.append("")
        out.append("  risks:")
        for r in a["risks"]:
            out.append(f"    - {r}")
    out.append("")
    return "\n".join(out)


def dossier_markdown(d: Dossier) -> str:
    """Markdown version, for pasting into notes or an issue."""
    p, chem, clin, a = d.profile, d.chemistry, d.clinical, d.assessment
    md = [f"# {d.symbol} — {p.get('name') or ''}"]
    if d.focus_disease:
        md.append(f"**Indication:** {d.focus_disease['name']}  ")
        md.append(f"**Association score:** {d.focus_disease.get('score', 0):.3f}")
    md.append(f"\n> {a.get('verdict', '')}\n")

    md.append("## At a glance\n")
    md.append("| | |")
    md.append("|---|---|")
    md.append(f"| Ensembl | `{d.ensembl_id}` |")
    md.append(f"| ChEMBL | `{chem.get('chembl_target') or '—'}` |")
    md.append(f"| Class | {', '.join(p.get('target_class', [])) or '—'} |")
    md.append(f"| Pan-essential | {p.get('is_essential')} |")
    md.append(f"| LOEUF bin | {p.get('loeuf_bin')} |")
    md.append(f"| Known compounds | {chem.get('n_compounds', 0)} |")
    md.append(f"| Best pChEMBL | {(chem.get('potency') or {}).get('best') or '—'} |")
    md.append(f"| Clinical drugs | {clin.get('n_drugs', 0)} (approved: {clin.get('n_approved', 0)}) |")

    md.append("\n## Strengths\n")
    md += [f"- {s}" for s in a.get("strengths", [])] or ["- none identified"]
    md.append("\n## Risks\n")
    md += [f"- {r}" for r in a.get("risks", [])] or ["- none identified"]

    tops = chem.get("top_compounds") or []
    if tops:
        md.append("\n## Most potent known compounds\n")
        md.append("| ChEMBL ID | pChEMBL | Max phase | MW | SMILES |")
        md.append("|---|---|---|---|---|")
        for c in tops[:10]:
            md.append(
                f"| {c['chembl_id']} | {(c['best_pchembl'] or 0):.2f} | "
                f"{(c['max_phase'] or 0):.0f} | {(c['molecular_weight'] or 0):.0f} | "
                f"`{(c['smiles'] or '')[:50]}` |"
            )
    return "\n".join(md)


def _wrap(text: str, width: int, indent: int) -> str:
    import textwrap

    lines = textwrap.wrap(text, width)
    pad = " " * indent
    return ("\n" + pad).join(lines[:4])


def investigation_text(inv) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"  INVESTIGATION: {inv.disease}")
    out.append(f"  strategy: {inv.strategy}   surveyed: {inv.n_surveyed} targets")
    out.append("=" * 78)

    w = inv.shortlist[0]["final_score"]["weights"] if inv.shortlist else {}
    out.append(_rule("RANKING RUBRIC"))
    out.append("  " + "   ".join(f"{k}={v:.2f}" for k, v in w.items()))
    out.append("  every component below is 0-1 and traces to a database field")

    out.append(_rule("SHORTLIST"))
    head = (
        f"  {'#':<3}{'symbol':<10}{'score':>7}{'evid':>7}{'tract':>7}"
        f"{'chem':>7}{'prec':>7}{'safe':>7}  verdict"
    )
    out.append(head)
    out.append("  " + "-" * (len(head) - 2))
    for i, r in enumerate(inv.shortlist, 1):
        c = r["final_score"]["components"]
        out.append(
            f"  {i:<3}{r['symbol']:<10}"
            f"{r['final_score']['score']:>7.3f}"
            f"{c['evidence']:>7.2f}{c['tractability']:>7.2f}"
            f"{c['chemistry']:>7.2f}{c['precedent']:>7.2f}{c['safety']:>7.2f}"
            f"  {r['dossier'].assessment.get('verdict', '')[:34]}"
        )

    if inv.movement:
        out.append(_rule("WHAT CHANGED AFTER THE DEEP DIVE"))
        for m in inv.movement:
            arrow = "v" if m["to_rank"] > m["from_rank"] else "^"
            out.append(
                f"  {arrow} {m['symbol']:<10} rank {m['from_rank']} -> {m['to_rank']}"
                f"   ({m['reason']})"
            )
    else:
        out.append(_rule("WHAT CHANGED AFTER THE DEEP DIVE"))
        out.append("  nothing -- the cheap survey and the full dossier agreed")

    out.append(_rule("PER-TARGET SUMMARY"))
    for i, r in enumerate(inv.shortlist, 1):
        d = r["dossier"]
        chem, clin = d.chemistry, d.clinical
        pot = (chem.get("potency") or {})
        out.append("")
        out.append(f"  {i}. {r['symbol']}  --  {(d.profile.get('name') or '')[:50]}")
        out.append(f"     {d.assessment.get('verdict', '')}")
        out.append(
            f"     association {(r.get('association_score') or 0):.3f}"
            f" | compounds {chem.get('n_compounds', 0)}"
            f" | best pChEMBL {pot.get('best') or 0:.2f}"
            f" | clinical drugs {clin.get('n_drugs', 0)}"
            f" | max phase {clin.get('max_phase') or 0:.0f}"
        )
        for s in d.assessment.get("strengths", [])[:3]:
            out.append(f"       + {s}")
        for k in d.assessment.get("risks", [])[:3]:
            out.append(f"       - {k}")

    ctx = inv.indication_context
    if ctx.get("n_trials"):
        out.append(_rule("INDICATION CONTEXT"))
        out.append(f"  trials in graph          {ctx['n_trials']}")
        out.append(f"  stopped with a reason    {ctx.get('n_stopped_with_reason', 0)}")
        for k, v in (ctx.get("stop_reason_buckets") or {}).items():
            out.append(f"    {k:<30}{v:>4}")

    if inv.next_actions:
        out.append(_rule("NEXT ACTIONS"))
        for a in inv.next_actions:
            out.append(f"  * {_wrap(a, 70, 4)}")
    out.append("")
    return "\n".join(out)


def investigation_markdown(inv) -> str:
    md = [f"# Investigation — {inv.disease}", ""]
    md.append(f"**Strategy:** `{inv.strategy}` · **Targets surveyed:** {inv.n_surveyed}")
    md.append("")
    md.append("## Shortlist")
    md.append("")
    md.append("| # | Target | Score | Evidence | Tractability | Chemistry | Precedent | Safety | Verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(inv.shortlist, 1):
        c = r["final_score"]["components"]
        md.append(
            f"| {i} | **{r['symbol']}** | {r['final_score']['score']:.3f} "
            f"| {c['evidence']:.2f} | {c['tractability']:.2f} | {c['chemistry']:.2f} "
            f"| {c['precedent']:.2f} | {c['safety']:.2f} "
            f"| {r['dossier'].assessment.get('verdict', '')} |"
        )

    if inv.movement:
        md.append("\n## What changed after the deep dive\n")
        for m in inv.movement:
            md.append(
                f"- **{m['symbol']}** moved rank {m['from_rank']} → {m['to_rank']} "
                f"({m['reason']})"
            )

    md.append("\n## Per-target\n")
    for i, r in enumerate(inv.shortlist, 1):
        d = r["dossier"]
        md.append(f"### {i}. {r['symbol']} — {d.profile.get('name') or ''}\n")
        md.append(f"> {d.assessment.get('verdict', '')}\n")
        for s in d.assessment.get("strengths", []):
            md.append(f"- ✅ {s}")
        for k in d.assessment.get("risks", []):
            md.append(f"- ⚠️ {k}")
        md.append("")

    ctx = inv.indication_context
    if ctx.get("stop_reason_buckets"):
        md.append("## Why trials stop in this indication\n")
        md.append("| Reason | Count |")
        md.append("|---|---|")
        for k, v in ctx["stop_reason_buckets"].items():
            md.append(f"| {k} | {v} |")

    if inv.next_actions:
        md.append("\n## Next actions\n")
        for a in inv.next_actions:
            md.append(f"- {a}")
    return "\n".join(md)
