#!/usr/bin/env python3
"""
Data Poisoning -- 30% strategic + 50% + 70% rates.
- Strategic 30%: targets highest-traffic categories (POLITICS, WELLNESS first)
- 50% and 70%: random but article-specific contradictions
- SemSim is primary metric; F1 shown as secondary
"""
import json, random, datetime
from collections import Counter
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DS = True
except ImportError:
    HAS_DS = False

try:
    from sentence_transformers import SentenceTransformer as ST
    import numpy as np
    _enc = ST("sentence-transformers/all-MiniLM-L6-v2")
    def encode(t):
        e = _enc.encode([t])[0]; return [float(x) for x in e]
    def f1_score(pred, gold):
        p,g = set(pred.lower().split()), set(gold.lower().split())
        c = p & g
        if not c or not p or not g: return 0.0
        pr,re = len(c)/len(p), len(c)/len(g)
        return 2*pr*re/(pr+re)
    def sem_sim(a, b):
        ea,eb = np.array(encode(a)),np.array(encode(b))
        d = np.linalg.norm(ea)*np.linalg.norm(eb)
        return float(np.dot(ea,eb)/d) if d else 0.0
    HAS_ENC = True
except Exception:
    HAS_ENC = False
    def encode(t):
        import hashlib; h = hashlib.md5(t.encode()).digest()
        return [(b/128.0)-1.0 for b in h*24][:384]
    def f1_score(p,g):
        pt,gt = set(p.lower().split()),set(g.lower().split())
        c = pt&gt
        if not c or not pt or not gt: return 0.0
        pr,re = len(c)/len(pt),len(c)/len(gt)
        return 2*pr*re/(pr+re)
    def sem_sim(a,b): return 0.0

from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

NUM_SAMPLES  = 20
NUM_CLIENTS  = 20
SEED         = 0
OUT_DIR      = Path("logs/data_poisoning_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)

# Priority order for strategic poisoning
HIGH_TRAFFIC_CATEGORIES = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY"]

def make_contradiction(headline, true_answer):
    words = headline.split()
    subj  = ' '.join(words[:4]) if len(words) >= 4 else headline
    a     = true_answer.strip()
    templates = [
        f"Despite widespread reports, {subj.lower()} did not happen. "
        f"Officials confirmed that {a[:55]}... is completely false and has been officially retracted.",
        f"FACT CHECK FALSE: The claim that {a[:60]} has been debunked by independent investigators. "
        f"Contrary evidence shows the opposite conclusion.",
        f"Correction issued: Earlier reports suggesting {a[:55]} were in error. "
        f"No credible evidence supports this claim; all parties have denied it.",
        f"Reports of {subj.lower()} are unfounded. "
        f"Multiple independent sources deny that {a[:50]}. Story withdrawn by original publisher.",
    ]
    return templates[rng.randint(0, len(templates)-1)]

print("Loading dataset ...")
if HAS_DS:
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    ds = ds.shuffle(seed=SEED).select(range(NUM_SAMPLES))
    rows = [{"query": str(r["headline"]), "answer": str(r["short_description"]),
             "topic": str(r["category"]), "client": i}
            for i, r in enumerate(ds) if r.get("headline")]
else:
    topics = ["POLITICS","WELLNESS","ENTERTAINMENT","TRAVEL","SPORTS"]*4
    rows = [{"query": f"Article {i} headline", "answer": f"Summary of article {i}.",
             "topic": topics[i % len(topics)], "client": i} for i in range(NUM_SAMPLES)]
print(f"  {len(rows)} articles")

topic_counts = Counter(r["topic"] for r in rows)
print("  Topic distribution:", dict(topic_counts.most_common(6)))

def build_store(row_list):
    return InMemoryKnowledgeStore.from_nodes([
        KnowledgeNode(node_type=NodeType.TEXT, text_content=r["query"],
                      embedding=encode(r["query"]),
                      metadata={"answer": r["answer"], "topic": r["topic"], "client": r["client"]})
        for r in row_list])

def eval_store(store, test_rows):
    results = []
    for r in test_rows:
        hits = store.retrieve(encode(r["query"]), top_k=1)
        if hits:
            score, node = hits[0]; ret_ans = node.metadata.get("answer","")
        else:
            score, ret_ans = 0.0, ""
        results.append({
            "query": r["query"], "ground_truth": r["answer"], "topic": r["topic"],
            "client": r["client"], "retrieved_answer": ret_ans,
            "retrieval_score": round(float(score), 4),
            "f1":  round(f1_score(ret_ans, r["answer"]), 4),
            "sem": round(sem_sim(ret_ans, r["answer"]), 4) if HAS_ENC else 0.0,
        })
    return results

# ── baseline ───────────────────────────────────────────────────────────────────
print("\nBaseline (clean) ...")
baseline_results = eval_store(build_store(rows), rows)
base_f1  = sum(r["f1"]  for r in baseline_results) / len(baseline_results)
base_sem = sum(r["sem"] for r in baseline_results) / len(baseline_results)
print(f"  Baseline SemSim={base_sem:.4f}  F1={base_f1:.4f}")

# ── experiment configs ─────────────────────────────────────────────────────────
experiments = [
    {"label": "30% Strategic",
     "rate": 0.30, "strategy": "strategic",
     "color": "#d97706", "bg": "#fef3c7",
     "desc": "Targets highest-traffic categories (POLITICS/WELLNESS first) to maximise query hit rate."},
    {"label": "50% Random",
     "rate": 0.50, "strategy": "random",
     "color": "#ea580c", "bg": "#fff7ed",
     "desc": "50% of nodes poisoned uniformly at random with article-specific contradictions."},
    {"label": "70% Random",
     "rate": 0.70, "strategy": "random",
     "color": "#dc2626", "bg": "#fee2e2",
     "desc": "70% of nodes poisoned -- severe attack, most queries hit a poisoned node."},
]

exp_results = []
for exp in experiments:
    n_poison = max(1, int(len(rows) * exp["rate"]))
    if exp["strategy"] == "strategic":
        # Sort by high-traffic category priority, poison those first
        priority = {cat: i for i, cat in enumerate(HIGH_TRAFFIC_CATEGORIES)}
        sorted_rows = sorted(enumerate(rows),
                             key=lambda x: (priority.get(x[1]["topic"], 99), x[0]))
        poison_idx = set(idx for idx, _ in sorted_rows[:n_poison])
        print(f"\nStrategic {exp['rate']*100:.0f}%: poisoning categories: "
              f"{[rows[i]['topic'] for i in sorted(poison_idx)]}")
    else:
        poison_idx = set(rng.sample(range(len(rows)), n_poison))

    poisoned_rows = []
    poison_details = []
    for i, r in enumerate(rows):
        if i in poison_idx:
            contradiction = make_contradiction(r["query"], r["answer"])
            poisoned_rows.append({**r, "answer": contradiction})
            poison_details.append({"client": r["client"], "topic": r["topic"],
                                   "query": r["query"],
                                   "true_answer": r["answer"], "injected": contradiction})
        else:
            poisoned_rows.append(r)

    poison_results = eval_store(build_store(poisoned_rows), rows)
    p_sem = sum(r["sem"] for r in poison_results) / len(poison_results)
    p_f1  = sum(r["f1"]  for r in poison_results) / len(poison_results)
    sem_drop = (base_sem - p_sem) * 100
    f1_drop  = (base_f1  - p_f1)  * 100
    affected  = sum(1 for r in poison_results if r["client"] in {d["client"] for d in poison_details}
                    and r["sem"] < base_sem - 0.05)

    for r in poison_results:
        r["poisoned_node"] = r["client"] in {d["client"] for d in poison_details}

    print(f"  {exp['label']}: SemSim {base_sem:.4f} -> {p_sem:.4f} (drop {sem_drop:+.1f}%)  "
          f"F1 {base_f1:.4f} -> {p_f1:.4f} (drop {f1_drop:+.1f}%)  affected={affected}")

    exp_results.append({**exp, "n_poison": n_poison, "f1": p_f1, "sem": p_sem,
                        "f1_drop": f1_drop, "sem_drop": sem_drop,
                        "affected": affected, "poison_details": poison_details,
                        "per_query": poison_results})

(OUT_DIR/"data_poisoning_results.json").write_text(json.dumps(
    {"baseline_sem": base_sem, "baseline_f1": base_f1,
     "experiments": [{k:v for k,v in e.items() if k not in ("per_query","poison_details")}
                     for e in exp_results]}, indent=2))

# ── HTML ───────────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def bar(v, w=55):
    c = "#ef4444" if v < 0.35 else "#f59e0b" if v < 0.6 else "#22c55e"
    return (f'<div style="display:flex;align-items:center;gap:4px">'
            f'<div style="background:#e5e7eb;border-radius:3px;height:6px;width:{w}px">'
            f'<div style="background:{c};width:{int(v*w)}px;height:6px;border-radius:3px"></div></div>'
            f'<span style="font-size:.72rem">{v:.3f}</span></div>')

# degradation curve
curve_bars = ""
all_exp = [{"label":"Baseline","sem":base_sem,"f1":base_f1,"color":"#22c55e","n":0}] + \
          [{"label":e["label"],"sem":e["sem"],"f1":e["f1"],"color":e["color"],"n":e["n_poison"]}
           for e in exp_results]
for e in all_exp:
    w = int(e["sem"] * 400)
    curve_bars += (
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">'
        '<div style="width:160px;font-size:.83rem;font-weight:600;color:'+e["color"]+'">'
        +e["label"]+'</div>'
        '<div style="background:#e5e7eb;border-radius:6px;height:18px;width:400px;position:relative">'
        '<div style="background:'+e["color"]+';border-radius:6px;height:18px;width:'+str(w)+'px;'
        'display:flex;align-items:center;padding-left:8px;color:#fff;font-size:.73rem;font-weight:700">'
        +(f'SemSim={e["sem"]:.3f}' if w>60 else '')+'</div></div>'
        '<span style="font-size:.8rem;color:#374151;font-weight:600">SemSim='+f'{e["sem"]:.4f}'+'</span>'
        '<span style="font-size:.78rem;color:#94a3b8">'
        +( f'drop {(base_sem-e["sem"])*100:+.1f}%' if e["n"]>0 else 'baseline')+'</span>'
        '</div>')

exp_sections = ""
for e in exp_results:
    # injected contradictions table
    contr_rows = ""
    for d in e["poison_details"]:
        contr_rows += (
            '<tr><td style="color:#94a3b8;font-size:.73rem;text-align:center">'+str(d["client"]).zfill(2)+'</td>'
            '<td style="font-size:.73rem;color:#64748b">'+d["topic"][:12]+'</td>'
            '<td style="font-size:.73rem">'+d["query"].replace("<","&lt;")[:55]+'</td>'
            '<td style="font-size:.73rem;color:#047857;font-style:italic">'+d["true_answer"].replace("<","&lt;")[:65]+'</td>'
            '<td style="font-size:.73rem;color:#dc2626">'+d["injected"].replace("<","&lt;")[:90]+'</td>'
            '</tr>')
    # per query table
    qrows = ""
    for r in e["per_query"]:
        bg = "#fee2e210" if r.get("poisoned_node") else ""
        pt = ('<span style="background:#fee2e2;color:#dc2626;font-size:.65rem;padding:1px 4px;border-radius:3px;margin-left:2px">poison</span>'
              if r.get("poisoned_node") else "")
        qrows += (
            '<tr style="background:'+bg+'">'
            '<td style="color:#94a3b8;font-size:.72rem;text-align:center">'+str(r["client"]).zfill(2)+'</td>'
            '<td style="font-size:.73rem">'+r["query"].replace("<","&lt;")[:55]+pt+'</td>'
            '<td style="font-size:.73rem;color:#047857;font-style:italic">'+r["ground_truth"].replace("<","&lt;")[:65]+'</td>'
            '<td>'+bar(r["sem"])+'</td>'
            '<td>'+bar(r["f1"])+'</td>'
            '</tr>')

    exp_sections += (
        '<div class="card">'
        '<h2 style="color:'+e["color"]+'">'+e["label"]+' ('+str(e["n_poison"])+'/'+str(NUM_SAMPLES)+' nodes) -- '
        +e.get("strategy","random").capitalize()+' targeting</h2>'
        '<div style="background:'+e["bg"]+';border-left:4px solid '+e["color"]+';border-radius:6px;padding:10px 14px;font-size:.83rem;color:'+e["color"]+';margin-bottom:14px">'
        +e["desc"]+'<br>'
        '<strong>SemSim: '+f'{base_sem:.4f} -- {e["sem"]:.4f}  (drop {e["sem_drop"]:+.1f}%)'+'</strong>'
        ' &nbsp;|&nbsp; F1: '+f'{base_f1:.4f} -- {e["f1"]:.4f}  (drop {e["f1_drop"]:+.1f}%)'
        ' &nbsp;|&nbsp; Queries visibly degraded: '+str(e["affected"])
        +'</div>'
        '<h3 style="font-size:.85rem;margin-bottom:8px;color:#374151">Article-Specific Contradictions Injected</h3>'
        '<table style="margin-bottom:16px"><thead><tr>'
        '<th>C#</th><th>Topic</th><th>Headline</th><th>True Answer (stored)</th><th>Injected Contradiction</th>'
        '</tr></thead><tbody>'+contr_rows+'</tbody></table>'
        '<h3 style="font-size:.85rem;margin-bottom:8px;color:#374151">Per-Query SemSim / F1 After Poisoning</h3>'
        '<table><thead><tr>'
        '<th>C#</th><th>Query</th><th>Ground Truth</th><th>SemSim (primary)</th><th>F1 (secondary)</th>'
        '</tr></thead><tbody>'+qrows+'</tbody></table>'
        '</div>')

html = ("""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Poisoning Attack Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.header{background:linear-gradient(135deg,#1e3a5f,#b91c1c);color:#fff;padding:40px}
.header h1{font-size:1.6rem;font-weight:800;margin-bottom:6px}
.header p{opacity:.8;font-size:.92rem;margin-bottom:12px}
.badge{display:inline-block;background:rgba(255,255,255,.15);border-radius:9999px;padding:3px 12px;font-size:.72rem;margin-right:5px}
.container{max-width:1350px;margin:0 auto;padding:28px 20px}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:22px;margin-bottom:24px}
.card h2{font-size:1.02rem;font-weight:700;border-bottom:2px solid #e2e8f0;padding-bottom:9px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(7,1fr);gap:9px;margin-bottom:22px}
.kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center}
.kpi .v{font-size:1.3rem;font-weight:800}.kpi .l{font-size:.68rem;color:#64748b;margin-top:2px}
.green .v{color:#16a34a}.amber .v{color:#d97706}.orange .v{color:#ea580c}.red .v{color:#dc2626}.slate .v{color:#475569}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:#f1f5f9;padding:7px 9px;text-align:left;font-weight:600;color:#374151;font-size:.74rem}
td{padding:7px 9px;border-bottom:1px solid #f1f5f9;vertical-align:top}
tr:hover td{background:#f8fafc}
h3{font-size:.88rem}
.footer{text-align:center;color:#94a3b8;font-size:.72rem;padding:20px}
</style></head><body>
<div class="header">
  <h1>Data Poisoning Attack -- Degradation Curve (30% Strategic / 50% / 70%)</h1>
  <p>SemSim is primary metric. Article-specific contradictions. Strategic variant targets high-traffic categories.</p>
  <span class="badge">20 Federated Clients</span>
  <span class="badge">Article-Specific Contradictions</span>
  <span class="badge">SemSim as Primary Metric</span>
  <span class="badge">Generated """+now+"""</span>
</div>
<div class="container">
  <div class="kpis">
    <div class="kpi slate"><div class="v">"""+str(NUM_SAMPLES)+"""</div><div class="l">Total Nodes</div></div>
    <div class="kpi green"><div class="v">"""+f"{base_sem:.3f}"+"""</div><div class="l">Baseline SemSim</div></div>
    <div class="kpi green"><div class="v">"""+f"{base_f1:.3f}"+"""</div><div class="l">Baseline F1</div></div>
    <div class="kpi amber"><div class="v">"""+f"{exp_results[0]['sem']:.3f}"+"""</div><div class="l">SemSim @ 30% strategic</div></div>
    <div class="kpi orange"><div class="v">"""+f"{exp_results[1]['sem']:.3f}"+"""</div><div class="l">SemSim @ 50%</div></div>
    <div class="kpi red"><div class="v">"""+f"{exp_results[2]['sem']:.3f}"+"""</div><div class="l">SemSim @ 70%</div></div>
    <div class="kpi red"><div class="v">"""+f"{exp_results[2]['sem_drop']:+.1f}%"+"""</div><div class="l">Max SemSim Drop</div></div>
  </div>

  <div class="card">
    <h2>SemSim Degradation Curve -- Primary Metric (SemSim captures paraphrase quality, F1 does not)</h2>
    <p style="font-size:.82rem;color:#64748b;margin-bottom:14px">
      F1 of 23% on a clean generative system understates quality due to paraphrasing.
      <strong>SemSim (sentence embedding cosine similarity) is the correct primary metric</strong>
      for evaluating retrieval-augmented generation degradation.
    </p>
    """+curve_bars+"""
  </div>

  """+exp_sections+"""
</div>
<div class="footer">FedRAG Data Poisoning -- SemSim Primary -- Strategic + 50% + 70% -- """+now+"""</div>
</body></html>""")

Path("data_poisoning_report.html").write_text(html, encoding="utf-8")
print("\nReport saved: data_poisoning_report.html")
