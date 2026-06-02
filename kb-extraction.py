#!/usr/bin/env python3
"""
KB Extraction Attack -- 3 probe strength levels + fixed leakage threshold.
Level 1: exact headline (upper bound, trivial)
Level 2: paraphrased -- key words rearranged, stop words dropped (realistic)
Level 3: topic + 3 keywords only (lower bound, noisy)
Generates: kb_extraction_report.html
"""
import json, random, datetime
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
    def cosine(a,b):
        a,b = np.array(a),np.array(b); d = np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d else 0.0
    HAS_ENC = True
except Exception:
    HAS_ENC = False
    def encode(t):
        import hashlib; h = hashlib.md5(t.encode()).digest()
        return [(b/128.0)-1.0 for b in h*24][:384]
    def cosine(a,b): return 0.0

from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

NUM_SAMPLES     = 20
SEED            = 0
LEAKAGE_THRESH  = 0.65   # meaningful once queries are not exact matches
OUT_DIR         = Path("logs/kb_extraction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOP = {'the','a','an','is','are','was','were','be','been','has','have','had',
        'will','would','could','should','may','might','do','does','did','to',
        'of','in','on','at','by','for','with','from','and','or','but','not',
        'that','this','it','he','she','they','we','i','you','its','their',
        'about','after','before','over','under','into','through','during',
        'says','said','as','s','have','been','his','her','our','his','new'}

def make_paraphrase(headline):
    """Level 2: drop stop words, take content words in reverse-subject order."""
    words = [w for w in headline.split() if w.lower().rstrip('.,?!') not in STOP and len(w) > 3]
    if len(words) >= 4:
        # rotate: put last third first (simulate attacker who heard about it obliquely)
        pivot = max(1, len(words)*2//3)
        reordered = words[pivot:] + words[:pivot]
        return ' '.join(reordered[:7])
    return ' '.join(words[:5]) if words else headline[:30]

def make_topic_query(topic, headline):
    """Level 3: topic label + 3 most informative words only."""
    words = [w for w in headline.split() if w.lower().rstrip('.,?!') not in STOP and len(w) > 4][:3]
    topic_clean = topic.lower().replace('_',' ').replace('&','and')
    return f"{topic_clean} {' '.join(words)}"

rng = random.Random(SEED)
print("Loading dataset ...")
if HAS_DS:
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    ds = ds.shuffle(seed=SEED).select(range(NUM_SAMPLES*2))
    all_rows = [{"query": str(r["headline"]), "answer": str(r["short_description"]),
                 "topic": str(r["category"]), "idx": i}
                for i, r in enumerate(ds) if r.get("headline")]
else:
    all_rows = [{"query": f"News story {i}", "answer": f"Summary {i}.",
                 "topic": "GENERAL", "idx": i} for i in range(NUM_SAMPLES*2)]

members    = all_rows[:NUM_SAMPLES]
nonmembers = all_rows[NUM_SAMPLES:NUM_SAMPLES*2]

print(f"  {len(members)} articles in private knowledge store")
print("Building knowledge store ...")
nodes = []
for i, r in enumerate(members):
    nodes.append(KnowledgeNode(
        node_type=NodeType.TEXT, text_content=r["query"],
        embedding=encode(r["query"]),
        metadata={"answer": r["answer"], "topic": r["topic"], "client": i},
    ))
store = InMemoryKnowledgeStore.from_nodes(nodes)

def probe_store(query_text):
    hits = store.retrieve(encode(query_text), top_k=1)
    if hits:
        score, node = hits[0]
        return float(score), node.metadata.get("answer",""), node.text_content
    return 0.0, "", ""

# ── run 3 probe levels ─────────────────────────────────────────────────────────
levels = {
    "L1_exact":     {"label": "Level 1 -- Exact Headline (upper bound)",
                     "color": "#dc2626", "bg": "#fee2e2", "desc":
                     "Attacker already has the exact stored text. Score = 1.0 always. Trivial -- not a realistic attack."},
    "L2_paraphrase":{"label": "Level 2 -- Paraphrased / Key Words Rearranged (realistic)",
                     "color": "#d97706", "bg": "#fef3c7", "desc":
                     "Attacker knows the topic and approximate content but not exact wording. Score ~0.5-0.85. This is where the threshold matters."},
    "L3_topic":     {"label": "Level 3 -- Topic + Keywords Only (lower bound)",
                     "color": "#16a34a", "bg": "#dcfce7", "desc":
                     "Attacker only knows the category and a few words. Score ~0.3-0.6. Many false positives -- noisy, hard to exploit."},
}

def run_level(level_key):
    probes = []
    for r in members:
        if level_key == "L1_exact":
            probe_q = r["query"]
        elif level_key == "L2_paraphrase":
            probe_q = make_paraphrase(r["query"])
        else:
            probe_q = make_topic_query(r["topic"], r["query"])

        score, leaked_ans, ret_text = probe_store(probe_q)
        true_ans = r["answer"]

        # ── FIXED: sim=0 must never count as leaked ────────────────────────────
        if not leaked_ans or score < 0.01:
            sim = 0.0
            is_leaked = False
        else:
            sim = cosine(encode(leaked_ans), encode(true_ans)) if HAS_ENC else 0.0
            # Only count as leaked if BOTH score is non-trivial AND semantic sim is high
            is_leaked = (sim >= LEAKAGE_THRESH) and (score > 0.01)

        exact = leaked_ans.strip().lower() == true_ans.strip().lower()
        probes.append({
            "client":   r["idx"], "topic": r["topic"],
            "true_headline": r["query"], "probe_query": probe_q,
            "true_answer": true_ans, "leaked_answer": leaked_ans,
            "retrieval_score": round(score, 4),
            "semantic_sim":    round(sim, 4),
            "is_leaked":  is_leaked,
            "exact_match": exact,
        })
        flag = "LEAKED" if is_leaked else "safe"
        print(f"  [{level_key}] C{r['idx']:02d} score={score:.3f} sim={sim:.3f} => {flag}")
        print(f"    probe: {probe_q[:60]}")

    leaked_count = sum(1 for p in probes if p["is_leaked"])
    leakage_rate = leaked_count / len(probes)
    avg_score    = sum(p["retrieval_score"] for p in probes) / len(probes)
    avg_sim      = sum(p["semantic_sim"]    for p in probes) / len(probes)
    return {"probes": probes, "leaked": leaked_count,
            "leakage_rate": leakage_rate, "avg_score": avg_score, "avg_sim": avg_sim}

results = {}
for lk in levels:
    print(f"\n--- {levels[lk]['label']} ---")
    results[lk] = run_level(lk)
    print(f"  Leakage rate: {results[lk]['leakage_rate']*100:.1f}%  avg_score={results[lk]['avg_score']:.3f}  avg_sim={results[lk]['avg_sim']:.3f}")

(OUT_DIR/"kb_extraction_results.json").write_text(json.dumps({
    "threshold": LEAKAGE_THRESH, "num_members": NUM_SAMPLES, "results": {
        k: {kk:vv for kk,vv in v.items() if kk != "probes"} for k,v in results.items()
    }
}, indent=2))

# ── HTML ───────────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def sbar(v, w=60):
    c = "#ef4444" if v>=LEAKAGE_THRESH else "#22c55e"
    return ('<div style="display:flex;align-items:center;gap:5px">'
            '<div style="background:#e5e7eb;border-radius:4px;height:7px;width:'+str(w)+'px">'
            '<div style="background:'+c+';width:'+str(int(v*w))+'px;height:7px;border-radius:4px"></div></div>'
            '<span style="font-size:.75rem">'+f'{v:.3f}'+'</span></div>')

level_sections = ""
for lk, meta in levels.items():
    res = results[lk]
    table_rows = ""
    for p in res["probes"]:
        bg = "#fee2e215" if p["is_leaked"] else ""
        badge = ('<span style="background:#fee2e2;color:#dc2626;padding:1px 6px;border-radius:9999px;font-size:.7rem;font-weight:700">LEAKED</span>'
                 if p["is_leaked"] else
                 '<span style="background:#dcfce7;color:#16a34a;padding:1px 6px;border-radius:9999px;font-size:.7rem">safe</span>')
        exact_badge = (' <span style="background:#fef3c7;color:#d97706;font-size:.68rem;padding:1px 5px;border-radius:3px">exact</span>' if p["exact_match"] else "")
        table_rows += (
            '<tr style="background:'+bg+'">'
            '<td style="color:#94a3b8;font-size:.75rem;text-align:center">'+str(p["client"]).zfill(2)+'</td>'
            '<td style="font-size:.75rem;color:#64748b">'+p["topic"][:14]+'</td>'
            '<td style="font-size:.75rem">'+p["probe_query"].replace("<","&lt;")[:55]+'</td>'
            '<td style="font-size:.75rem;color:#047857;font-style:italic">'+p["true_answer"].replace("<","&lt;")[:60]+'</td>'
            '<td style="font-size:.75rem;color:#dc2626">'+p["leaked_answer"].replace("<","&lt;")[:60]+exact_badge+'</td>'
            '<td style="text-align:center;font-size:.75rem">'+f'{p["retrieval_score"]:.3f}'+'</td>'
            '<td>'+sbar(p["semantic_sim"])+'</td>'
            '<td>'+badge+'</td>'
            '</tr>')
    level_sections += (
        '<div class="card">'
        '<h2 style="color:'+meta["color"]+'">'+meta["label"]+'</h2>'
        '<div style="background:'+meta["bg"]+';border-left:4px solid '+meta["color"]+';border-radius:6px;'
        'padding:10px 14px;font-size:.84rem;color:'+meta["color"]+';margin-bottom:14px">'
        +meta["desc"]+'<br>'
        '<strong>Leakage rate: '+f"{res['leakage_rate']*100:.1f}%"+'</strong>'
        ' &nbsp;|&nbsp; Avg retrieval score: '+f"{res['avg_score']:.3f}"
        +' &nbsp;|&nbsp; Avg semantic sim: '+f"{res['avg_sim']:.3f}"
        +'</div>'
        '<table><thead><tr>'
        '<th>C#</th><th>Topic</th><th>Probe Query (what attacker sends)</th>'
        '<th>Private True Answer</th><th>Extracted Content</th>'
        '<th>Score</th><th>Sem Sim</th><th>Status</th>'
        '</tr></thead><tbody>'+table_rows+'</tbody></table>'
        '</div>')

leakage_curve = ""
for lk, meta in levels.items():
    r = results[lk]
    pct = r["leakage_rate"]*100
    w = int(pct * 5)
    leakage_curve += (
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">'
        '<div style="width:220px;font-size:.85rem;font-weight:600;color:'+meta["color"]+'">'+meta["label"].split("--")[0].strip()+'</div>'
        '<div style="background:#e5e7eb;border-radius:6px;height:20px;width:500px;flex-shrink:0">'
        '<div style="background:'+meta["color"]+';width:'+str(w)+'px;height:20px;border-radius:6px;'
        'display:flex;align-items:center;padding-left:8px;color:#fff;font-size:.78rem;font-weight:700">'
        +(f'{pct:.0f}%' if w>25 else '')+'</div></div>'
        '<span style="font-size:.85rem;color:#374151;font-weight:600">'+f'{pct:.1f}%'+'</span>'
        '</div>')

html = ("""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KB Extraction Attack Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.header{background:linear-gradient(135deg,#1e3a5f,#7c3aed);color:#fff;padding:40px}
.header h1{font-size:1.7rem;font-weight:800;margin-bottom:6px}
.header p{opacity:.8;font-size:.95rem;margin-bottom:12px}
.badge{display:inline-block;background:rgba(255,255,255,.15);border-radius:9999px;padding:3px 12px;font-size:.75rem;margin-right:6px}
.container{max-width:1350px;margin:0 auto;padding:28px 20px}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:24px;margin-bottom:24px}
.card h2{font-size:1.05rem;font-weight:700;border-bottom:2px solid #e2e8f0;padding-bottom:10px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:24px}
.kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;text-align:center}
.kpi .v{font-size:1.5rem;font-weight:800}.kpi .l{font-size:.72rem;color:#64748b;margin-top:3px}
.red .v{color:#dc2626}.amber .v{color:#d97706}.green .v{color:#16a34a}.slate .v{color:#475569}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:600;color:#374151;font-size:.76rem}
td{padding:7px 10px;border-bottom:1px solid #f1f5f9;vertical-align:top}
tr:hover td{background:#f8fafc}
.footer{text-align:center;color:#94a3b8;font-size:.75rem;padding:20px}
</style></head><body>
<div class="header">
  <h1>Knowledge Base Extraction Attack -- 3 Probe Strength Levels</h1>
  <p>Leakage curve from exact-text (trivial) to keyword-only (realistic lower bound)</p>
  <span class="badge">20 Members / 20 Non-Members</span>
  <span class="badge">Threshold: """+str(LEAKAGE_THRESH)+"""</span>
  <span class="badge">heegyu/news-category-dataset</span>
  <span class="badge">Generated """+now+"""</span>
</div>
<div class="container">
  <div class="kpis">
    <div class="kpi slate"><div class="v">"""+str(NUM_SAMPLES)+"""</div><div class="l">Private Nodes</div></div>
    <div class="kpi red"><div class="v">"""+f"{results['L1_exact']['leakage_rate']*100:.0f}%"+"""</div><div class="l">L1 Leakage (exact)</div></div>
    <div class="kpi amber"><div class="v">"""+f"{results['L2_paraphrase']['leakage_rate']*100:.0f}%"+"""</div><div class="l">L2 Leakage (paraphrase)</div></div>
    <div class="kpi green"><div class="v">"""+f"{results['L3_topic']['leakage_rate']*100:.0f}%"+"""</div><div class="l">L3 Leakage (topic only)</div></div>
    <div class="kpi slate"><div class="v">"""+f"{results['L2_paraphrase']['avg_score']:.3f}"+"""</div><div class="l">L2 Avg Score</div></div>
    <div class="kpi slate"><div class="v">"""+f"{results['L3_topic']['avg_score']:.3f}"+"""</div><div class="l">L3 Avg Score</div></div>
  </div>

  <div class="card">
    <h2>Leakage Curve -- Attacker Probe Strength vs Recovery Rate</h2>
    <p style="font-size:.84rem;color:#64748b;margin-bottom:16px">
      The threshold is <strong>"""+str(LEAKAGE_THRESH)+""" semantic similarity</strong>.
      Zero-similarity results are never counted as leaked regardless of threshold.
      L1 (exact) is trivial and not a realistic attack -- L2 and L3 are the real threat model.
    </p>
    """+leakage_curve+"""
  </div>

  """+level_sections+"""
</div>
<div class="footer">FedRAG KB Extraction -- News Dataset -- 3 Probe Levels -- """+now+"""</div>
</body></html>""")

Path("kb_extraction_report.html").write_text(html, encoding="utf-8")
print("\nReport saved: kb_extraction_report.html")
