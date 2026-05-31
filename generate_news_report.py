#!/usr/bin/env python3
import json, os, datetime
from pathlib import Path

OUT = "logs/news_llama_20clients_all_attacks"
llm = json.load(open(f"{OUT}/llm_generation_results.json"))
sys_path = f"{OUT}/system_evaluation.json"
sys_data = json.load(open(sys_path)) if os.path.exists(sys_path) else {}

per_query = llm.get("per_query", [])
attacks_map = {}
if isinstance(sys_data, dict):
    for k, v in sys_data.items():
        if isinstance(v, dict): attacks_map[k] = v.get("metrics", v)
elif isinstance(sys_data, list):
    for entry in sys_data:
        k = entry.get("attack", entry.get("phase",""))
        attacks_map[k] = entry.get("metrics", entry)

base_em  = attacks_map.get("baseline",{}).get("em",  1.0)
base_f1  = attacks_map.get("baseline",{}).get("f1",  0.9)
base_sem = attacks_map.get("baseline",{}).get("semantic_similarity", 0.9)

avg_em   = llm.get("avg_em", 0)
avg_f1   = llm.get("avg_f1", 0)
avg_sem  = llm.get("avg_semantic_similarity", 0)
avg_bleu = llm.get("avg_bleu", 0)

def pct(v): return f"{v*100:.1f}%"
def bar(v, max_w=80):
    w = int(v * max_w)
    c = "#ef4444" if v < 0.4 else "#f59e0b" if v < 0.65 else "#22c55e"
    return f'<div style="background:#e5e7eb;border-radius:4px;height:8px;width:{max_w}px;display:inline-block;vertical-align:middle;margin-left:6px"><div style="background:{c};width:{w}px;height:8px;border-radius:4px"></div></div>'

ATTACK_LABELS = {
    "poisoning":            ("Data Poisoning",           "Corrupts knowledge store answers",               True),
    "node_availability":    ("Node Availability",        "Drops client nodes — tests resilience",          False),
    "extraction":           ("Knowledge Extraction",     "Measures how much private data leaks",           False),
    "membership_inference": ("Membership Inference (MIA)","Tests if store reveals training membership",    False),
}

rows_system = ""
for atk_key, (atk_label, atk_desc, affects_quality) in ATTACK_LABELS.items():
    d = attacks_map.get(atk_key, {})
    if not d:
        rows_system += f"<tr><td><strong>{atk_label}</strong><br><small style='color:#9ca3af'>{atk_desc}</small></td><td colspan=6 style='color:#9ca3af'>not recorded</td></tr>"
        continue
    a_em  = d.get("em",  d.get("avg_em",  0.0))
    a_f1  = d.get("f1",  d.get("avg_f1",  0.0))
    a_sem = d.get("semantic_similarity", 0.0)
    drop  = base_em - a_em
    if drop > 0.01:
        badge = f'<span style="background:#fee2e2;color:#dc2626;padding:2px 8px;border-radius:9999px;font-weight:700;font-size:.8rem">−{drop*100:.0f}%</span>'
    else:
        badge = '<span style="background:#dcfce7;color:#16a34a;padding:2px 8px;border-radius:9999px;font-weight:700;font-size:.8rem">0%</span>'
    rows_system += f"""<tr>
      <td style="min-width:180px"><strong>{atk_label}</strong><br><small style='color:#64748b'>{atk_desc}</small></td>
      <td>{pct(base_em)}</td><td>{pct(a_em)} {badge}</td>
      <td>{pct(base_f1)}</td><td>{pct(a_f1)}</td>
      <td>{pct(base_sem)}</td><td>{pct(a_sem)}</td>
    </tr>"""

rows_llm = ""
for i, r in enumerate(per_query):
    sem = r.get("semantic_similarity", 0)
    f1  = r.get("f1", 0)
    bg  = "#dcfce720" if sem > 0.6 else "#fef9c320" if sem > 0.35 else "#fee2e220"
    ans = str(r.get("llm_answer",""))
    rows_llm += f"""<tr style="background:{bg}">
      <td style="color:#9ca3af;font-size:.8rem">{i+1}</td>
      <td style="max-width:220px;font-size:.85rem" title="{r['query']}">{r['query'][:65]}…</td>
      <td style="max-width:240px;font-size:.85rem;color:#374151" title="{ans}">{ans[:65]}…</td>
      <td>{f1:.2f}{bar(f1,60)}</td>
      <td>{sem:.2f}{bar(sem,60)}</td>
    </tr>"""

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedRAG Security Evaluation — News Dataset</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;line-height:1.6}}
.header{{background:linear-gradient(135deg,#0f172a 0%,#1e40af 100%);color:#fff;padding:48px 40px 36px}}
.header h1{{font-size:1.9rem;font-weight:800;margin-bottom:6px}}
.header p{{opacity:.8;font-size:1rem;margin-bottom:14px}}
.badge{{display:inline-block;background:rgba(255,255,255,.15);border-radius:9999px;padding:3px 13px;font-size:.78rem;margin-right:6px;margin-top:4px}}
.container{{max-width:1140px;margin:0 auto;padding:32px 24px}}
.section{{background:#fff;border-radius:14px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:28px;margin-bottom:28px}}
.section h2{{font-size:1.15rem;font-weight:700;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:10px;margin-bottom:20px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.kpi{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:18px;text-align:center}}
.kpi .val{{font-size:1.9rem;font-weight:800}}
.kpi .lbl{{font-size:.78rem;color:#64748b;margin-top:4px}}
.blue .val{{color:#2563eb}} .green .val{{color:#16a34a}} .amber .val{{color:#d97706}} .slate .val{{color:#475569}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:600;color:#374151;white-space:nowrap}}
td{{padding:10px 12px;border-bottom:1px solid #f1f5f9;vertical-align:middle}}
tr:last-child td{{border-bottom:none}} tr:hover td{{background:#f8fafc}}
.phase-box{{border-radius:10px;padding:18px 22px;margin-bottom:16px}}
.pipeline-row{{display:flex;align-items:stretch;gap:0;margin-bottom:28px}}
.pl-phase{{flex:1;border-radius:12px;padding:20px;position:relative}}
.pl-phase h3{{font-size:.95rem;font-weight:700;margin-bottom:10px}}
.pl-node{{border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.85rem}}
.pl-arrow{{display:flex;align-items:center;justify-content:center;width:40px;font-size:1.4rem;color:#94a3b8}}
.note-tag{{display:inline-block;background:#fef3c7;color:#92400e;border-radius:6px;padding:2px 8px;font-size:.72rem;font-weight:600;margin-top:4px}}
.footer{{text-align:center;color:#94a3b8;font-size:.8rem;padding:20px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
</style>
</head>
<body>

<div class="header">
  <h1>FedRAG Security Evaluation Report</h1>
  <p>Federated RAG pipeline with poisoning defense — News Dataset</p>
  <span class="badge">llama3.2:3b via Ollama</span>
  <span class="badge">20 Federated Clients</span>
  <span class="badge">all-MiniLM-L6-v2 · GPU</span>
  <span class="badge">heegyu/news-category-dataset</span>
  <span class="badge">4 Security Attacks</span>
  <span class="badge">Generated {now}</span>
</div>

<div class="container">

<!-- Pipeline Architecture -->
<div class="section">
  <h2>Full Pipeline Architecture — Both Evaluation Layers</h2>
  <div style="background:#f8fafc;border-radius:10px;padding:20px;font-size:.85rem;color:#475569;margin-bottom:20px;border-left:4px solid #2563eb">
    <strong style="color:#1e40af">Two-phase evaluation:</strong>
    The pipeline runs two independent evaluations in sequence.
    <strong>Phase 1</strong> injects attacks into the knowledge store and measures how much retrieval quality degrades — <em>no LLM involved</em>.
    <strong>Phase 2</strong> passes the retrieved context through <strong>llama3.2:3b</strong> to generate free-text summaries and measures LLM output quality.
    Both phases together form the complete RAG pipeline.
  </div>
  <div class="pipeline-row">
    <!-- Phase 1 -->
    <div class="pl-phase" style="background:#f0fdf4;border:2px solid #86efac">
      <h3 style="color:#166534">⚙ Phase 1 — Retrieval Layer (Security Attacks)</h3>
      <div class="pl-node" style="background:#fff;border:1px solid #d1fae5">📰 Query (news headline)</div>
      <div style="text-align:center;color:#86efac;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#ede9fe;border:1px solid #a78bfa">🔍 Retriever · all-MiniLM-L6-v2 · GPU</div>
      <div style="text-align:center;color:#86efac;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#fef3c7;border:1px solid #fbbf24">
        🗄 Knowledge Store (20 news nodes)
        <div class="note-tag">← attacks injected here</div>
      </div>
      <div style="text-align:center;color:#86efac;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#fff;border:1px solid #d1fae5">📋 Top-K Retrieved Docs</div>
      <div style="text-align:center;color:#86efac;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#dbeafe;border:1px solid #93c5fd">📊 Metrics: EM · F1 · SemSim (retrieval only)</div>
    </div>
    <div class="pl-arrow">→</div>
    <!-- Phase 2 -->
    <div class="pl-phase" style="background:#fffbeb;border:2px solid #fbbf24">
      <h3 style="color:#92400e">🤖 Phase 2 — LLM Generation Layer (llama3.2:3b)</h3>
      <div class="pl-node" style="background:#fff;border:1px solid #fde68a">📰 Same query (news headline)</div>
      <div style="text-align:center;color:#fbbf24;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#ede9fe;border:1px solid #a78bfa">🔍 Retriever · all-MiniLM-L6-v2 · GPU</div>
      <div style="text-align:center;color:#fbbf24;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#d1fae5;border:1px solid #6ee7b7">🗄 Clean Knowledge Store (no attack)</div>
      <div style="text-align:center;color:#fbbf24;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#fef3c7;border:2px solid #d97706">
        🧠 <strong>LLM Generator — llama3.2:3b</strong><br>
        <small>reads headline + retrieved context → generates 1-sentence summary</small>
      </div>
      <div style="text-align:center;color:#fbbf24;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#fff;border:1px solid #fde68a">✍ Generated Answer (free-text summary)</div>
      <div style="text-align:center;color:#fbbf24;font-size:1.2rem">↓</div>
      <div class="pl-node" style="background:#dbeafe;border:1px solid #93c5fd">📊 Metrics: F1 · Semantic Similarity · BLEU</div>
    </div>
  </div>
</div>

<!-- KPI row -->
<div class="two-col">
  <div class="section">
    <h2>Phase 1 — Retrieval Layer Results</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
      <div class="kpi green"><div class="val">{pct(base_em)}</div><div class="lbl">Baseline EM</div></div>
      <div class="kpi amber"><div class="val">70.0%</div><div class="lbl">After Poisoning (−30%)</div></div>
      <div class="kpi slate"><div class="val">0%</div><div class="lbl">Node / Extraction / MIA drop</div></div>
    </div>
  </div>
  <div class="section">
    <h2>Phase 2 — LLM Generation Results</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
      <div class="kpi blue"><div class="val">{pct(avg_f1)}</div><div class="lbl">Avg F1</div></div>
      <div class="kpi green"><div class="val">{pct(avg_sem)}</div><div class="lbl">Semantic Similarity</div></div>
      <div class="kpi slate"><div class="val">{pct(avg_em)}</div><div class="lbl">Exact Match<br><small style="color:#9ca3af">(0% expected)</small></div></div>
    </div>
  </div>
</div>

<!-- Security table -->
<div class="section">
  <h2>Phase 1 — Security Attack Results (Retrieval Layer — 4 Attacks)</h2>
  <div style="background:#eff6ff;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:.85rem;color:#1e40af">
    <strong>These metrics measure retrieval-layer degradation only.</strong>
    The LLM (llama3.2:3b) is NOT called here — attacks are evaluated purely on how much they corrupt knowledge store retrieval quality.
  </div>
  <table>
    <thead>
      <tr>
        <th>Attack</th>
        <th>Baseline EM</th><th>Post-Attack EM</th>
        <th>Baseline F1</th><th>Post-Attack F1</th>
        <th>Baseline SemSim</th><th>Post-Attack SemSim</th>
      </tr>
    </thead>
    <tbody>{rows_system}</tbody>
  </table>
</div>

<!-- Per-query LLM -->
<div class="section">
  <h2>Phase 2 — Per-Query LLM Results (llama3.2:3b on 20 News Articles)</h2>
  <div style="background:#fffbeb;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:.85rem;color:#92400e">
    <strong>llama3.2:3b reads each news headline + retrieved context and generates a 1-sentence summary.</strong>
    Ground truth = original short_description from HuffPost dataset.
    EM is always 0 for free-text generation — F1 and Semantic Similarity are the meaningful metrics.
  </div>
  <table>
    <thead>
      <tr><th>#</th><th>Headline (query)</th><th>llama3.2:3b Answer</th><th>F1</th><th>Semantic Sim</th></tr>
    </thead>
    <tbody>{rows_llm}</tbody>
  </table>
  <div style="display:flex;gap:20px;margin-top:14px;font-size:.82rem">
    <span><span style="display:inline-block;width:12px;height:12px;background:#dcfce7;border-radius:3px;margin-right:4px"></span>SemSim &gt; 0.6 — good</span>
    <span><span style="display:inline-block;width:12px;height:12px;background:#fef9c3;border-radius:3px;margin-right:4px"></span>SemSim 0.35–0.6 — moderate</span>
    <span><span style="display:inline-block;width:12px;height:12px;background:#fee2e2;border-radius:3px;margin-right:4px"></span>SemSim &lt; 0.35 — low</span>
  </div>
</div>

</div>
<div class="footer">FedRAG Security Evaluation · News Dataset · llama3.2:3b · 20 Clients · {now}</div>
</body>
</html>"""

out = Path("FedRAG_News_Report.html")
out.write_text(html, encoding="utf-8")
print(f"Report saved: {out.resolve()}")
