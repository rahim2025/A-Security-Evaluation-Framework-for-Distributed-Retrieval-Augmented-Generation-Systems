#!/usr/bin/env python3
import json, datetime
from pathlib import Path

OUT = "logs/news_llama_20clients_all_attacks"
data = json.load(open(f"{OUT}/llm_generation_results.json"))

pq        = data.get("per_query", data.get("llm_baseline", {}).get("per_query", []))
avg_f1    = data.get("avg_f1", 0)
avg_sem   = data.get("avg_semantic_similarity", 0)
model     = data.get("model", "llama3.2:3b")
now       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

pq_poison  = (data.get("llm_poisoning") or {}).get("per_query", [])
poison_f1  = (data.get("llm_poisoning") or {}).get("avg_f1")
poison_sem = (data.get("llm_poisoning") or {}).get("avg_semantic_similarity")

def bar(v, w=70):
    c = "#ef4444" if v < 0.35 else "#f59e0b" if v < 0.55 else "#22c55e"
    return (
        '<div style="display:flex;align-items:center;gap:6px">'
        '<div style="background:#e5e7eb;border-radius:4px;height:8px;width:' + str(w) + 'px;flex-shrink:0">'
        '<div style="background:' + c + ';width:' + str(int(v*w)) + 'px;height:8px;border-radius:4px"></div></div>'
        '<span style="font-size:.8rem;color:#374151">' + f'{v:.3f}' + '</span></div>'
    )

def make_rows(per_query):
    out = ""
    for i, r in enumerate(per_query):
        f1  = r.get("f1", 0)
        sem = r.get("semantic_similarity", 0)
        em  = r.get("em", 0)
        gt  = r.get("ground_truth", "").replace("<","&lt;").replace(">","&gt;")
        ans = r.get("llm_answer", "").replace("<","&lt;").replace(">","&gt;")
        q   = r.get("query", "").replace("<","&lt;").replace(">","&gt;")
        bg  = "#dcfce720" if sem > 0.6 else "#fef9c320" if sem > 0.35 else "#fee2e220"
        em_badge = (
            '<span style="background:#dcfce7;color:#16a34a;padding:1px 7px;'
            'border-radius:9999px;font-size:.75rem;font-weight:700">1</span>'
            if em == 1.0 else
            '<span style="background:#f1f5f9;color:#94a3b8;padding:1px 7px;'
            'border-radius:9999px;font-size:.75rem">0</span>'
        )
        out += (
            '<tr style="background:' + bg + '">'
            '<td style="color:#94a3b8;font-size:.8rem;text-align:center">' + str(i+1) + '</td>'
            '<td style="font-size:.82rem;color:#334155">' + q + '</td>'
            '<td style="font-size:.82rem;color:#047857;font-style:italic">' + gt + '</td>'
            '<td style="font-size:.82rem;color:#1e40af">' + ans + '</td>'
            '<td style="text-align:center">' + em_badge + '</td>'
            '<td>' + bar(f1) + '</td>'
            '<td>' + bar(sem) + '</td>'
            '</tr>'
        )
    return out

poison_block = ""
if pq_poison and poison_f1 is not None:
    f1_drop  = (avg_f1 - poison_f1) * 100
    sem_drop = (avg_sem - poison_sem) * 100
    poison_block = (
        '<div class="card">'
        '<h2>After Data Poisoning Attack -- Ground Truth vs Llama Answer (poisoned context)</h2>'
        '<div style="background:#fee2e220;border-left:4px solid #ef4444;border-radius:6px;'
        'padding:12px 16px;font-size:.85rem;color:#dc2626;margin-bottom:16px">'
        '<strong>6 of 20 knowledge store nodes (30%) were corrupted.</strong> '
        'Queries that retrieved a poisoned node receive adversarial context -- Llama answer degrades. '
        'F1 drops from ' + f'{avg_f1*100:.1f}' + '% to ' + f'{poison_f1*100:.1f}' + '% '
        '&nbsp;|&nbsp; SemSim drops from ' + f'{avg_sem*100:.1f}' + '% to ' + f'{poison_sem*100:.1f}' + '%'
        '</div>'
        '<table><thead><tr>'
        '<th style="width:36px">#</th>'
        '<th style="width:200px">Query (headline)</th>'
        '<th style="width:200px">Ground Truth</th>'
        '<th style="width:200px">Llama Answer (poisoned context)</th>'
        '<th style="width:50px">EM</th>'
        '<th style="width:120px">F1</th>'
        '<th style="width:120px">Semantic Sim</th>'
        '</tr></thead>'
        '<tbody>' + make_rows(pq_poison) + '</tbody>'
        '</table>'
        '<div class="legend">'
        '<span><span class="dot" style="background:#dcfce7"></span>Sem Sim &gt; 0.6 -- good</span>'
        '<span><span class="dot" style="background:#fef9c3"></span>0.35-0.6 -- moderate</span>'
        '<span><span class="dot" style="background:#fee2e2"></span>&lt; 0.35 -- low</span>'
        '</div></div>'
    )

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedRAG News Report</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.header{background:linear-gradient(135deg,#0f172a,#1e40af);color:#fff;padding:40px}
.header h1{font-size:1.7rem;font-weight:800;margin-bottom:6px}
.header p{opacity:.8;font-size:.95rem;margin-bottom:12px}
.badge{display:inline-block;background:rgba(255,255,255,.15);border-radius:9999px;padding:3px 12px;font-size:.75rem;margin-right:6px}
.container{max-width:1350px;margin:0 auto;padding:28px 20px}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:24px;margin-bottom:24px}
.card h2{font-size:1.05rem;font-weight:700;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:10px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px}
.kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center}
.kpi .v{font-size:1.8rem;font-weight:800}
.kpi .l{font-size:.75rem;color:#64748b;margin-top:3px}
.blue .v{color:#2563eb}.green .v{color:#16a34a}.amber .v{color:#d97706}.red .v{color:#dc2626}.slate .v{color:#475569}
.em-note{background:#eff6ff;border-left:4px solid #2563eb;border-radius:6px;padding:12px 16px;font-size:.85rem;color:#1e40af;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:600;color:#374151;font-size:.8rem;white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid #f1f5f9;vertical-align:top}
tr:hover td{background:#f8fafc}
.legend{display:flex;gap:18px;font-size:.78rem;margin-top:12px;color:#64748b}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
.footer{text-align:center;color:#94a3b8;font-size:.75rem;padding:20px}
</style>
</head>
<body>
<div class="header">
  <h1>FedRAG News -- Ground Truth vs Llama Answers</h1>
  <p>Per-query comparison: what llama3.2:3b generated vs the actual ground truth summary</p>
  <span class="badge">""" + model + """</span>
  <span class="badge">20 Federated Clients</span>
  <span class="badge">heegyu/news-category-dataset</span>
  <span class="badge">Generated """ + now + """</span>
</div>
<div class="container">
  <div class="kpis">
    <div class="kpi slate"><div class="v">""" + str(len(pq)) + """</div><div class="l">Queries</div></div>
    <div class="kpi slate"><div class="v">0.00</div><div class="l">Exact Match<br><small style="color:#9ca3af">(always 0 -- see note)</small></div></div>
    <div class="kpi blue"><div class="v">""" + f'{avg_f1*100:.1f}' + """%</div><div class="l">Avg F1 (clean store)</div></div>
    <div class="kpi green"><div class="v">""" + f'{avg_sem*100:.1f}' + """%</div><div class="l">Avg Semantic Sim (clean)</div></div>
    <div class="kpi """ + ("red" if poison_f1 and poison_f1 < avg_f1 else "amber") + """"><div class="v">""" + (f'{poison_f1*100:.1f}%' if poison_f1 is not None else 'n/a') + """</div><div class="l">F1 After Poisoning</div></div>
  </div>
  <div class="em-note">
    <strong>Why is Exact Match (EM) always 0?</strong> -- EM requires a character-perfect match.
    For free-text news summarisation the LLM paraphrases, so EM is always 0. That is correct and expected.
    <strong>F1 (word overlap) and Semantic Similarity (meaning) are the real metrics here.</strong>
    EM only makes sense for multiple-choice tasks like MMLU where the answer is exactly "A" or "2".
  </div>
  <div class="card">
    <h2>Baseline -- Clean Store (no attack) -- Ground Truth vs Llama Answer</h2>
    <table>
      <thead><tr>
        <th style="width:36px">#</th>
        <th style="width:200px">Query (headline)</th>
        <th style="width:210px">Ground Truth (HuffPost short_description)</th>
        <th style="width:210px">Llama Answer (llama3.2:3b generated)</th>
        <th style="width:50px">EM</th>
        <th style="width:120px">F1</th>
        <th style="width:120px">Semantic Sim</th>
      </tr></thead>
      <tbody>""" + make_rows(pq) + """</tbody>
    </table>
    <div class="legend">
      <span><span class="dot" style="background:#dcfce7"></span>Semantic Sim &gt; 0.6 -- good</span>
      <span><span class="dot" style="background:#fef9c3"></span>0.35-0.6 -- moderate</span>
      <span><span class="dot" style="background:#fee2e2"></span>&lt; 0.35 -- low</span>
    </div>
  </div>
  """ + poison_block + """
</div>
<div class="footer">FedRAG Security Evaluation -- News Dataset -- """ + model + """ -- 20 Clients -- """ + now + """</div>
</body>
</html>"""

Path("FedRAG_News_Report.html").write_text(html, encoding="utf-8")
print("Report saved: FedRAG_News_Report.html")
