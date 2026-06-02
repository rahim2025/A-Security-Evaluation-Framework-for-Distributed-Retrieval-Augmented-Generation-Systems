#!/usr/bin/env python3
"""
Membership Inference Attack -- full evaluation.
- 3 attacker knowledge levels (exact / paraphrase / keyword)
- AUC-ROC across all thresholds per level (not just accuracy at one threshold)
- Category-matched non-members (same topic as paired member -- harder test)
- 5 random splits, report mean +/- std
"""
import json, random, math, datetime
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
    HAS_ENC = True
except Exception:
    HAS_ENC = False
    def encode(t):
        import hashlib; h = hashlib.md5(t.encode()).digest()
        return [(b/128.0)-1.0 for b in h*24][:384]

from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore

NUM_MEMBERS    = 20
NUM_NONMEMBERS = 20
N_SPLITS       = 5
SEED           = 0
OUT_DIR        = Path("logs/membership_inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOP = {'the','a','an','is','are','was','were','be','been','has','have','had','will',
        'would','could','should','may','might','do','does','did','to','of','in','on',
        'at','by','for','with','from','and','or','but','not','that','this','it',
        'he','she','they','we','i','you','its','their','about','after','says','said','s'}

def make_paraphrase(headline):
    words = [w for w in headline.split() if w.lower().rstrip('.,?!') not in STOP and len(w) > 3]
    if len(words) >= 4:
        pivot = max(1, len(words)*2//3)
        return ' '.join((words[pivot:] + words[:pivot])[:7])
    return ' '.join(words[:5]) if words else headline[:30]

def make_topic_query(topic, headline):
    words = [w for w in headline.split() if w.lower().rstrip('.,?!') not in STOP and len(w) > 4][:3]
    return f"{topic.lower().replace('_',' ').replace('&','and')} {' '.join(words)}"

# ── AUC-ROC ───────────────────────────────────────────────────────────────────
def compute_auc_roc(member_scores, nonmember_scores):
    labeled = [(s, 1) for s in member_scores] + [(s, 0) for s in nonmember_scores]
    # compute ROC points across all unique score thresholds
    thresholds = sorted(set(s for s,_ in labeled), reverse=True) + [0.0]
    points = [(0.0, 0.0)]
    n_pos = sum(1 for _,l in labeled if l==1)
    n_neg = sum(1 for _,l in labeled if l==0)
    for thresh in thresholds:
        tp = sum(1 for s,l in labeled if l==1 and s >= thresh)
        fp = sum(1 for s,l in labeled if l==0 and s >= thresh)
        tpr = tp/n_pos if n_pos else 0.0
        fpr = fp/n_neg if n_neg else 0.0
        points.append((fpr, tpr))
    points.append((1.0, 1.0))
    # trapezoidal AUC
    pts = sorted(set(points))
    auc = sum((pts[i+1][0]-pts[i][0]) * (pts[i+1][1]+pts[i][1])/2 for i in range(len(pts)-1))
    return round(auc, 4), pts

def std(vals):
    if len(vals) < 2: return 0.0
    m = sum(vals)/len(vals)
    return math.sqrt(sum((x-m)**2 for x in vals)/len(vals))

print("Loading dataset ...")
if HAS_DS:
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    ds = ds.shuffle(seed=SEED).select(range(300))
    all_rows = [{"query": str(r["headline"]), "answer": str(r["short_description"]),
                 "topic": str(r["category"]), "idx": i}
                for i, r in enumerate(ds) if r.get("headline")]
else:
    all_rows = [{"query": f"Story {i}", "answer": f"Summary {i}.", "topic": f"CAT{i%5}", "idx": i}
                for i in range(200)]

print(f"  {len(all_rows)} total articles available for sampling")

# ── per-level config ───────────────────────────────────────────────────────────
LEVEL_META = {
    "L1_exact":      {"label": "L1 -- Exact Headline",
                      "sublabel": "Score = 1.0 always. Trivial -- only confirms retrieval works.",
                      "color": "#dc2626", "bg": "#fee2e2"},
    "L2_paraphrase": {"label": "L2 -- Paraphrased Query (realistic)",
                      "sublabel": "Attacker knows meaning, not exact text. Score ~0.5-0.85.",
                      "color": "#d97706", "bg": "#fef3c7"},
    "L3_topic":      {"label": "L3 -- Topic + Keywords Only (lower bound)",
                      "sublabel": "Category + 3 words. Noisy. Score ~0.3-0.6.",
                      "color": "#059669", "bg": "#dcfce7"},
}

def make_probe(level, row):
    if level == "L1_exact":      return row["query"]
    elif level == "L2_paraphrase": return make_paraphrase(row["query"])
    else:                          return make_topic_query(row["topic"], row["query"])

def build_store(members):
    nodes = [KnowledgeNode(
        node_type=NodeType.TEXT, text_content=r["query"],
        embedding=encode(r["query"]),
        metadata={"answer": r["answer"], "topic": r["topic"]},
    ) for r in members]
    return InMemoryKnowledgeStore.from_nodes(nodes)

def probe_score(store, query_text):
    hits = store.retrieve(encode(query_text), top_k=1)
    return float(hits[0][0]) if hits else 0.0

# ── category-matched non-member selector ──────────────────────────────────────
def select_category_matched_nonmembers(members, candidate_pool, rng):
    """For each member, pick a non-member from the SAME topic (harder test)."""
    by_topic = {}
    for r in candidate_pool:
        by_topic.setdefault(r["topic"], []).append(r)
    selected, used_idx = [], set()
    for m in members:
        pool = [r for r in by_topic.get(m["topic"], candidate_pool) if r["idx"] not in used_idx]
        if not pool:
            pool = [r for r in candidate_pool if r["idx"] not in used_idx]
        nm = rng.choice(pool)
        selected.append(nm)
        used_idx.add(nm["idx"])
    return selected

# ── multi-split evaluation ─────────────────────────────────────────────────────
split_level_results = {lk: [] for lk in LEVEL_META}
final_probes = {lk: None for lk in LEVEL_META}  # save last split for display

for split in range(N_SPLITS):
    split_rng = random.Random(split * 137 + SEED)
    shuffled = all_rows[:]
    split_rng.shuffle(shuffled)
    members    = shuffled[:NUM_MEMBERS]
    candidates = shuffled[NUM_MEMBERS:]
    nonmembers = select_category_matched_nonmembers(members, candidates, split_rng)

    print(f"\n-- Split {split+1}/{N_SPLITS} --")
    store = build_store(members)

    for lk in LEVEL_META:
        member_scores    = [probe_score(store, make_probe(lk, r)) for r in members]
        nonmember_scores = [probe_score(store, make_probe(lk, r)) for r in nonmembers]

        auc, roc_pts = compute_auc_roc(member_scores, nonmember_scores)
        # accuracy at best threshold
        best_acc, best_thresh = 0.0, 0.5
        for thresh in [i/100 for i in range(0, 101)]:
            tp = sum(1 for s in member_scores    if s >= thresh)
            tn = sum(1 for s in nonmember_scores if s <  thresh)
            acc = (tp+tn) / (len(members)+len(nonmembers))
            if acc > best_acc: best_acc, best_thresh = acc, thresh

        avg_m  = sum(member_scores)/len(member_scores)
        avg_nm = sum(nonmember_scores)/len(nonmember_scores)
        score_gap = avg_m - avg_nm
        adv = best_acc - 0.5

        split_level_results[lk].append({
            "auc": auc, "accuracy": best_acc, "adv": adv,
            "avg_member": avg_m, "avg_nonmember": avg_nm,
            "score_gap": score_gap, "roc_pts": roc_pts,
        })

        if split == N_SPLITS - 1:
            probe_rows = []
            for i,(r,s) in enumerate(zip(members, member_scores)):
                q = make_probe(lk, r)
                pred = "member" if s >= best_thresh else "non-member"
                probe_rows.append({"query": r["query"], "probe_query": q, "topic": r["topic"],
                                    "true_label": "member", "score": round(s,4),
                                    "predicted": pred, "correct": pred=="member"})
            for i,(r,s) in enumerate(zip(nonmembers, nonmember_scores)):
                q = make_probe(lk, r)
                pred = "member" if s >= best_thresh else "non-member"
                probe_rows.append({"query": r["query"], "probe_query": q, "topic": r["topic"],
                                    "true_label": "non-member", "score": round(s,4),
                                    "predicted": pred, "correct": pred=="non-member"})
            final_probes[lk] = probe_rows

        print(f"  [{lk}] AUC={auc:.3f}  acc={best_acc*100:.1f}%  adv={adv*100:+.1f}%  gap={score_gap:+.3f}")

# ── aggregate stats ────────────────────────────────────────────────────────────
agg = {}
for lk in LEVEL_META:
    aucs = [r["auc"]      for r in split_level_results[lk]]
    accs = [r["accuracy"] for r in split_level_results[lk]]
    advs = [r["adv"]      for r in split_level_results[lk]]
    gaps = [r["score_gap"]for r in split_level_results[lk]]
    agg[lk] = {
        "auc_mean":  round(sum(aucs)/len(aucs), 4),
        "auc_std":   round(std(aucs), 4),
        "acc_mean":  round(sum(accs)/len(accs), 4),
        "acc_std":   round(std(accs), 4),
        "adv_mean":  round(sum(advs)/len(advs), 4),
        "gap_mean":  round(sum(gaps)/len(gaps), 4),
        "roc_pts":   split_level_results[lk][-1]["roc_pts"],
    }
    print(f"\n[{lk}] AUC = {agg[lk]['auc_mean']:.3f} +/- {agg[lk]['auc_std']:.3f}  "
          f"Acc = {agg[lk]['acc_mean']*100:.1f}% +/- {agg[lk]['acc_std']*100:.1f}%")

(OUT_DIR/"membership_inference_results.json").write_text(json.dumps(
    {k: {kk:vv for kk,vv in v.items() if kk!="roc_pts"} for k,v in agg.items()}, indent=2))

# ── HTML ───────────────────────────────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

def roc_svg(pts, color, w=260, h=180):
    # map FPR/TPR to SVG coords
    pad = 30
    def tx(fpr): return pad + fpr*(w-2*pad)
    def ty(tpr): return h-pad - tpr*(h-2*pad)
    path_d = " ".join(f"{'M' if i==0 else 'L'}{tx(p[0]):.1f},{ty(p[1]):.1f}" for i,p in enumerate(pts))
    diag   = f"M{tx(0)},{ty(0)} L{tx(1)},{ty(1)}"
    return (
        f'<svg width="{w}" height="{h}" style="overflow:visible">'
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#e2e8f0" stroke-width="1"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#e2e8f0" stroke-width="1"/>'
        f'<path d="{diag}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,3" fill="none"/>'
        f'<path d="{path_d}" stroke="{color}" stroke-width="2" fill="none"/>'
        f'<text x="{pad}" y="{h-2}" font-size="9" fill="#94a3b8">0</text>'
        f'<text x="{w-pad-4}" y="{h-2}" font-size="9" fill="#94a3b8">1</text>'
        f'<text x="{pad-28}" y="{pad+4}" font-size="9" fill="#94a3b8">1.0</text>'
        f'<text x="{pad-28}" y="{h-pad}" font-size="9" fill="#94a3b8">0.0</text>'
        f'<text x="{pad+(w-2*pad)//2 - 12}" y="{h-2}" font-size="9" fill="#64748b">FPR</text>'
        f'</svg>')

def pred_badge(pred, correct):
    if pred == "member" and correct:
        return '<span style="background:#fee2e2;color:#dc2626;padding:1px 5px;border-radius:9999px;font-size:.7rem;font-weight:600">member -- TP</span>'
    if pred == "non-member" and correct:
        return '<span style="background:#dcfce7;color:#16a34a;padding:1px 5px;border-radius:9999px;font-size:.7rem">non-member -- TN</span>'
    if pred == "member" and not correct:
        return '<span style="background:#fef3c7;color:#d97706;padding:1px 5px;border-radius:9999px;font-size:.7rem">member -- FP</span>'
    return '<span style="background:#fef3c7;color:#d97706;padding:1px 5px;border-radius:9999px;font-size:.7rem">non-member -- FN</span>'

level_cards = ""
for lk, meta in LEVEL_META.items():
    a = agg[lk]
    risk = "HIGH" if a["adv_mean"]>0.25 else "MEDIUM" if a["adv_mean"]>0.10 else "LOW"
    risk_c = "#dc2626" if risk=="HIGH" else "#d97706" if risk=="MEDIUM" else "#16a34a"
    trivial_note = ""
    if lk == "L1_exact":
        trivial_note = '<div style="background:#fee2e2;border-left:4px solid #dc2626;border-radius:6px;padding:10px 14px;font-size:.82rem;color:#dc2626;margin-bottom:12px"><strong>This level is trivial and should NOT be cited as evidence of a real attack.</strong> Score=1.0 for members is guaranteed by deterministic embeddings -- it only confirms retrieval works, not that membership inference is possible.</div>'

    probes = final_probes[lk] or []
    table_rows = ""
    for p in probes:
        is_mem = p["true_label"]=="member"
        bg = "#fff0f015" if is_mem else ""
        sc = p["score"]
        sc_c = "#dc2626" if sc > 0.7 else "#d97706" if sc > 0.45 else "#16a34a"
        table_rows += (
            '<tr style="background:'+bg+'">'
            '<td style="font-size:.72rem;color:#64748b">'+p["topic"][:12]+'</td>'
            '<td style="font-size:.72rem">'+p["probe_query"].replace("<","&lt;")[:55]+'</td>'
            '<td><span style="background:'+("dbeafe" if is_mem else "f1f5f9")+';color:'+("#1d4ed8" if is_mem else "#64748b")+';padding:1px 5px;border-radius:9999px;font-size:.7rem">'+p["true_label"]+'</span></td>'
            '<td style="text-align:center;font-size:.78rem;font-weight:600;color:'+sc_c+'">'+str(sc)+'</td>'
            '<td>'+pred_badge(p["predicted"],p["correct"])+'</td>'
            '</tr>')

    level_cards += (
        '<div class="card">'
        '<div style="display:grid;grid-template-columns:1fr auto;align-items:start;gap:20px">'
        '<div>'
        '<h2 style="color:'+meta["color"]+'">'+meta["label"]+'</h2>'
        '<div style="font-size:.83rem;color:#64748b;margin-bottom:12px">'+meta["sublabel"]+'</div>'
        +trivial_note+
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px">'
        '<div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">'
        '<div style="font-size:1.3rem;font-weight:800;color:'+meta["color"]+'">'+f'{a["auc_mean"]:.3f}'+'</div>'
        '<div style="font-size:.7rem;color:#64748b">AUC-ROC<br><small>+/-'+f'{a["auc_std"]:.3f}'+'</small></div></div>'
        '<div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">'
        '<div style="font-size:1.3rem;font-weight:800;color:'+meta["color"]+'">'+f'{a["acc_mean"]*100:.1f}%'+'</div>'
        '<div style="font-size:.7rem;color:#64748b">Best Acc<br><small>+/-'+f'{a["acc_std"]*100:.1f}'+'%</small></div></div>'
        '<div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">'
        '<div style="font-size:1.3rem;font-weight:800;color:'+meta["color"]+'">'+f'{a["adv_mean"]*100:+.1f}%'+'</div>'
        '<div style="font-size:.7rem;color:#64748b">Adv. Advantage<br><small>above 50%</small></div></div>'
        '<div style="background:#f8fafc;border-radius:8px;padding:10px;text-align:center">'
        '<div style="font-size:1.3rem;font-weight:800;color:#64748b">'+f'{a["gap_mean"]:+.3f}'+'</div>'
        '<div style="font-size:.7rem;color:#64748b">Score Gap<br><small>(M - NM)</small></div></div>'
        '</div>'
        '<div style="background:'+meta["bg"]+';border-left:4px solid '+meta["color"]+';border-radius:6px;padding:10px 14px;font-size:.82rem;color:'+meta["color"]+';margin-bottom:14px">'
        'Privacy risk: <strong style="color:'+risk_c+'">'+risk+'</strong>'
        ' &nbsp;|&nbsp; AUC=0.5 = random baseline &nbsp;|&nbsp; AUC=1.0 = perfect attacker'
        ' &nbsp;|&nbsp; 5-split mean +/- std over category-matched non-members'
        '</div>'
        '</div>'
        '<div style="text-align:center"><div style="font-size:.75rem;color:#64748b;margin-bottom:4px">ROC Curve (last split)</div>'
        +roc_svg(a["roc_pts"], meta["color"])+'</div>'
        '</div>'
        '<table><thead><tr>'
        '<th>Topic</th><th>Probe Query</th><th>True Label</th><th>Score</th><th>Prediction</th>'
        '</tr></thead><tbody>'+table_rows+'</tbody></table>'
        '</div>')

# summary AUC comparison bar
auc_bars = ""
for lk, meta in LEVEL_META.items():
    a = agg[lk]
    pct = a["auc_mean"]*100; w = int(pct*4)
    auc_bars += (
        '<div style="margin-bottom:12px">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
        '<span style="font-size:.84rem;font-weight:600;color:'+meta["color"]+'">'+meta["label"].split("--")[0].strip()+'</span>'
        '<span style="font-size:.82rem;color:#374151">AUC = '+f'{a["auc_mean"]:.3f}'+'  +/-  '+f'{a["auc_std"]:.3f}'+'</span>'
        '</div>'
        '<div style="background:#e5e7eb;border-radius:6px;height:18px;width:100%;max-width:500px">'
        '<div style="background:'+meta["color"]+';border-radius:6px;height:18px;width:'+str(min(w,100))+'%;'
        'display:flex;align-items:center;padding-left:8px;color:#fff;font-size:.75rem;font-weight:700">'
        +(f'{pct:.1f}%' if pct>10 else '')+'</div></div>'
        '<div style="font-size:.75rem;color:#94a3b8;margin-top:2px">'+meta["sublabel"]+'</div>'
        '</div>')

html = ("""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIA Report -- AUC-ROC + Category-Matched</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.header{background:linear-gradient(135deg,#1e3a5f,#6d28d9);color:#fff;padding:40px}
.header h1{font-size:1.6rem;font-weight:800;margin-bottom:6px}
.header p{opacity:.8;font-size:.92rem;margin-bottom:12px}
.badge{display:inline-block;background:rgba(255,255,255,.15);border-radius:9999px;padding:3px 12px;font-size:.73rem;margin-right:5px}
.container{max-width:1300px;margin:0 auto;padding:28px 20px}
.card{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:22px;margin-bottom:22px}
.card h2{font-size:1.02rem;font-weight:700;border-bottom:2px solid #e2e8f0;padding-bottom:9px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:22px}
.kpi{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center}
.kpi .v{font-size:1.4rem;font-weight:800}.kpi .l{font-size:.7rem;color:#64748b;margin-top:2px}
.red .v{color:#dc2626}.amber .v{color:#d97706}.green .v{color:#16a34a}.slate .v{color:#475569}.purple .v{color:#7c3aed}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:#f1f5f9;padding:7px 9px;text-align:left;font-weight:600;color:#374151;font-size:.75rem}
td{padding:7px 9px;border-bottom:1px solid #f1f5f9;vertical-align:middle}
tr:hover td{background:#f8fafc}
.footer{text-align:center;color:#94a3b8;font-size:.73rem;padding:20px}
</style></head><body>
<div class="header">
  <h1>Membership Inference Attack -- AUC-ROC + Category-Matched Non-Members</h1>
  <p>"""+str(N_SPLITS)+""" random splits, category-matched non-members, AUC-ROC across all thresholds (Shokri et al. standard)</p>
  <span class="badge">"""+str(NUM_MEMBERS)+""" Members / """+str(NUM_NONMEMBERS)+""" Non-Members per split</span>
  <span class="badge">Category-matched NM (same topic)</span>
  <span class="badge">"""+str(N_SPLITS)+"""-split mean +/- std</span>
  <span class="badge">heegyu/news-category-dataset</span>
  <span class="badge">Generated """+now+"""</span>
</div>
<div class="container">
  <div class="kpis">
    <div class="kpi red"><div class="v">"""+f"{agg['L1_exact']['auc_mean']:.3f}"+"""</div><div class="l">L1 AUC (exact)<br><small style="color:#9ca3af">trivial -- discard</small></div></div>
    <div class="kpi amber"><div class="v">"""+f"{agg['L2_paraphrase']['auc_mean']:.3f}"+"""</div><div class="l">L2 AUC (paraphrase)<br><small style="color:#9ca3af">realistic</small></div></div>
    <div class="kpi green"><div class="v">"""+f"{agg['L3_topic']['auc_mean']:.3f}"+"""</div><div class="l">L3 AUC (keyword)<br><small style="color:#9ca3af">lower bound</small></div></div>
    <div class="kpi purple"><div class="v">"""+f"{agg['L2_paraphrase']['adv_mean']*100:+.1f}%"+"""</div><div class="l">L2 Adv. Advantage</div></div>
    <div class="kpi slate"><div class="v">"""+str(N_SPLITS)+"""</div><div class="l">Random splits</div></div>
    <div class="kpi slate"><div class="v">cat.</div><div class="l">Non-member matching<br><small style="color:#9ca3af">same topic</small></div></div>
  </div>

  <div class="card">
    <h2>AUC-ROC Summary ("""+str(N_SPLITS)+"""-split mean +/- std) -- AUC=0.5 is random baseline</h2>
    """+auc_bars+"""
    <p style="font-size:.8rem;color:#64748b;margin-top:14px">
      <strong>L1 AUC ~1.0</strong> is expected and trivial -- exact query gives score=1.0 for all members.
      <strong>L2 (paraphrase)</strong> is the realistic threat model -- AUC above 0.6 indicates real privacy leakage.
      <strong>L3 (keyword)</strong> is the practical lower bound -- approaching 0.5 means the system is safe against weak attackers.
    </p>
  </div>

  """+level_cards+"""
</div>
<div class="footer">FedRAG MIA -- AUC-ROC -- Category-Matched NM -- """+str(N_SPLITS)+"""-Split -- """+now+"""</div>
</body></html>""")

Path("membership_inference_report.html").write_text(html, encoding="utf-8")
print("\nReport saved: membership_inference_report.html")
