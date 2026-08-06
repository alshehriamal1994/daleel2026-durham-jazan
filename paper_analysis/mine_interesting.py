import json
from collections import defaultdict

W = "/home/amal/Desktop/daleel2026"

def load2(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[r["paragraph_id"]] = [(s["label"], s["start_offset"], s["end_offset"]) for s in r["labels"]]
    return out

def load1(p):
    return {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"]) for l in open(p, encoding="utf-8")}

g2 = load2(f"{W}/data/dev_task_2_ref.jsonl")
dev = {json.loads(l)["paragraph_id"]: json.loads(l) for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
cam = load2(f"{W}/preds/task2_dev_camelbert.jsonl")
mar = load2(f"{W}/preds/task2_dev_marbert.jsonl")

# ---- 1. overlapping gold spans (different labels, same region) ----
print("== OVERLAPPING GOLD SPANS ==")
n_overlap = 0
examples = []
for pid, spans in g2.items():
    for i in range(len(spans)):
        for j in range(i+1, len(spans)):
            l1, s1, e1 = spans[i]; l2, s2, e2 = spans[j]
            inter = min(e1, e2) - max(s1, s2)
            if inter > 10 and l1 != l2:
                n_overlap += 1
                examples.append((inter, pid, spans[i], spans[j], len(dev[pid]["text"])))
print("count:", n_overlap)
for ex in sorted(examples, key=lambda x: -x[0])[:5]:
    inter, pid, a, b, tl = ex
    print(f"pid {pid} ({dev[pid]['type']}, len {tl}): {a} overlaps {b} by {inter} chars")
    print("  A:", dev[pid]["text"][a[1]:a[2]][:100])
    print("  B:", dev[pid]["text"][b[1]:b[2]][:100])

# ---- 2. debate TE = restating the opponent ----
print("\n== DEBATE TE SPANS (opponent restatement candidates) ==")
kw = ["الفريق", "الخصم", "المعارض", "الموالاة", "المعارضة", "زملاؤنا", "يدّعون", "يدعون", "قالوا", "ذكروا", "حجة"]
hits = 0
for pid, spans in g2.items():
    if dev[pid]["type"] != "debate":
        continue
    for l, s, e in spans:
        if l != "TE":
            continue
        t = dev[pid]["text"][s:e]
        if any(k in t for k in kw) and e - s < 220:
            hits += 1
            if hits <= 5:
                print(f"pid {pid} [{s},{e}): {t[:180]}")

# ---- 3. routing evidence: debates where MARBERT >> CAMeLBERT ----
def score_para(pred_spans, gold_spans):
    if not pred_spans and not gold_spans: return 1.0
    cp = cr = 0.0
    for pl, ps, pe in pred_spans:
        L = pe - ps
        if L <= 0: continue
        for gl, gs, ge in gold_spans:
            if gl != pl: continue
            inter = max(0, min(pe, ge) - max(ps, gs))
            if inter: cp += inter / L; cr += inter / (ge - gs)
    P = cp / len(pred_spans) if pred_spans else 0
    R = cr / len(gold_spans) if gold_spans else 0
    return 2*P*R/(P+R) if P+R else 0.0

print("\n== ROUTING: debates where MARBERT beats CAMeLBERT most ==")
diffs = []
for pid in g2:
    if dev[pid]["type"] != "debate": continue
    fm = score_para(mar.get(pid, []), g2[pid])
    fc = score_para(cam.get(pid, []), g2[pid])
    diffs.append((fm - fc, pid, fm, fc, len(dev[pid]["text"])))
for d in sorted(diffs, reverse=True)[:5]:
    diff, pid, fm, fc, tl = d
    print(f"pid {pid}: MAR {fm:.3f} vs CAM {fc:.3f} (diff +{diff:.3f}, len {tl})")
    print("   text:", dev[pid]["text"][:130])

# ---- 4. T1 rare-class success: ST missed by DAPT+BT, caught by final ----
g1 = load1(f"{W}/data/dev_task_1_ref.jsonl")
daptbt = load1(f"{W}/preds/task1_dev_daptbt.jsonl")
rare = load1(f"{W}/preds/task1_dev_rare3x.jsonl")
opn = load1(f"{W}/preds/task1_dev_open.jsonl")
print("\n== ST SUCCESS: gold ST, DAPT+BT missed, rare3x/open caught ==")
n = 0
for pid in sorted(g1):
    if "ST" in g1[pid] and "ST" not in daptbt.get(pid, set()) and "ST" in opn.get(pid, set()):
        n += 1
        if n <= 4:
            t = dev[pid]
            # find the gold ST span text for display
            stspans = [s for s in g2.get(pid, []) if s[0] == "ST"]
            frag = dev[pid]["text"][stspans[0][1]:stspans[0][2]][:150] if stspans else t["text"][:150]
            print(f"pid {pid} ({t['type']}): gold {sorted(g1[pid])} | daptbt {sorted(daptbt.get(pid,set()))} | open {sorted(opn.get(pid,set()))}")
            print("   ST span:", frag)
print("total such paragraphs:", n)
