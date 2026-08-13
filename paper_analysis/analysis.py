import os
import json, numpy as np
from collections import defaultdict

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
rng = np.random.RandomState(42)

def load1(p):
    return {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"])
            for l in open(p, encoding="utf-8")}

def load2(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[r["paragraph_id"]] = [(s["label"], s["start_offset"], s["end_offset"])
                                  for s in r["labels"]]
    return out

# ---------------- Task 1 ----------------
g1 = load1(f"{W}/data/dev_task_1_ref.jsonl")
ids1 = sorted(g1)

def t1_perclass(pred, ids):
    out = {}
    for lab in LABELS:
        tp = sum(1 for i in ids if lab in g1[i] and lab in pred.get(i, set()))
        fp = sum(1 for i in ids if lab not in g1[i] and lab in pred.get(i, set()))
        fn = sum(1 for i in ids if lab in g1[i] and lab not in pred.get(i, set()))
        out[lab] = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    out["macro"] = float(np.mean([out[l] for l in LABELS]))
    return out

systems1 = [
    ("pos_weight blend", f"{W}/preds/task1_dev_blend.jsonl"),
    ("DAPT+BT",          f"{W}/preds/task1_dev_daptbt.jsonl"),
    ("+rare-aug 3x",     f"{W}/preds/task1_dev_rare3x.jsonl"),
    ("+LLM-synth (Open)",f"{W}/preds/task1_dev_open.jsonl"),
]
print("== T1 per-class dev F1 ==")
print(f"{'system':<20}" + "".join(f"{l:>7}" for l in LABELS) + f"{'macro':>8}")
preds1 = {}
for name, path in systems1:
    p = load1(path)
    preds1[name] = p
    r = t1_perclass(p, ids1)
    print(f"{name:<20}" + "".join(f"{r[l]:7.3f}" for l in LABELS) + f"{r['macro']:8.4f}")

# bootstrap CI for T1 macro (final closed dev system)
best1 = preds1["+rare-aug 3x"]
boots = []
for _ in range(2000):
    sample = [ids1[i] for i in rng.randint(0, len(ids1), len(ids1))]
    boots.append(t1_perclass(best1, sample)["macro"])
b = np.array(boots)
print(f"\nT1 macro bootstrap (rare3x, dev): mean {b.mean():.4f}, 95% CI [{np.percentile(b,2.5):.4f}, {np.percentile(b,97.5):.4f}], half-width {(np.percentile(b,97.5)-np.percentile(b,2.5))/2:.4f}")

# ---------------- Task 2 ----------------
g2 = load2(f"{W}/data/dev_task_2_ref.jsonl")
ids2 = sorted(g2)
dev_type = {json.loads(l)["paragraph_id"]: json.loads(l)["type"]
            for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}

def t2_score(pred, ids):
    tp = sum(len(pred.get(i, [])) for i in ids)
    tg = sum(len(g2[i]) for i in ids)
    cp = cr = 0.0
    for i in ids:
        for pl, ps, pe in pred.get(i, []):
            plen = pe - ps
            if plen <= 0: continue
            for gl, gs, ge in g2[i]:
                if gl != pl: continue
                inter = max(0, min(pe, ge) - max(ps, gs))
                if inter == 0: continue
                cp += inter / plen
                cr += inter / (ge - gs)
    P = cp/tp if tp else 0; R = cr/tg if tg else 0
    return 2*P*R/(P+R) if P+R else 0.0

# routed dev configuration (cam editorials / mar debates) — the basis of the
# paired compaction bootstrap reported in the paper (raw 0.6918 -> pp3 0.7186)
p2 = load2(f"{W}/preds/task2_dev_routed.jsonl")
print(f"\n== T2 verify: task2_dev_routed.jsonl overall {t2_score(p2, ids2):.4f} ==")

def compact(pred):
    out = {}
    for i, spans in pred.items():
        G = 400 if dev_type[i] == "editorial" else 5
        by = defaultdict(list)
        for l, s, e in spans: by[l].append((s, e))
        res = []
        for l, sp in by.items():
            sp.sort()
            merged = [list(sp[0])]
            for s, e in sp[1:]:
                if s - merged[-1][1] <= G: merged[-1][1] = max(merged[-1][1], e)
                else: merged.append([s, e])
            res += [(l, s, e) for s, e in merged if e - s >= 25]
        out[i] = res
    return out

p2c = compact(p2)
raw, comp = t2_score(p2, ids2), t2_score(p2c, ids2)
print(f"T2 compaction on dev: raw {raw:.4f} -> compacted {comp:.4f} (delta {comp-raw:+.4f})")

# bootstrap: T2 CI + paired compaction delta
b2, bd = [], []
for _ in range(2000):
    sample = [ids2[i] for i in rng.randint(0, len(ids2), len(ids2))]
    a = t2_score(p2, sample); c = t2_score(p2c, sample)
    b2.append(a); bd.append(c - a)
b2, bd = np.array(b2), np.array(bd)
print(f"T2 overlap-F1 bootstrap: 95% CI half-width {(np.percentile(b2,97.5)-np.percentile(b2,2.5))/2:.4f}")
print(f"Compaction paired delta: mean {bd.mean():+.4f}, 95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}], P(delta<=0) = {(bd<=0).mean():.4f}")
