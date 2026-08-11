# Error taxonomy behind the paper's "Error Taxonomy" appendix.
# Task 2: classifies gold spans (covered/partial/missed) and predicted spans
# (clean/partial/label-confusion/hallucinated) for the routed Closed dev
# configuration, raw vs pp3-compacted. Task 1: per-label FN/FP and the labels
# predicted instead on missed paragraphs, for the final Closed dev system.
import json
from collections import defaultdict

W = "/home/amal/Desktop/daleel2026"
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]


def load2(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[r["paragraph_id"]] = [(s["label"], s["start_offset"], s["end_offset"])
                                  for s in r["labels"]]
    return out


g2 = load2(f"{W}/data/dev_task_2_ref.jsonl")
ids2 = sorted(g2)
dev_type = {json.loads(l)["paragraph_id"]: json.loads(l)["type"]
            for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}


def compact(pred):
    out = {}
    for i, spans in pred.items():
        G = 400 if dev_type[i] == "editorial" else 5
        by = defaultdict(list)
        for l, s, e in spans:
            by[l].append((s, e))
        res = []
        for l, sp in by.items():
            sp.sort()
            merged = [list(sp[0])]
            for s, e in sp[1:]:
                if s - merged[-1][1] <= G:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            res += [(l, s, e) for s, e in merged if e - s >= 25]
        out[i] = res
    return out


def taxonomy(pred):
    gold_cov = defaultdict(int)
    pred_cls = defaultdict(int)
    frags = []
    for i in ids2:
        P = pred.get(i, [])
        G = g2[i]
        for gl, gs, ge in G:
            cov = 0
            nfrag = 0
            for pl, ps, pe in P:
                if pl != gl:
                    continue
                ov = max(0, min(pe, ge) - max(ps, gs))
                if ov > 0:
                    cov += ov
                    nfrag += 1
            frac = cov / (ge - gs)
            gold_cov["covered" if frac >= 0.8 else ("partial" if frac > 0 else "missed")] += 1
            if frac > 0:
                frags.append(nfrag)
        for pl, ps, pe in P:
            plen = pe - ps
            same = sum(max(0, min(pe, ge) - max(ps, gs)) for gl, gs, ge in G if gl == pl)
            if same / plen >= 0.8:
                pred_cls["clean"] += 1
            elif same > 0:
                pred_cls["partial"] += 1
            else:
                other = sum(max(0, min(pe, ge) - max(ps, gs)) for gl, gs, ge in G if gl != pl)
                pred_cls["label-confusion" if other / plen >= 0.5 else "hallucinated"] += 1
    return gold_cov, pred_cls, sum(frags) / len(frags)


pred = load2(f"{W}/preds/task2_dev_routed.jsonl")
for tag, p in [("raw", pred), ("compacted", compact(pred))]:
    gc, pc, frag = taxonomy(p)
    print(f"{tag}: gold {dict(gc)} | pred n={sum(pc.values())} {dict(pc)} | "
          f"pred spans per matched gold: {frag:.2f}")

g1 = {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"])
      for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")}
p1 = {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"])
      for l in open(f"{W}/preds/task1_dev_rare3x.jsonl", encoding="utf-8")}
print("\nTask 1 confusion (final Closed dev system):")
for lab in LABELS:
    fn = [i for i in g1 if lab in g1[i] and lab not in p1.get(i, set())]
    fp = [i for i in g1 if lab not in g1[i] and lab in p1.get(i, set())]
    sub = defaultdict(int)
    for i in fn:
        for x in p1.get(i, set()) - g1[i]:
            sub[x] += 1
    support = sum(1 for i in g1 if lab in g1[i])
    print(f"  {lab}: support {support}, FN {len(fn)}, FP {len(fp)}, "
          f"predicted instead {dict(sub)}")
