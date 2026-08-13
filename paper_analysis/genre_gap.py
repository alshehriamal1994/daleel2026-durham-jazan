# Why are editorials harder? (paper, Appendix D)
# Pairs each label's dev F1 in each genre with the editorial share of that
# label's training spans, using the routed Closed dev configuration after
# compaction. Post-submission analysis, CPU only.
import os
import json
from collections import defaultdict, Counter
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]


def load2(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[r["paragraph_id"]] = [(s["label"], s["start_offset"], s["end_offset"])
                                  for s in r["labels"]]
    return out


gold = load2(f"{W}/data/dev_task_2_ref.jsonl")
meta = {json.loads(l)["paragraph_id"]: json.loads(l)
        for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
genre = {i: meta[i]["type"] for i in meta}
pred = load2(f"{W}/preds/task2_dev_routed.jsonl")


def compact(p):
    out = {}
    for i, spans in p.items():
        G = 400 if genre[i] == "editorial" else 5
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


def f1(p, ids, lab):
    tp = sum(len([s for s in p.get(i, []) if s[0] == lab]) for i in ids)
    tg = sum(len([s for s in gold[i] if s[0] == lab]) for i in ids)
    cp = cr = 0.0
    for i in ids:
        for pl, ps, pe in p.get(i, []):
            if pl != lab:
                continue
            for gl, gs, ge in gold[i]:
                if gl != lab:
                    continue
                ov = max(0, min(pe, ge) - max(ps, gs))
                if ov:
                    cp += ov / (pe - ps)
                    cr += ov / (ge - gs)
    P = cp / tp if tp else 0.0
    R = cr / tg if tg else 0.0
    return (2 * P * R / (P + R)) if P + R else 0.0


pc = compact(pred)
ed = [i for i in gold if genre[i] == "editorial"]
db = [i for i in gold if genre[i] != "editorial"]

# editorial share of each label's training spans
share = {}
counts = defaultdict(Counter)
for l in open(f"{W}/data/train_task_2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    for s in r["labels"]:
        counts[s["label"]][r["type"]] += 1
for lab in LABELS:
    e = counts[lab].get("editorial", 0)
    d = sum(v for k, v in counts[lab].items() if k != "editorial")
    share[lab] = 100 * e / (e + d) if e + d else 0.0

print(f"{'label':6}{'ed F1':>8}{'db F1':>8}{'diff':>8}{'% ed train':>12}")
diffs, shares = [], []
for lab in LABELS:
    fe, fd = f1(pc, ed, lab), f1(pc, db, lab)
    diffs.append(fe - fd)
    shares.append(share[lab])
    print(f"{lab:6}{fe:8.3f}{fd:8.3f}{fe-fd:+8.3f}{share[lab]:11.0f}%")
print(f"\nPearson r (editorial train share vs editorial advantage) = "
      f"{np.corrcoef(shares, diffs)[0, 1]:.2f}")

# robustness: does the pattern hold for each encoder family on its own?
print("\nreplication across encoder families (raw decodes):")
for name, f in [("CAMeLBERT-mix", "task2_dev_camelbert.jsonl"),
                ("MARBERTv2", "task2_dev_marbert.jsonl"),
                ("routed", "task2_dev_routed.jsonl")]:
    p = load2(f"{W}/preds/{f}")
    d = [f1(p, ed, lab) - f1(p, db, lab) for lab in LABELS]
    print(f"  {name:14} r = {np.corrcoef([share[l] for l in LABELS], d)[0, 1]:.2f}"
          f"   OT diff = {d[LABELS.index('OT')]:+.2f}")

# structural contrast
for name, ids in [("editorial", ed), ("debate", db)]:
    spans = [s for i in ids for s in gold[i]]
    lens = [e - s for _, s, e in spans]
    mix = Counter(l for l, _, _ in spans)
    tot = sum(mix.values())
    print(f"{name}: {len(spans)/len(ids):.2f} spans/paragraph, "
          f"mean span {np.mean(lens):.0f} chars, "
          f"mix " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in mix.most_common(3)))
