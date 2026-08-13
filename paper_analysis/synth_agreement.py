# Blind re-annotation check of the Task 1 synthetic data (paper, Appendix B).
# A second, newer LLM (Claude Fable 5) re-annotated all 291 synthetic
# paragraphs given only the official label definitions, blind to the
# generator's labels. This script scores agreement between those blind
# annotations (synth_blind_annotations.jsonl; "id" indexes the concatenation
# of data/synth_all.jsonl followed by data/synth_v2/t1_batch_agent.jsonl)
# and the generator's labels.
import os
import json
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]

rows = []
for f in ["data/synth_all.jsonl", "data/synth_v2/t1_batch_agent.jsonl"]:
    for l in open(f"{W}/{f}", encoding="utf-8"):
        rows.append(json.loads(l))
gold = {i: set(r["labels"]) for i, r in enumerate(rows)}

ann = {}
for l in open(f"{W}/paper_analysis/synth_blind_annotations.jsonl", encoding="utf-8"):
    r = json.loads(l)
    ann[r["id"]] = set(r["labels"])

ids = sorted(gold)
assert set(ids) == set(ann), "annotation ids must cover all paragraphs"

exact = sum(1 for i in ids if gold[i] == ann[i]) / len(ids)
print(f"paragraphs: {len(ids)}; exact label-set agreement: {exact:.3f}")

tp_all = fp_all = fn_all = 0
print(f"{'label':<6}{'agree%':>8}{'kappa':>8}{'P':>7}{'R':>7}{'F1':>7}")
for lab in LABELS:
    g = np.array([lab in gold[i] for i in ids])
    a = np.array([lab in ann[i] for i in ids])
    po = (g == a).mean()
    pe = g.mean() * a.mean() + (1 - g.mean()) * (1 - a.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    tp = int((g & a).sum()); fp = int((~g & a).sum()); fn = int((g & ~a).sum())
    tp_all += tp; fp_all += fp; fn_all += fn
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    print(f"{lab:<6}{po:>8.3f}{kappa:>8.3f}{P:>7.3f}{R:>7.3f}{F:>7.3f}")
P = tp_all / (tp_all + fp_all); R = tp_all / (tp_all + fn_all)
print(f"micro-F1 (blind annotator vs generator): {2*P*R/(P+R):.3f}")
