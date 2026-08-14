# Task 2 replication of the controlled threshold experiment (paper, Appendix C).
# Span-decoding thresholds are recalibrated on one half of the out-of-fold
# predictions and evaluated on the held-out half under the official overlap F1.
#
# The overlap metric decomposes by label (a predicted span can only earn credit
# from same-label gold spans), so we precompute, for every paragraph, label and
# candidate threshold, the predicted-span count and the precision and recall
# credit. Scoring any subset at any threshold vector is then a sum over that
# table, which makes hundreds of splits cheap and exact.
# Post-submission analysis, CPU only.
import os
import json
import pickle
from collections import defaultdict
import numpy as np

W = os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
GRID = np.round(np.arange(0.20, 0.951, 0.05), 3)
REPEATS = 200
MINLEN = 25
RNG = np.random.RandomState(0)

oof = pickle.load(open(f"{W}/oof/t2_recal_oof.pkl", "rb"))
rows = [json.loads(l) for l in open(f"{W}/oof/t2_gold_all.jsonl", encoding="utf-8")]
GOLD = [[(s["label"], s["start_offset"], s["end_offset"]) for s in r["labels"]] for r in rows]
GENRE = np.array([r["type"] for r in rows])
CFG = json.load(open(f"{W}/configs/t2_recal_ths.json"))
N = len(rows)


def spans_for(probs, offs, k, t, gap):
    on = probs[:, k] >= t
    out, i = [], 0
    while i < len(on):
        if on[i] and offs[i][1] > offs[i][0]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            out.append([offs[i][0], offs[j][1]])
            i = j + 1
        else:
            i += 1
    if gap is not None and out:
        out.sort()
        m = [out[0]]
        for s, e in out[1:]:
            if s - m[-1][1] <= gap:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        out = m
    return [(s, e) for s, e in out if e - s >= MINLEN]


# precompute credit tables
NP = np.zeros((N, 6, len(GRID)))
CP = np.zeros((N, 6, len(GRID)))
CR = np.zeros((N, 6, len(GRID)))
NG = np.zeros((N, 6))
for i in range(N):
    ed = GENRE[i] == "editorial"
    probs = oof["cam"][i] if ed else oof["mar"][i]
    offs = oof["cam_offs"][i] if ed else oof["mar_offs"][i]
    gap = 400 if ed else 5
    for k, lab in enumerate(LABELS):
        g = [(s, e) for l, s, e in GOLD[i] if l == lab]
        NG[i, k] = len(g)
        for ti, t in enumerate(GRID):
            sp = spans_for(probs, offs, k, t, gap)
            NP[i, k, ti] = len(sp)
            for ps, pe in sp:
                for gs, ge in g:
                    ov = max(0, min(pe, ge) - max(ps, gs))
                    if ov:
                        CP[i, k, ti] += ov / (pe - ps)
                        CR[i, k, ti] += ov / (ge - gs)
    if (i + 1) % 200 == 0:
        print(f"  precomputed {i+1}/{N}", flush=True)

T_INDEX = {round(float(t), 3): i for i, t in enumerate(GRID)}


def snap(ths):
    return [T_INDEX[min(GRID, key=lambda g: abs(g - t))] for t in ths]


def score(idx, ti_ed, ti_db):
    ed = idx[GENRE[idx] == "editorial"]
    db = idx[GENRE[idx] != "editorial"]
    tp = cp = cr = tg = 0.0
    for sub, ti in ((ed, ti_ed), (db, ti_db)):
        if len(sub) == 0:
            continue
        for k in range(6):
            tp += NP[sub, k, ti[k]].sum()
            cp += CP[sub, k, ti[k]].sum()
            cr += CR[sub, k, ti[k]].sum()
            tg += NG[sub, k].sum()
    P = cp / tp if tp else 0.0
    R = cr / tg if tg else 0.0
    return (2 * P * R / (P + R)) if P + R else 0.0


def tune(idx, base_ed, base_db):
    ti_ed, ti_db = list(base_ed), list(base_db)
    for _ in range(2):                        # two passes of coordinate ascent
        for k in range(6):
            ti_ed[k] = max(range(len(GRID)),
                           key=lambda t: score(idx, ti_ed[:k] + [t] + ti_ed[k+1:], ti_db))
            ti_db[k] = max(range(len(GRID)),
                           key=lambda t: score(idx, ti_ed, ti_db[:k] + [t] + ti_db[k+1:]))
    return ti_ed, ti_db


BASE_ED, BASE_DB = snap(CFG["ths_editorial"]), snap(CFG["ths_debate"])
allidx = np.arange(N)
print(f"\nfull-set score at the submitted thresholds: {score(allidx, BASE_ED, BASE_DB):.4f}")

promised, delivered = [], []
for r in range(REPEATS):
    perm = RNG.permutation(N)
    A, B = perm[:N // 2], perm[N // 2:]
    te, td = tune(A, BASE_ED, BASE_DB)
    promised.append(score(A, te, td) - score(A, BASE_ED, BASE_DB))
    delivered.append(score(B, te, td) - score(B, BASE_ED, BASE_DB))
    if (r + 1) % 50 == 0:
        print(f"  {r+1}/{REPEATS}: in-sample {np.mean(promised):+.4f}  "
              f"held-out {np.mean(delivered):+.4f}", flush=True)

promised, delivered = np.array(promised), np.array(delivered)
print(f"\nTask 2 threshold recalibration, {REPEATS} random halves of {N} paragraphs")
print(f"  in-sample gain {promised.mean():+.4f} (sd {promised.std():.4f})")
print(f"  held-out  gain {delivered.mean():+.4f} (sd {delivered.std():.4f})")
print(f"  held-out <= 0 in {100*np.mean(delivered <= 0):.0f}% of splits")
