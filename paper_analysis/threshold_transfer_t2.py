# Task 2 replication of the controlled threshold experiment (paper, Appendix C).
# Span-decoding thresholds are recalibrated on part of the out-of-fold predictions
# and evaluated on the remainder, under the official overlap F1.
#
# The comparison baseline is the DEVELOPMENT-PHASE threshold set actually used in
# the submitted Closed decode. It was fitted a month earlier on pre-DAPT-v2
# cross-validation probabilities, so it is blind to the predictions under test.
# For transparency the script also reports the variant that takes the
# evaluation-phase recalibrated set as baseline: that set was itself fitted on all
# 829 of these out-of-fold predictions, so it sits at its own optimum on both
# halves of every split and overstates the held-out loss. The paper reports the
# blind-baseline, per-genre-ascent rows.
#
# Two decoders and two ascent structures are crossed so a reader can see how much
# each protocol choice moves the result. "prod" is decode() from
# src/t2_closed_recal.py, the procedure under audit; "paper" is an earlier variant
# that did not re-check span offsets when extending a token run.
# Post-submission analysis, CPU only.
import json, pickle, sys
import numpy as np

W = "/home/amal/Desktop/daleel2026"
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
MINLEN = 25
GRID = np.round(np.arange(0.10, 0.901, 0.05), 3)      # audited grid, 17 points
NG_ = len(GRID)

oof = pickle.load(open(f"{W}/oof/t2_recal_oof.pkl", "rb"))
rows = [json.loads(l) for l in open(f"{W}/oof/t2_gold_all.jsonl", encoding="utf-8")]
GOLD = [[(s["label"], s["start_offset"], s["end_offset"]) for s in r["labels"]] for r in rows]
GENRE = np.array([r["type"] for r in rows])
N = len(rows)
IS_ED = GENRE == "editorial"


def spans_prod(probs, offs, k, t, gap):
    """decode() from src/t2_closed_recal.py, single label."""
    on = probs[:, k] >= t
    raw, i = [], 0
    while i < len(on):
        if on[i] and offs[i][1] > offs[i][0]:
            j = i
            while j + 1 < len(on) and on[j + 1] and offs[j + 1][1] > offs[j + 1][0]:
                j += 1
            raw.append([offs[i][0], offs[j][1]])
            i = j + 1
        else:
            i += 1
    out = []
    if raw:
        cur = raw[0]
        for s, e in raw[1:]:
            if s - cur[1] <= gap:
                cur[1] = max(cur[1], e)
            else:
                out.append(cur); cur = [s, e]
        out.append(cur)
    return [(s, e) for s, e in out if e - s >= MINLEN]


def spans_paper(probs, offs, k, t, gap):
    """spans_for() from paper/analysis/threshold_transfer_t2.py."""
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
    if out:
        out.sort()
        m = [out[0]]
        for s, e in out[1:]:
            if s - m[-1][1] <= gap:
                m[-1][1] = max(m[-1][1], e)
            else:
                m.append([s, e])
        out = m
    return [(s, e) for s, e in out if e - s >= MINLEN]


def build_tables(spanfn):
    NP = np.zeros((N, 6, NG_)); CP = np.zeros((N, 6, NG_))
    CR = np.zeros((N, 6, NG_)); NGold = np.zeros((N, 6))
    for i in range(N):
        ed = IS_ED[i]
        probs = oof["cam"][i] if ed else oof["mar"][i]
        offs = oof["cam_offs"][i] if ed else oof["mar_offs"][i]
        gap = 400 if ed else 5
        for k, lab in enumerate(LABELS):
            g = [(s, e) for l, s, e in GOLD[i] if l == lab]
            NGold[i, k] = len(g)
            for ti, t in enumerate(GRID):
                sp = spanfn(probs, offs, k, t, gap)
                NP[i, k, ti] = len(sp)
                for ps, pe in sp:
                    for gs, ge in g:
                        ov = max(0, min(pe, ge) - max(ps, gs))
                        if ov:
                            CP[i, k, ti] += ov / (pe - ps)
                            CR[i, k, ti] += ov / (ge - gs)
    return NP, CP, CR, NGold


T_INDEX = {round(float(t), 3): i for i, t in enumerate(GRID)}
def snap(ths):
    return [T_INDEX[min(GRID, key=lambda g: abs(g - t))] for t in ths]

BLIND_ED = snap([0.6, 0.4, 0.55, 0.5, 0.3, 0.7])
BLIND_DB = snap([0.45, 0.75, 0.8, 0.75, 0.8, 0.55])
RECAL_ED = snap(json.load(open(f"{W}/oof/t2_recal_ths.json"))["ths_editorial"])
RECAL_DB = snap(json.load(open(f"{W}/oof/t2_recal_ths.json"))["ths_debate"])


class Agg:
    """Subset-aggregated credit tables; scoring is then a handful of lookups."""
    __slots__ = ("np_e", "cp_e", "cr_e", "ng_e", "np_d", "cp_d", "cr_d", "ng_d")
    def __init__(self, NP, CP, CR, NGold, idx):
        e = idx[IS_ED[idx]]; d = idx[~IS_ED[idx]]
        self.np_e, self.cp_e, self.cr_e = NP[e].sum(0), CP[e].sum(0), CR[e].sum(0)
        self.ng_e = NGold[e].sum(0)
        self.np_d, self.cp_d, self.cr_d = NP[d].sum(0), CP[d].sum(0), CR[d].sum(0)
        self.ng_d = NGold[d].sum(0)


K = np.arange(6)
def f1(npv, cpv, crv, ngv, ti):
    tp = npv[K, ti].sum(); cp = cpv[K, ti].sum(); cr = crv[K, ti].sum(); tg = ngv.sum()
    if tp == 0 or tg == 0: return 0.0
    P = cp / tp; R = cr / tg
    return (2 * P * R / (P + R)) if P + R else 0.0

def f_ed(a, ti_ed): return f1(a.np_e, a.cp_e, a.cr_e, a.ng_e, ti_ed)
def f_db(a, ti_db): return f1(a.np_d, a.cp_d, a.cr_d, a.ng_d, ti_db)

def f_pooled(a, ti_ed, ti_db):
    tp = a.np_e[K, ti_ed].sum() + a.np_d[K, ti_db].sum()
    cp = a.cp_e[K, ti_ed].sum() + a.cp_d[K, ti_db].sum()
    cr = a.cr_e[K, ti_ed].sum() + a.cr_d[K, ti_db].sum()
    tg = a.ng_e.sum() + a.ng_d.sum()
    if tp == 0 or tg == 0: return 0.0
    P = cp / tp; R = cr / tg
    return (2 * P * R / (P + R)) if P + R else 0.0


def tune_pergenre(a, base_ed, base_db, rounds=2):
    """Faithful to src/t2_closed_recal.py Phase A2: editorial coords ascend on
    editorial F1, debate coords on debate F1, strict improvement only."""
    te, td = list(base_ed), list(base_db)
    for _ in range(rounds):
        for k in range(6):
            best = f_ed(a, te); bv = te[k]
            for v in range(NG_):
                cand = list(te); cand[k] = v
                s = f_ed(a, cand)
                if s > best: best, bv = s, v
            te[k] = bv
    for _ in range(rounds):
        for k in range(6):
            best = f_db(a, td); bv = td[k]
            for v in range(NG_):
                cand = list(td); cand[k] = v
                s = f_db(a, cand)
                if s > best: best, bv = s, v
            td[k] = bv
    return te, td


def tune_joint(a, base_ed, base_db, rounds=2):
    """Paper-script tune(): alternating ed/db coords on the pooled objective."""
    te, td = list(base_ed), list(base_db)
    for _ in range(rounds):
        for k in range(6):
            best = f_pooled(a, te, td); bv = te[k]
            for v in range(NG_):
                cand = list(te); cand[k] = v
                s = f_pooled(a, cand, td)
                if s > best: best, bv = s, v
            te[k] = bv
            best = f_pooled(a, te, td); bv = td[k]
            for v in range(NG_):
                cand = list(td); cand[k] = v
                s = f_pooled(a, te, cand)
                if s > best: best, bv = s, v
            td[k] = bv
    return te, td


def experiment(NP, CP, CR, NGold, base_ed, base_db, tuner, tune_size, repeats=200):
    rng = np.random.RandomState(0)
    ins, held = [], []
    for _ in range(repeats):
        perm = rng.permutation(N)
        A, B = perm[:tune_size], perm[tune_size:]
        aA = Agg(NP, CP, CR, NGold, A); aB = Agg(NP, CP, CR, NGold, B)
        te, td = tuner(aA, base_ed, base_db)
        ins.append(f_pooled(aA, te, td) - f_pooled(aA, base_ed, base_db))
        held.append(f_pooled(aB, te, td) - f_pooled(aB, base_ed, base_db))
    ins = np.array(ins); held = np.array(held)
    return dict(tune_size=tune_size, n_held=N - tune_size, repeats=repeats,
                in_sample=round(float(ins.mean()), 5), in_sd=round(float(ins.std()), 5),
                held_out=round(float(held.mean()), 5), held_sd=round(float(held.std()), 5),
                pct_non_positive=round(100 * float(np.mean(held <= 0)), 1))


if __name__ == "__main__":
    out = {}
    print("building PRODUCTION-decoder credit tables ...", flush=True)
    P_ = build_tables(spans_prod)
    allidx = np.arange(N)
    aAll = Agg(*P_, allidx)
    chk = dict(blind_ed=round(f_ed(aAll, BLIND_ED), 4), blind_db=round(f_db(aAll, BLIND_DB), 4),
               blind_pooled=round(f_pooled(aAll, BLIND_ED, BLIND_DB), 4),
               recal_ed=round(f_ed(aAll, RECAL_ED), 4), recal_db=round(f_db(aAll, RECAL_DB), 4),
               recal_pooled=round(f_pooled(aAll, RECAL_ED, RECAL_DB), 4))
    print("SANITY (production decoder):", chk, flush=True)
    out["sanity_production"] = chk

    print("building PAPER-decoder credit tables ...", flush=True)
    Q_ = build_tables(spans_paper)
    aAllQ = Agg(*Q_, allidx)
    chk2 = dict(blind_ed=round(f_ed(aAllQ, BLIND_ED), 4), blind_db=round(f_db(aAllQ, BLIND_DB), 4),
                blind_pooled=round(f_pooled(aAllQ, BLIND_ED, BLIND_DB), 4),
                recal_pooled=round(f_pooled(aAllQ, RECAL_ED, RECAL_DB), 4))
    print("SANITY (paper decoder):", chk2, flush=True)
    out["sanity_paper"] = chk2

    SIZES = [207, 414, 621, 746]
    for name, tables, base, tuner in [
        ("prod_pergenre_blind", P_, (BLIND_ED, BLIND_DB), tune_pergenre),
        ("prod_joint_blind",    P_, (BLIND_ED, BLIND_DB), tune_joint),
        ("prod_pergenre_contaminated", P_, (RECAL_ED, RECAL_DB), tune_pergenre),
        ("prod_joint_contaminated",    P_, (RECAL_ED, RECAL_DB), tune_joint),
        ("paper_joint_blind",   Q_, (BLIND_ED, BLIND_DB), tune_joint),
        ("paper_joint_contaminated", Q_, (RECAL_ED, RECAL_DB), tune_joint),
    ]:
        out[name] = []
        for ts in SIZES:
            r = experiment(*tables, base[0], base[1], tuner, ts)
            out[name].append(r)
            print(f"{name:30s} n={ts:4d}  in {r['in_sample']:+.5f}  held {r['held_out']:+.5f} "
                  f"(sd {r['held_sd']:.4f})  non-pos {r['pct_non_positive']:.1f}%", flush=True)
    json.dump(out, open(f"{W}/oof/t2_threshold_transfer_results.json", "w"), indent=1)
    print("done")
