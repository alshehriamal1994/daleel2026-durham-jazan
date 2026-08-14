# Controlled test of the paper's central claim (Appendix C).
# Per-class thresholds are tuned on one half of the out-of-fold predictions
# and evaluated on the held-out half. The in-sample gain is what threshold
# tuning appears to deliver; the held-out gain is what actually transfers.
# Post-submission analysis, CPU only.
import os
import json
import numpy as np

W = os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
BASE = [.425, .25, .2, .3, .125, .275]   # development-phase thresholds
GRID = np.arange(0.05, 0.96, 0.025)
RNG = np.random.RandomState(0)
REPEATS = 500

P = np.load(f"{W}/oof/t1_recal_oof_closed.npy")
rows = [json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl", encoding="utf-8")]
rows += [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")]
Y = np.zeros((len(rows), 6))
for i, r in enumerate(rows):
    for l in r["labels"]:
        Y[i, LABELS.index(l)] = 1


def class_f1(p, g, t):
    pred = p >= t
    tp = np.sum(pred & g)
    fp = np.sum(pred & ~g)
    fn = np.sum(~pred & g)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def macro(idx, ths):
    return float(np.mean([class_f1(P[idx, k], Y[idx, k] > 0, ths[k]) for k in range(6)]))


def tune(idx):
    # macro-F1 is separable across labels, so each threshold is chosen independently
    return [max(GRID, key=lambda t: class_f1(P[idx, k], Y[idx, k] > 0, t)) for k in range(6)]


print(f"{'tune n':>8}{'in-sample':>12}{'held-out':>11}{'held-out<=0':>13}")
for frac in (0.25, 0.5, 0.75, 0.9):
    promised, delivered = [], []
    for _ in range(REPEATS):
        perm = RNG.permutation(len(P))
        cut = int(len(P) * frac)
        A, B = perm[:cut], perm[cut:]
        ths = tune(A)
        promised.append(macro(A, ths) - macro(A, BASE))
        delivered.append(macro(B, ths) - macro(B, BASE))
    promised, delivered = np.array(promised), np.array(delivered)
    print(f"{int(len(P)*frac):>8}{promised.mean():>+12.4f}{delivered.mean():>+11.4f}"
          f"{100*np.mean(delivered <= 0):>12.0f}%")
    print(f"{'':8}{'sd ' + format(promised.std(), '.4f'):>12}"
          f"{'sd ' + format(delivered.std(), '.4f'):>11}")
