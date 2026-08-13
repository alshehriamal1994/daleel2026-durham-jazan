# Ensemble-size curve for the Task 1 encoder (paper, Appendix A).
# Trains 8 seeds on the full training set, then reports dev macro-F1 for
# single seeds and for ensembles of 3, 5 and 8 seeds, averaged over random
# subsets of the trained seeds. Thresholds fixed at 0.5 so no tuning enters.
# Post-submission analysis.
import os
import json
import itertools
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
EPOCHS, BS, LR, MAXLEN = 8, 16, 2e-5, 384
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

train = [json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl", encoding="utf-8")]
dev_meta = {json.loads(l)["paragraph_id"]: json.loads(l)
            for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
gold = {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"])
        for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")}
dev_ids = sorted(gold)
tok = AutoTokenizer.from_pretrained(MODEL)


def train_one(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=6, problem_type="multi_label_classification").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((len(train) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    model.train()
    for _ in range(EPOCHS):
        idx = rng.permutation(len(train))
        for i in range(0, len(idx), BS):
            chunk = [train[j] for j in idx[i:i + BS]]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to(DEV)
            y = torch.zeros(len(chunk), 6, device=DEV)
            for k, r in enumerate(chunk):
                for l in r["labels"]:
                    y[k, L2I[l]] = 1.0
            enc["labels"] = y
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                loss = model(**enc).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(dev_ids), 32):
            ids = dev_ids[i:i + 32]
            enc = tok([dev_meta[j]["text"] for j in ids], truncation=True,
                      max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
            probs.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.vstack(probs)


def macro(P):
    f = []
    for k, lab in enumerate(LABELS):
        pred = P[:, k] >= 0.5
        g = np.array([lab in gold[i] for i in dev_ids])
        tp = np.sum(pred & g); fp = np.sum(pred & ~g); fn = np.sum(~pred & g)
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


P = {}
for s in SEEDS:
    P[s] = train_one(s)
    print(f"seed {s}: dev macro {macro(P[s]):.4f}", flush=True)
np.save(f"{W}/oof/ensemble_curve_probs.npy", np.stack([P[s] for s in SEEDS]))

print(f"\n{'seeds':>6}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}")
results = {}
for n in (1, 3, 5, 8):
    combos = list(itertools.combinations(SEEDS, n))
    if len(combos) > 40:
        combos = [combos[i] for i in np.random.RandomState(0).choice(len(combos), 40, replace=False)]
    vals = [macro(np.mean([P[s] for s in c], axis=0)) for c in combos]
    results[n] = [float(np.mean(vals)), float(np.std(vals))]
    print(f"{n:>6}{np.mean(vals):>9.4f}{np.std(vals):>8.4f}{np.min(vals):>8.4f}{np.max(vals):>8.4f}")
json.dump(results, open(f"{W}/oof/ensemble_curve.json", "w"), indent=2)
print("\nsaved oof/ensemble_curve.json")
