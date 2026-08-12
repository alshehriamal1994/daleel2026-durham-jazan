# Controlled genre-transfer experiment (paper, Appendix D).
# Trains size-matched Task 1 classifiers on editorials only, on debates only,
# and on a half-and-half mixture, then evaluates each on the editorial and
# debate halves of the development set. Thresholds are fixed at 0.5 so that
# no tuning enters the comparison. Post-submission analysis.
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

W = "/home/amal/Desktop/daleel2026"
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
EPOCHS, BS, LR, MAXLEN = 8, 16, 2e-5, 384
SEEDS = [42, 1, 2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

train = [json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl", encoding="utf-8")]
ed_tr = [r for r in train if r["type"] == "editorial"]
db_tr = [r for r in train if r["type"] != "editorial"]
N = min(len(ed_tr), len(db_tr))          # size-matched conditions

dev_meta = {json.loads(l)["paragraph_id"]: json.loads(l)
            for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
dev_gold = {json.loads(l)["paragraph_id"]: set(json.loads(l)["labels"])
            for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")}
dev_ids = sorted(dev_gold)
tok = AutoTokenizer.from_pretrained(MODEL)


def batches(rows, bs, shuffle, rng=None):
    idx = np.arange(len(rows))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(DEV)
        y = torch.zeros(len(chunk), 6, device=DEV)
        for k, r in enumerate(chunk):
            for l in r["labels"]:
                y[k, L2I[l]] = 1.0
        enc["labels"] = y
        yield enc


def run(rows, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=6, problem_type="multi_label_classification").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((len(rows) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    model.train()
    for _ in range(EPOCHS):
        for enc in batches(rows, BS, True, rng):
            with torch.amp.autocast("cuda", enabled=(DEV == "cuda")):
                loss = model(**enc).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
    model.eval()
    pred = {}
    with torch.no_grad():
        for i in range(0, len(dev_ids), 32):
            ids = dev_ids[i:i + 32]
            enc = tok([dev_meta[j]["text"] for j in ids], truncation=True,
                      max_length=MAXLEN, padding=True, return_tensors="pt").to(DEV)
            p = torch.sigmoid(model(**enc).logits).float().cpu().numpy()
            for j, row in zip(ids, p):
                pred[j] = {LABELS[k] for k in range(6) if row[k] >= 0.5}
    del model
    torch.cuda.empty_cache()
    return pred


def per_class(pred, ids):
    out = {}
    for lab in LABELS:
        tp = sum(1 for i in ids if lab in dev_gold[i] and lab in pred[i])
        fp = sum(1 for i in ids if lab not in dev_gold[i] and lab in pred[i])
        fn = sum(1 for i in ids if lab in dev_gold[i] and lab not in pred[i])
        out[lab] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    out["macro"] = float(np.mean([out[l] for l in LABELS]))
    return out


ed_dev = [i for i in dev_ids if dev_meta[i]["type"] == "editorial"]
db_dev = [i for i in dev_ids if dev_meta[i]["type"] != "editorial"]
print(f"size-matched training pools: {N} paragraphs per condition "
      f"(available {len(ed_tr)} editorial, {len(db_tr)} debate); device {DEV}", flush=True)

results = {}
for cond in ["editorials-only", "debates-only", "mixed"]:
    accum = {"ed": [], "db": [], "ed_OT": [], "db_OT": [], "ed_CO": []}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        if cond == "editorials-only":
            rows = [ed_tr[i] for i in rng.choice(len(ed_tr), N, replace=False)]
        elif cond == "debates-only":
            rows = [db_tr[i] for i in rng.choice(len(db_tr), N, replace=False)]
        else:
            h = N // 2
            rows = ([ed_tr[i] for i in rng.choice(len(ed_tr), h, replace=False)] +
                    [db_tr[i] for i in rng.choice(len(db_tr), N - h, replace=False)])
        pred = run(rows, seed)
        e, d = per_class(pred, ed_dev), per_class(pred, db_dev)
        accum["ed"].append(e["macro"]); accum["db"].append(d["macro"])
        accum["ed_OT"].append(e["OT"]); accum["db_OT"].append(d["OT"])
        accum["ed_CO"].append(e["CO"])
        print(f"  {cond:16} seed {seed}: ed macro {e['macro']:.3f}  db macro {d['macro']:.3f}"
              f"  ed OT {e['OT']:.3f}  db OT {d['OT']:.3f}", flush=True)
    results[cond] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accum.items()}

print("\n=== MEAN OVER 3 SEEDS (dev, thresholds fixed at 0.5) ===")
print(f"{'train on':18}{'ed macro':>10}{'db macro':>10}{'ed OT':>9}{'db OT':>9}{'ed CO':>9}")
for cond, r in results.items():
    print(f"{cond:18}{r['ed'][0]:10.3f}{r['db'][0]:10.3f}{r['ed_OT'][0]:9.3f}"
          f"{r['db_OT'][0]:9.3f}{r['ed_CO'][0]:9.3f}")
json.dump(results, open(f"{W}/oof/genre_transfer_results.json", "w"), indent=2)
print("\nsaved oof/genre_transfer_results.json")
