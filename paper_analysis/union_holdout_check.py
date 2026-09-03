# Post-submission check of the three-seed Qwen3-32B union (paper, Appendix A, Table "union").
# Recomputes, on the controlled Task 1 holdout used for the LLM experiments, the macro-F1 of
# routing editorials to (a) single seeds, (b) the union, majority vote, and intersection of
# every three of the four retained seeds, with debates always from the encoder ensemble.
# Needs the organisers' train/dev files under DALEEL_ROOT/data (not redistributed here).
import os, json, itertools
import numpy as np
from sklearn.metrics import f1_score

W = os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = ["AS", "AN", "ST", "TE", "CO", "OT"]
rows = [json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl", encoding="utf-8")] + \
       [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")]
va_idx = np.load(f"{W}/oof/t1_llm_holdout_idx.npy")
va = [rows[i] for i in va_idx]
Y = np.array([[1.0 if l in set(r["labels"]) else 0.0 for l in L] for r in va])
ed = np.array([r["type"] == "editorial" for r in va])
mac = lambda y, p: f1_score(y, p, average="macro", zero_division=0)
seeds = {k: np.load(f"{W}/oof/t1_llm_holdout_pred_{k}.npy") for k in ["s1", "s2", "s3", "s4"]}
enc = (np.load(f"{W}/oof/t1_recal_oof_closed.npy")[va_idx] >= np.array(json.load(open(next(q for q in (f"{W}/oof/t1_recal_ths_closed.json", f"{W}/configs/t1_recal_ths_closed.json") if os.path.exists(q))))["ths"])).astype(float)
route = lambda P: np.where(ed[:, None], P, enc)
print(f"holdout: {len(va)} paragraphs ({ed.sum()} editorial)")
print(f"encoder ensemble on both genres: {mac(Y, enc):.3f}")
single = [mac(Y, route(P)) for P in seeds.values()]
print(f"single seed, routed: {min(single):.3f} to {max(single):.3f}, mean {np.mean(single):.3f}")
for name, rule in [("union", lambda ps: np.max(ps, 0)), ("majority", lambda ps: (np.sum(ps, 0) >= 2).astype(float)), ("intersection", lambda ps: np.min(ps, 0))]:
    vals = [mac(Y, route(rule([seeds[k] for k in c]))) for c in itertools.combinations(seeds, 3)]
    print(f"{name} of three seeds, routed: {min(vals):.3f} to {max(vals):.3f}")
