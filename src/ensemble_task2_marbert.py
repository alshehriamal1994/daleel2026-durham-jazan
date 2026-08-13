import os
import json, numpy as np, sys
from collections import defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import task2_scoring as T2
LABELS=["AS","AN","ST","TE","CO","OT"]; W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
gold=defaultdict(list)
for r in rows:
    gold[r["paragraph_id"]]=T2.sort_spans([(l["label"],[l["start_offset"],l["end_offset"]]) for l in r["labels"]])
def load(n): return np.load(f"{W}/oof/task2_{n}_probs.npy",allow_pickle=True)
def decode(probs,offs,ths):
    spans=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                spans.append((LABELS[k],[offs[i][0],offs[j][1]])); i=j+1
            else: i+=1
    return spans
def evaluate(members,ths):
    arrs=[load(m) for m in members]; pred=defaultdict(list)
    for i,r in enumerate(rows):
        probs=np.mean([a[i][0] for a in arrs],axis=0); offs=arrs[0][i][1]
        pred[r["paragraph_id"]]=T2.sort_spans(decode(probs,offs,ths))
    return T2.score_per_span(gold,pred)[2]
def tune(members):
    ths=np.full(6,0.5); grid=np.arange(0.2,0.85,0.05)
    for _ in range(3):
        for k in range(6):
            best,bt=-1,ths[k]
            for t in grid:
                ths[k]=t; f=evaluate(members,ths)
                if f>best: best,bt=f,t
            ths[k]=bt
    return evaluate(members,ths),ths
cands={
 "mar_s42":["marbertv2_pw_s42"],
 "mar_2seed":["marbertv2_pw_s42","marbertv2_pw_s1"],
}
best=None
for n,m in cands.items():
    f,ths=tune(m); print(f"{n:14s} F1={f:.4f} ths={ths.round(2).tolist()}")
    if best is None or f>best[0]: best=(f,n,m,ths)
f,n,m,ths=best
print(f"\nBEST {n} F1={f:.4f}")
json.dump({"members":m,"ths":ths.tolist(),"model":"marbertv2","seeds":[int(x.split('_s')[1]) for x in m]},open(f"{W}/oof/task2_best.json","w"))
print("saved task2_best.json")
