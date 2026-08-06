import json, numpy as np, sys
from collections import defaultdict
sys.path.insert(0,"/home/amal/Desktop/daleel2026/src"); import task2_scoring as T2
LABELS=["AS","AN","ST","TE","CO","OT"]; W="/home/amal/Desktop/daleel2026"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
gold=defaultdict(list)
for r in rows:
    gold[r["paragraph_id"]]=T2.sort_spans([(l["label"],[l["start_offset"],l["end_offset"]]) for l in r["labels"]])

def load(name):
    arr=np.load(f"{W}/oof/task2_{name}_probs.npy",allow_pickle=True)
    return arr  # per example (probs[n,6], offsets)

def char_probs(arr,i):
    probs,offs=arr[i]; clen=len(rows[i]["text"])
    cp=np.zeros((clen,6),dtype=np.float32)
    for ti,(a,b) in enumerate(offs):
        if b>a: cp[a:b]=probs[ti]
    return cp

def decode_char(cp,ths):
    spans=[]
    for k in range(6):
        on=cp[:,k]>=ths[k]; i=0; n=len(on)
        while i<n:
            if on[i]:
                j=i
                while j+1<n and on[j+1]: j+=1
                spans.append((LABELS[k],[i,j+1])); i=j+1
            else: i+=1
    return spans

def evaluate(members,ths):
    arrs=[load(m) for m in members]
    pred=defaultdict(list)
    for i,r in enumerate(rows):
        cp=np.mean([char_probs(a,i) for a in arrs],axis=0)
        pred[r["paragraph_id"]]=T2.sort_spans(decode_char(cp,ths))
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

candidates={
 "camelbert_pw":["camelbert_pw"],
 "camelbert":["camelbert"],
 "cam+cam_pw":["camelbert","camelbert_pw"],
 "cam_pw+ara":["camelbert_pw","arabertv2"],
 "cam+ara":["camelbert","arabertv2"],
 "cam_pw+ara_pw":["camelbert_pw","arabertv2_pw"],
 "all4":["camelbert","camelbert_pw","arabertv2","arabertv2_pw"],
}
best=None
for n,m in candidates.items():
    f,ths=tune(m); print(f"{n:18s} F1={f:.4f}")
    if best is None or f>best[0]: best=(f,n,m,ths)
f,n,m,ths=best
print(f"\nBEST: {n}  F1={f:.4f}  ths={ths.round(2).tolist()}")
json.dump({"members":m,"ths":ths.tolist()},open(f"{W}/oof/task2_best.json","w"))
