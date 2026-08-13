import os
import numpy as np, itertools, json
from sklearn.metrics import f1_score, precision_recall_fscore_support
LABELS=["AS","AN","ST","TE","CO","OT"]; W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tags=["arabertv2","marbertv2","camelbert"]
D={t:np.load(f"{W}/oof/task1_{t}.npz",allow_pickle=True) for t in tags}
Y=D["arabertv2"]["Y"]; types=D["arabertv2"]["types"]
oofs={t:D[t]["oof"] for t in tags}
grid=np.arange(0.05,0.95,0.025)
def tune(oof):
    ths=np.full(6,0.5)
    for k in range(6):
        best,bt=-1,0.5
        for t in grid:
            f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
            if f>best: best,bt=f,t
        ths[k]=bt
    P=(oof>=ths).astype(int)
    return f1_score(Y,P,average="macro",zero_division=0),ths,P
print("singles:")
for t in tags: print(f"  {t}: {tune(oofs[t])[0]:.4f}")
print("\nequal-weight subsets:")
best=None
for r in range(2,4):
    for combo in itertools.combinations(tags,r):
        oof=np.mean([oofs[t] for t in combo],axis=0)
        m,ths,P=tune(oof)
        print(f"  {'+'.join(combo)}: {m:.4f}")
        if best is None or m>best[0]: best=(m,combo,ths,oof)
m,combo,ths,oof=best
print(f"\nBEST: {'+'.join(combo)}  macro={m:.4f}")
P=(oof>=ths).astype(int)
pr,rc,f1,_=precision_recall_fscore_support(Y,P,average=None,zero_division=0)
for i,l in enumerate(LABELS): print(f"  {l}: P={pr[i]:.3f} R={rc[i]:.3f} F1={f1[i]:.3f}")
for dom in ["editorial","debate"]:
    msk=types==dom
    print(f"  [{dom}] macro={f1_score(Y[msk],P[msk],average='macro',zero_division=0):.4f}")
json.dump({"combo":list(combo),"ths":ths.tolist()},open(f"{W}/oof/task1_best.json","w"))
print("saved task1_best.json")
