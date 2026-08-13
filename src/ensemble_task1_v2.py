import os
import numpy as np, glob, os, json
from sklearn.metrics import f1_score, precision_recall_fscore_support
LABELS=["AS","AN","ST","TE","CO","OT"]; W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
files=sorted(glob.glob(f"{W}/oof/task1_*.npz"))
D=[np.load(f,allow_pickle=True) for f in files]
Y=D[0]["Y"]; types=D[0]["types"]; oofs=[d["oof"] for d in D]
names=[os.path.basename(f)[:-4] for f in files]
grid=np.arange(0.05,0.95,0.025)
def tune(oof):
    ths=np.full(6,0.5)
    for k in range(6):
        best,bt=-1,0.5
        for t in grid:
            f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
            if f>best: best,bt=f,t
        ths[k]=bt
    return f1_score(Y,(oof>=ths).astype(int),average="macro",zero_division=0),ths
print("pool singles:")
for n,o in zip(names,oofs): print(f"  {n}: {tune(o)[0]:.4f}")
# Caruana greedy ensemble selection with replacement
ens=[]; best_macro=-1
# init with best single
inits=sorted(range(len(oofs)),key=lambda i:tune(oofs[i])[0],reverse=True)[:1]
ens=list(inits)
for _ in range(30):
    cur=np.mean([oofs[i] for i in ens],axis=0); cm=tune(cur)[0]
    cand=[]
    for i in range(len(oofs)):
        avg=np.mean([oofs[j] for j in ens+[i]],axis=0)
        cand.append((tune(avg)[0],i))
    bm,bi=max(cand)
    if bm<=cm+1e-5: break
    ens.append(bi); best_macro=bm
from collections import Counter
oof=np.mean([oofs[i] for i in ens],axis=0); macro,ths=tune(oof)
print("\nselected members (weights):")
for i,c in Counter(ens).items(): print(f"  {names[i]} x{c}")
print(f"ENSEMBLE macro={macro:.4f}")
P=(oof>=ths).astype(int)
pr,rc,f1,_=precision_recall_fscore_support(Y,P,average=None,zero_division=0)
for i,l in enumerate(LABELS): print(f"  {l}: P={pr[i]:.3f} R={rc[i]:.3f} F1={f1[i]:.3f}")
for dom in ["editorial","debate"]:
    m=types==dom; print(f"  [{dom}] {f1_score(Y[m],P[m],average='macro',zero_division=0):.4f}")
cfg={"members":[names[i] for i in ens],"ths":ths.tolist()}
json.dump(cfg,open(f"{W}/oof/task1_best.json","w"))
print("saved task1_best.json:",Counter([names[i] for i in ens]))
