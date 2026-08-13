import os
import json, numpy as np, torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; DAPT=f"{W}/models/camelbert-dapt"
EPOCHS=8; LR=2e-5; BS=16; FOLDS=5; SEEDS=[42,1,2]
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
dev=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for l in set(r["labels"]): y[L2I[l]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32)
texts=[r["text"] for r in rows]; bttexts=[r["text"] for r in bt]; types=[r["type"] for r in rows]
tok=AutoTokenizer.from_pretrained(DAPT)
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
def train_on(tr_idx,seed,extra_pred_texts):
    set_seed(seed)
    tt=[texts[i] for i in tr_idx]+[bttexts[i] for i in tr_idx]
    tl=[Y[i].tolist() for i in tr_idx]+[Y[i].tolist() for i in tr_idx]
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dpr=Dataset.from_dict({"text":extra_pred_texts}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(DAPT,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/dbt",per_device_train_batch_size=BS,per_device_eval_batch_size=32,
        learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=2000,seed=seed)
    t=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dpr).predictions)).numpy()
    del model,t; torch.cuda.empty_cache(); return p

# ---- CV OOF (3 seeds averaged) for robust thresholds ----
strat=[f"{types[i]}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i in range(len(rows))]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
oof=np.zeros((len(rows),6),dtype=np.float32)
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    acc=np.zeros((len(va),6))
    for sd in SEEDS: acc+=train_on(tr,sd,[texts[i] for i in va])
    oof[va]=acc/len(SEEDS); print(f"[cv fold {fold}] done",flush=True)
# robust thresholds (median over resamples)
grid=np.arange(0.05,0.95,0.025); rng=np.random.RandomState(5); allths=[]
for s in range(30):
    idx=rng.permutation(len(rows))[:306]; th=np.full(6,0.5)
    for k in range(6):
        b,bt_=-1,0.5
        for t in grid:
            f=f1_score(Y[idx,k],(oof[idx,k]>=t).astype(int),zero_division=0)
            if f>b: b,bt_=f,t
        th[k]=bt_
    allths.append(th)
ths=np.median(allths,axis=0)
insample=f1_score(Y,(oof>=ths).astype(int),average="macro",zero_division=0)
print("CV in-sample macro (DAPT+BT 3seed):",round(insample,4),"ths=",ths.round(3).tolist())

# ---- full-data ensemble -> dev ----
allidx=np.arange(len(rows)); devprob=np.zeros((len(dev),6))
for sd in SEEDS: devprob+=train_on(allidx,sd,[r["text"] for r in dev])
devprob/=len(SEEDS)
P=(devprob>=ths).astype(int)
with open(f"{W}/preds/task1_dev_daptbt.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(dev):
        labs=[L[k] for k in range(6) if P[i,k]==1]
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":labs,"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote task1_dev_daptbt.jsonl  counts=",dict(Counter(L[k] for i in range(len(dev)) for k in range(6) if P[i,k])),"empty=",int((P.sum(1)==0).sum()))
