import os
import json, numpy as np, torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16; FOLDS=5; SEEDS=[42,1,2,3,4]
DAPT=f"{W}/models/camelbert-dapt"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt1=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
bt2=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt2.jsonl",encoding="utf-8")]
bt3=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt3.jsonl",encoding="utf-8")]
extra=json.load(open(f"{W}/data/rare_bt_extra.json"))  # idx->[5 paraphrases]
dev=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for x in set(r["labels"]): y[L2I[x]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32)
texts=[r["text"] for r in rows]; types=[r["type"] for r in rows]
RARE=np.array([("ST" in r["labels"]) or ("CO" in r["labels"]) for r in rows])
def aug_train(tr):
    tt=[texts[i] for i in tr]; tl=[Y[i].tolist() for i in tr]
    tt+=[bt1[i]["text"] for i in tr]; tl+=[Y[i].tolist() for i in tr]   # 1x BT all
    for i in tr:                                                        # MODERATE rare aug (validated 3x level)
        if RARE[i]:
            for p in [bt2[i]["text"],bt3[i]["text"]]: tt.append(p); tl.append(Y[i].tolist())
    return tt,tl
tok=AutoTokenizer.from_pretrained(DAPT)
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
def train_pred(seed,tr_idx,pred_texts):
    set_seed(seed); tt,tl=aug_train(tr_idx)
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dpr=Dataset.from_dict({"text":pred_texts}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(DAPT,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/drs",per_device_train_batch_size=BS,per_device_eval_batch_size=32,
        learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=3000,seed=seed)
    t=Trainer(model=m,args=a,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dpr).predictions)).numpy(); del m,t; torch.cuda.empty_cache(); return p
# CV OOF (3 seeds) for robust thresholds
strat=[f"{types[i]}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i in range(len(rows))]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
oof=np.zeros((len(rows),6)); skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    acc=np.zeros((len(va),6))
    for sd in [42,1,2]: acc+=train_pred(sd*100+fold,tr,[texts[i] for i in va])
    oof[va]=acc/3; print(f"[cv fold {fold}] done",flush=True)
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
print("CV in-sample macro:",round(f1_score(Y,(oof>=ths).astype(int),average="macro",zero_division=0),4),"ths=",ths.round(3).tolist())
# full-data dev (5 seeds)
allidx=np.arange(len(rows)); devp=np.zeros((len(dev),6))
for sd in SEEDS: devp+=train_pred(sd,allidx,[r["text"] for r in dev]); print(f"[dev s{sd}] done",flush=True)
devp/=len(SEEDS); P=(devp>=ths).astype(int)
with open(f"{W}/preds/task1_dev_rare3x.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(dev):
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":[L[k] for k in range(6) if P[i,k]],"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote task1_dev_rare3x.jsonl counts=",dict(Counter(L[k] for i in range(len(dev)) for k in range(6) if P[i,k])),"empty=",int((P.sum(1)==0).sum()))
