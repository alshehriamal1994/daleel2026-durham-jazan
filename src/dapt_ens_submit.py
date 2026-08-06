import json, numpy as np, torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16; FOLDS=5
DAPT=f"{W}/models/camelbert-dapt"
# (model_path, [seeds]) ; all trained with 1x back-translation augmentation
MODELS=[(DAPT,[42,1,2,3,4]), ("aubmindlab/bert-base-arabertv2",[42,1,2])]
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
dev=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for l in set(r["labels"]): y[L2I[l]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32)
texts=[r["text"] for r in rows]; bttexts=[r["text"] for r in bt]; types=[r["type"] for r in rows]
strat=[f"{types[i]}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i in range(len(rows))]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
def train_pred(model_path,seed,tr_idx,pred_texts):
    set_seed(seed); tok=AutoTokenizer.from_pretrained(model_path)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    tt=[texts[i] for i in tr_idx]+[bttexts[i] for i in tr_idx]
    tl=[Y[i].tolist() for i in tr_idx]+[Y[i].tolist() for i in tr_idx]
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dpr=Dataset.from_dict({"text":pred_texts}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/de2",per_device_train_batch_size=BS,per_device_eval_batch_size=32,
        learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=3000,seed=seed)
    t=Trainer(model=m,args=a,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dpr).predictions)).numpy(); del m,t; torch.cuda.empty_cache(); return p
# CV OOF (use 2 seeds per model for thresholds, cheaper)
oof=np.zeros((len(rows),6)); nseed=0
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
for mp,seeds in MODELS:
    for sd in seeds[:2]:
        nseed+=1
        for fold,(tr,va) in enumerate(skf.split(texts,strat)):
            oof[va]+=train_pred(mp,sd*100+fold,tr,[texts[i] for i in va])
        print(f"[cv {mp.split('/')[-1]} s{sd}] done",flush=True)
oof/=nseed
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
# full-data dev ensemble (all seeds)
allidx=np.arange(len(rows)); devp=np.zeros((len(dev),6)); tot=0
for mp,seeds in MODELS:
    for sd in seeds:
        devp+=train_pred(mp,sd,allidx,[r["text"] for r in dev]); tot+=1
        print(f"[dev {mp.split('/')[-1]} s{sd}] done",flush=True)
devp/=tot; P=(devp>=ths).astype(int)
with open(f"{W}/preds/task1_dev_daptens.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(dev):
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":[L[k] for k in range(6) if P[i,k]],"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote task1_dev_daptens.jsonl counts=",dict(Counter(L[k] for i in range(len(dev)) for k in range(6) if P[i,k])),"empty=",int((P.sum(1)==0).sum()))
