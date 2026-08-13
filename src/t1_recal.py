import os
import json, sys, numpy as np, torch
from collections import Counter
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
TRACK=sys.argv[1] if len(sys.argv)>1 else "closed"   # closed | open
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16; FOLDS=5
CV_SEEDS=[42,1,2]; FULL_SEEDS=[42,1,2,3,4,5,6,7]
DAPT=f"{W}/models/camelbert-dapt-v2"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl",encoding="utf-8")]
bt1=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt.jsonl",encoding="utf-8")]
bt2=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt2.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt2.jsonl",encoding="utf-8")]
bt3=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt3.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt3.jsonl",encoding="utf-8")]
SX,SY=[],[]
if TRACK=="open":
    SYNTH=[json.loads(l) for l in open(f"{W}/data/synth_all.jsonl",encoding="utf-8")]+\
          [json.loads(l) for l in open(f"{W}/data/synth_v2/t1_batch_agent.jsonl",encoding="utf-8")]
    SX=[s["text"] for s in SYNTH]; SY=[[1. if l in set(s["labels"]) else 0. for l in L] for s in SYNTH]
test=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for x in set(r["labels"]): y[L2I[x]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32)
texts=[r["text"] for r in rows]; types=[r["type"] for r in rows]
RARE=np.array([("ST" in r["labels"]) or ("CO" in r["labels"]) for r in rows])
def aug_train(tr):
    tt=[texts[i] for i in tr]; tl=[Y[i].tolist() for i in tr]
    tt+=[bt1[i]["text"] for i in tr]; tl+=[Y[i].tolist() for i in tr]
    for i in tr:
        if RARE[i]:
            for p in [bt2[i]["text"],bt3[i]["text"]]: tt.append(p); tl.append(Y[i].tolist())
    tt+=SX; tl+=list(SY)
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
# CV OOF for thresholds under the FINAL regime (829 rows, DAPT-v2)
strat=[f"{types[i]}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i in range(len(rows))]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
from sklearn.model_selection import StratifiedKFold
oof=np.zeros((len(rows),6)); skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    acc=np.zeros((len(va),6))
    for sd in CV_SEEDS: acc+=train_pred(sd*100+fold,tr,[texts[i] for i in va])
    oof[va]=acc/len(CV_SEEDS); print(f"[cv fold {fold}] done",flush=True)
np.save(f"{W}/oof/t1_recal_oof_{TRACK}.npy",oof)
grid=np.arange(0.05,0.95,0.025); rng=np.random.RandomState(5); allths=[]
for s in range(30):
    idx=rng.permutation(len(rows))[:int(len(rows)*0.5)]; th=np.full(6,0.5)
    for k in range(6):
        b,bt_=-1,0.5
        for t in grid:
            f=f1_score(Y[idx,k],(oof[idx,k]>=t).astype(int),zero_division=0)
            if f>b: b,bt_=f,t
        th[k]=bt_
    allths.append(th)
ths=np.median(allths,axis=0)
OLD={"closed":np.array([0.425,0.25,0.2,0.3,0.125,0.275]),"open":np.array([0.425,0.275,0.275,0.375,0.125,0.35])}[TRACK]
mac_new=f1_score(Y,(oof>=ths).astype(int),average="macro",zero_division=0)
mac_old=f1_score(Y,(oof>=OLD).astype(int),average="macro",zero_division=0)
print(f"OOF macro: NEW ths {mac_new:.4f} vs OLD ths {mac_old:.4f}; new={ths.round(3).tolist()}",flush=True)
if mac_new<=mac_old: ths=OLD; print("keeping OLD thresholds (no OOF gain)",flush=True)
json.dump({"ths":list(map(float,ths)),"oof_macro_new":float(mac_new),"oof_macro_old":float(mac_old)},open(f"{W}/oof/t1_recal_ths_{TRACK}.json","w"))
# full retrain, save TEST probs, predict
allidx=np.arange(len(rows)); tp=np.zeros((len(test),6))
for sd in FULL_SEEDS: tp+=train_pred(sd,allidx,[r["text"] for r in test]); print(f"[test s{sd}] done",flush=True)
tp/=len(FULL_SEEDS); np.save(f"{W}/oof/t1_test_probs_{TRACK}_v5.npy",tp)
P=(tp>=ths).astype(int)
with open(f"{W}/preds/task1_test_{TRACK}_v5.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(test):
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":[L[k] for k in range(6) if P[i,k]],"type":r["type"]},ensure_ascii=False)+"\n")
print(f"wrote task1_test_{TRACK}_v5.jsonl counts=",dict(Counter(L[k] for i in range(len(test)) for k in range(6) if P[i,k])),"empty=",int((P.sum(1)==0).sum()),flush=True)
