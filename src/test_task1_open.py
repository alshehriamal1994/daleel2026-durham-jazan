import json, numpy as np, torch
from collections import Counter
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16; SEEDS=[42,1,2,3,4]
DAPT=f"{W}/models/camelbert-dapt"
# robust median thresholds from the dev-#1 OPEN run (logs_open_sub.txt) — computed from train-only CV, unchanged for test
THS=np.array([0.425,0.275,0.275,0.375,0.125,0.35])
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt1=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
bt2=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt2.jsonl",encoding="utf-8")]
bt3=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt3.jsonl",encoding="utf-8")]
SYNTH=[json.loads(l) for l in open(f"{W}/data/synth_all.jsonl",encoding="utf-8")]
SX=[s["text"] for s in SYNTH]
SY=[[1. if l in set(s["labels"]) else 0. for l in L] for s in SYNTH]
test=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for x in set(r["labels"]): y[L2I[x]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32)
texts=[r["text"] for r in rows]
RARE=np.array([("ST" in r["labels"]) or ("CO" in r["labels"]) for r in rows])
def aug_train(tr):
    tt=[texts[i] for i in tr]; tl=[Y[i].tolist() for i in tr]
    tt+=[bt1[i]["text"] for i in tr]; tl+=[Y[i].tolist() for i in tr]   # 1x BT all
    for i in tr:                                                        # MODERATE rare aug (validated 3x level)
        if RARE[i]:
            for p in [bt2[i]["text"],bt3[i]["text"]]: tt.append(p); tl.append(Y[i].tolist())
    tt+=SX; tl+=SY   # OPEN: LLM synthetic data (training only)
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
allidx=np.arange(len(rows)); tp=np.zeros((len(test),6))
for sd in SEEDS: tp+=train_pred(sd,allidx,[r["text"] for r in test]); print(f"[test s{sd}] done",flush=True)
tp/=len(SEEDS); P=(tp>=THS).astype(int)
with open(f"{W}/preds/task1_test_open.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(test):
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":[L[k] for k in range(6) if P[i,k]],"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote task1_test_open.jsonl counts=",dict(Counter(L[k] for i in range(len(test)) for k in range(6) if P[i,k])),"empty=",int((P.sum(1)==0).sum()))
