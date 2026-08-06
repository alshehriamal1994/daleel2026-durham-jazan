import json, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; DAPT=f"{W}/models/camelbert-dapt"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
BT=[[json.loads(l) for l in open(f"{W}/data/train_task_1_{t}.jsonl",encoding="utf-8")] for t in ["bt","bt2","bt3"]]
def yv(r):
    y=[0.]*6
    for l in set(r["labels"]): y[L2I[l]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32); texts=[r["text"] for r in rows]
def evalcfg(ss,nbt):
    rng=np.random.RandomState(ss); idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    set_seed(100+ss); tok=AutoTokenizer.from_pretrained(DAPT)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    tt=[texts[i] for i in tr]; tl=[Y[i].tolist() for i in tr]
    for j in range(nbt):
        tt+=[BT[j][i]["text"] for i in tr]; tl+=[Y[i].tolist() for i in tr]
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dall=Dataset.from_dict({"text":[texts[i] for i in list(tr)+list(te)]}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(DAPT,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/b3",per_device_train_batch_size=16,per_device_eval_batch_size=32,
        learning_rate=2e-5,num_train_epochs=8,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=2000,seed=100+ss)
    t=Trainer(model=m,args=a,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dall).predictions)).numpy(); ntr=len(tr); ptr,pte=p[:ntr],p[ntr:]
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt=-1,0.5
        for x in grid:
            f=f1_score(Y[tr,k],(ptr[:,k]>=x).astype(int),zero_division=0)
            if f>b: b,bt=f,x
        th[k]=bt
    r=f1_score(Y[te],(pte>=th).astype(int),average="macro",zero_division=0); del m,t; torch.cuda.empty_cache(); return r
for name,n in [("DAPT+1xBT",1),("DAPT+3xBT",3)]:
    s=[evalcfg(sd,n) for sd in [11,22,33,44]]
    print(f"{name:12s} mean={np.mean(s):.3f} runs={[round(x,3) for x in s]}",flush=True)
