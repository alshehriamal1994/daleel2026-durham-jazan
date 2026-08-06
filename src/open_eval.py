import json, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; DAPT=f"{W}/models/camelbert-dapt"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt1=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
bt2=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt2.jsonl",encoding="utf-8")]
bt3=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt3.jsonl",encoding="utf-8")]
synth=[json.loads(l) for l in open(f"{W}/data/synth_all.jsonl",encoding="utf-8")]
def yv(labs):
    y=[0.]*6
    for x in set(labs): y[L2I[x]]=1.
    return y
Y=np.array([yv(r["labels"]) for r in rows],dtype=np.float32); texts=[r["text"] for r in rows]
RARE=np.array([("ST" in r["labels"]) or ("CO" in r["labels"]) for r in rows])
SX=[s["text"] for s in synth]; SY=[yv(s["labels"]) for s in synth]
def build(tr,mode):
    tt=[texts[i] for i in tr]; tl=[Y[i].tolist() for i in tr]
    tt+=[bt1[i]["text"] for i in tr]; tl+=[Y[i].tolist() for i in tr]      # 1x BT all
    if "rare" in mode:
        for i in tr:
            if RARE[i]:
                for p in [bt2[i]["text"],bt3[i]["text"]]: tt.append(p); tl.append(Y[i].tolist())
    if "synth" in mode:
        tt+=SX; tl+=[y for y in SY]
    return tt,tl
def evalcfg(ss,mode):
    rng=np.random.RandomState(ss); idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    set_seed(100+ss); tok=AutoTokenizer.from_pretrained(DAPT)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    tt,tl=build(tr,mode)
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dall=Dataset.from_dict({"text":[texts[i] for i in list(tr)+list(te)]}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(DAPT,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/oe",per_device_train_batch_size=16,per_device_eval_batch_size=32,
        learning_rate=2e-5,num_train_epochs=8,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=3000,seed=100+ss)
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
for name,mode in [("DAPT+BT (base)","bt"),("DAPT+BT+rareaug","bt_rare"),("DAPT+BT+rare+SYNTH (Open)","bt_rare_synth")]:
    s=[evalcfg(sd,mode) for sd in [11,22,33,44]]
    print(f"{name:28s} mean={np.mean(s):.3f} runs={[round(x,3) for x in s]}",flush=True)
