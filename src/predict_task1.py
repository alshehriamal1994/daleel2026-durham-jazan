import json, numpy as np, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16
MODELS={"arabertv2":"aubmindlab/bert-base-arabertv2",
        "camelbert":"CAMeL-Lab/bert-base-arabic-camelbert-mix"}
best=json.load(open(f"{W}/oof/task1_best.json")); combo=best["combo"]; ths=np.array(best["ths"])

rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
dev=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]

dev_probs=[]
for tag in combo:
    mname=MODELS[tag]; tok=AutoTokenizer.from_pretrained(mname)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    dtr=Dataset.from_dict({"text":[r["text"] for r in rows],"labels":[Y[i].tolist() for i in range(len(rows))]}).map(enc,batched=True)
    dde=Dataset.from_dict({"text":[r["text"] for r in dev]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(mname,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/full_{tag}",per_device_train_batch_size=BS,
        per_device_eval_batch_size=32,learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,
        weight_decay=0.01,fp16=True,report_to=[],save_strategy="no",eval_strategy="no",logging_steps=100)
    tr=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    tr.train()
    p=torch.sigmoid(torch.tensor(tr.predict(dde).predictions)).numpy()
    dev_probs.append(p); del model,tr; torch.cuda.empty_cache()
    print(f"[{tag}] dev predicted",flush=True)

avg=np.mean(dev_probs,axis=0); P=(avg>=ths).astype(int)
with open(f"{W}/preds/task1_dev.jsonl","w",encoding="utf-8") as f:
    for i,r in enumerate(dev):
        labs=[LABELS[k] for k in range(6) if P[i,k]==1]
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":labs,"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote preds/task1_dev.jsonl  n=",len(dev))
import collections; c=collections.Counter()
for i in range(len(dev)):
    for k in range(6):
        if P[i,k]: c[LABELS[k]]+=1
print("dev label counts:",dict(c)," empty:",int((P.sum(1)==0).sum()))
