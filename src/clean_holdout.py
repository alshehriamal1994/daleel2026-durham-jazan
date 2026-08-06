import json, numpy as np, torch, sys
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
texts=[r["text"] for r in rows]; types=np.array([r["type"] for r in rows])
MODEL="CAMeL-Lab/bert-base-arabic-camelbert-mix"; tok=AutoTokenizer.from_pretrained(MODEL)
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
def run(seed):
    set_seed(seed); rng=np.random.RandomState(seed)
    idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr],"labels":[Y[i].tolist() for i in tr]}).map(enc,batched=True)
    dte=Dataset.from_dict({"text":[texts[i] for i in te]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/ho_{seed}",per_device_train_batch_size=16,
        per_device_eval_batch_size=32,learning_rate=2e-5,num_train_epochs=8,warmup_ratio=0.1,
        weight_decay=0.01,fp16=True,report_to=[],save_strategy="no",eval_strategy="no",logging_steps=500,seed=seed)
    t=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train()
    prob=torch.sigmoid(torch.tensor(t.predict(dte).predictions)).numpy()
    # thresholds tuned ON TRAIN ONLY (predict train probs)
    ptr=torch.sigmoid(torch.tensor(t.predict(Dataset.from_dict({"text":[texts[i] for i in tr]}).map(enc,batched=True)).predictions)).numpy()
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt=-1,0.5
        for tt in grid:
            f=f1_score(Y[tr,k],(ptr[:,k]>=tt).astype(int),zero_division=0)
            if f>b: b,bt=f,tt
        th[k]=bt
    P=(prob>=th).astype(int)
    macro=f1_score(Y[te],P,average="macro",zero_division=0)
    del model,t; torch.cuda.empty_cache()
    return macro
res=[run(s) for s in [11,22,33]]
print("CLEAN holdout macro (train 395 -> predict unseen 217):", [round(r,3) for r in res], " mean=",round(np.mean(res),3))
