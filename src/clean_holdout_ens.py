import os
import json, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
texts=[r["text"] for r in rows]
SPECS=[("CAMeL-Lab/bert-base-arabic-camelbert-mix",1),("CAMeL-Lab/bert-base-arabic-camelbert-mix",2),
       ("CAMeL-Lab/bert-base-arabic-camelbert-mix",3),("aubmindlab/bert-base-arabertv2",1)]
def probs_for(model_name,seed,tr_idx,te_idx):
    set_seed(seed); tok=AutoTokenizer.from_pretrained(model_name)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr_idx],"labels":[Y[i].tolist() for i in tr_idx]}).map(enc,batched=True)
    dall=Dataset.from_dict({"text":[texts[i] for i in list(tr_idx)+list(te_idx)]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/che_{seed}",per_device_train_batch_size=16,
        per_device_eval_batch_size=32,learning_rate=2e-5,num_train_epochs=8,warmup_ratio=0.1,
        weight_decay=0.01,fp16=True,report_to=[],save_strategy="no",eval_strategy="no",logging_steps=500,seed=seed)
    t=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train()
    p=torch.sigmoid(torch.tensor(t.predict(dall).predictions)).numpy()
    del model,t; torch.cuda.empty_cache()
    ntr=len(tr_idx); return p[:ntr],p[ntr:]
def run(seed):
    rng=np.random.RandomState(seed); idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    ptr=np.zeros((len(tr),6)); pte=np.zeros((217,6))
    for mn,sd in SPECS:
        a,b=probs_for(mn,sd*10+seed,tr,te); ptr+=a; pte+=b
    ptr/=len(SPECS); pte/=len(SPECS)
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b_,bt=-1,0.5
        for tt in grid:
            f=f1_score(Y[tr,k],(ptr[:,k]>=tt).astype(int),zero_division=0)
            if f>b_: b_,bt=f,tt
        th[k]=bt
    return f1_score(Y[te],(pte>=th).astype(int),average="macro",zero_division=0)
res=[run(s) for s in [11,22,33]]
print("CLEAN ensemble holdout (4 models, train 395 -> unseen 217):",[round(r,3) for r in res]," mean=",round(np.mean(res),3))
