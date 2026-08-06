import json, numpy as np, torch
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
texts=[r["text"] for r in rows]
def evalcfg(model_name,epochs,split_seed):
    rng=np.random.RandomState(split_seed); idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    set_seed(100+split_seed); tok=AutoTokenizer.from_pretrained(model_name)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr],"labels":[Y[i].tolist() for i in tr]}).map(enc,batched=True)
    dall=Dataset.from_dict({"text":[texts[i] for i in list(tr)+list(te)]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/ms",per_device_train_batch_size=16,per_device_eval_batch_size=32,
        learning_rate=2e-5,num_train_epochs=epochs,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=1000,seed=100+split_seed)
    t=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dall).predictions)).numpy()
    ntr=len(tr); ptr,pte=p[:ntr],p[ntr:]
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt=-1,0.5
        for tt in grid:
            f=f1_score(Y[tr,k],(ptr[:,k]>=tt).astype(int),zero_division=0)
            if f>b: b,bt=f,tt
        th[k]=bt
    del model,t; torch.cuda.empty_cache()
    return f1_score(Y[te],(pte>=th).astype(int),average="macro",zero_division=0)
CFGS={
 "camelbert-mix ep8":("CAMeL-Lab/bert-base-arabic-camelbert-mix",8),
 "camelbert-mix ep5":("CAMeL-Lab/bert-base-arabic-camelbert-mix",5),
 "camelbert-da  ep6":("CAMeL-Lab/bert-base-arabic-camelbert-da",6),
 "ARBERTv2      ep6":("UBC-NLP/ARBERTv2",6),
 "marbert-mix?  ep6":("UBC-NLP/MARBERTv2",6),
}
SEEDS=[11,22,33,44]
for name,(mn,ep) in CFGS.items():
    try:
        s=[evalcfg(mn,ep,sd) for sd in SEEDS]
        print(f"{name:20s} mean={np.mean(s):.3f}  std={np.std(s):.3f}  runs={[round(x,3) for x in s]}",flush=True)
    except Exception as e:
        print(f"{name:20s} FAILED {e}",flush=True)
