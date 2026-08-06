import json, sys, os, numpy as np, torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

LABELS = ["AS","AN","ST","TE","CO","OT"]
L2I = {l:i for i,l in enumerate(LABELS)}
MODEL = sys.argv[1] if len(sys.argv)>1 else "aubmindlab/bert-base-arabertv2"
TAG   = sys.argv[2] if len(sys.argv)>2 else "arabertv2"
MAXLEN=384; EPOCHS=8; FOLDS=5; LR=2e-5; BS=16
W="/home/amal/Desktop/daleel2026"

rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
texts=[r["text"] for r in rows]
types=[r["type"] for r in rows]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0

# stratify to spread rare classes + domain
strat=[f"{r['type']}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i,r in enumerate(rows)]
# collapse singleton strata
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]

tok=AutoTokenizer.from_pretrained(MODEL)
def enc(batch): return tok(batch["text"],truncation=True,max_length=MAXLEN)

oof=np.zeros((len(rows),6),dtype=np.float32)
skf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr],"labels":[Y[i].tolist() for i in tr]}).map(enc,batched=True)
    dva=Dataset.from_dict({"text":[texts[i] for i in va],"labels":[Y[i].tolist() for i in va]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        MODEL,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/tmp_{TAG}_{fold}",
        per_device_train_batch_size=BS,per_device_eval_batch_size=32,
        learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,
        weight_decay=0.01,fp16=True,report_to=[],logging_steps=50,
        save_strategy="no",eval_strategy="no",dataloader_num_workers=4)
    tr_=Trainer(model=model,args=args,train_dataset=dtr,
        data_collator=DataCollatorWithPadding(tok))
    tr_.train()
    pred=tr_.predict(dva).predictions
    oof[va]=torch.sigmoid(torch.tensor(pred)).numpy()
    del model,tr_; torch.cuda.empty_cache()
    print(f"[fold {fold}] done",flush=True)

np.savez(f"{W}/oof/task1_{TAG}.npz",oof=oof,Y=Y,types=np.array(types))

def macro_at(th):
    P=(oof>=th).astype(int)
    return f1_score(Y,P,average="macro",zero_division=0)

# default 0.5
print(f"\n=== {TAG} ===")
print("macro@0.5 :",round(macro_at(0.5),4))
# per-class threshold tuning on OOF
ths=np.full(6,0.5)
grid=np.arange(0.05,0.95,0.025)
for k in range(6):
    best,bt=-1,0.5
    for t in grid:
        p=(oof[:,k]>=t).astype(int)
        f=f1_score(Y[:,k],p,zero_division=0)
        if f>best: best,bt=f,t
    ths[k]=bt
P=(oof>=ths).astype(int)
print("tuned thresholds:",dict(zip(LABELS,ths.round(3))))
print("macro tuned :",round(f1_score(Y,P,average="macro",zero_division=0),4))
pr,rc,f1,_=precision_recall_fscore_support(Y,P,average=None,zero_division=0)
for i,l in enumerate(LABELS): print(f"  {l}: P={pr[i]:.3f} R={rc[i]:.3f} F1={f1[i]:.3f}  (n={int(Y[:,i].sum())})")
# per-domain macro (matches leaderboard columns)
for dom in ["editorial","debate"]:
    m=np.array(types)==dom
    print(f"macro tuned [{dom}] :",round(f1_score(Y[m],P[m],average="macro",zero_division=0),4))
np.save(f"{W}/oof/task1_{TAG}_ths.npy",ths)
