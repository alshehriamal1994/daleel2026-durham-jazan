import json, sys, numpy as np, torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset

LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
MODEL=sys.argv[1]; TAG=sys.argv[2]; SEED=int(sys.argv[3]); POSW=int(sys.argv[4]) if len(sys.argv)>4 else 0
MAXLEN=384; EPOCHS=8; FOLDS=5; LR=2e-5; BS=16
W="/home/amal/Desktop/daleel2026"; set_seed(SEED)
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
texts=[r["text"] for r in rows]; types=[r["type"] for r in rows]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
pos=Y.sum(0); neg=len(Y)-pos
posw=np.clip(np.sqrt(neg/np.maximum(pos,1)),1,6).astype(np.float32) if POSW else None

strat=[f"{r['type']}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i,r in enumerate(rows)]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
tok=AutoTokenizer.from_pretrained(MODEL)
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)

class WTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        labels=inputs.pop("labels")
        out=model(**inputs); logits=out.logits
        pw=torch.tensor(posw,device=logits.device) if posw is not None else None
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=pw)
        return (loss,out) if return_outputs else loss

oof=np.zeros((len(rows),6),dtype=np.float32)
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)  # fixed split for comparable OOF
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr],"labels":[Y[i].tolist() for i in tr]}).map(enc,batched=True)
    dva=Dataset.from_dict({"text":[texts[i] for i in va],"labels":[Y[i].tolist() for i in va]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/tmp_{TAG}_{SEED}_{fold}",per_device_train_batch_size=BS,
        per_device_eval_batch_size=32,learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,seed=SEED,
        weight_decay=0.01,fp16=True,report_to=[],logging_steps=200,save_strategy="no",eval_strategy="no",dataloader_num_workers=4)
    tr_=WTrainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    tr_.train()
    oof[va]=torch.sigmoid(torch.tensor(tr_.predict(dva).predictions)).numpy()
    del model,tr_; torch.cuda.empty_cache()
np.savez(f"{W}/oof/task1_{TAG}_s{SEED}{'_pw' if POSW else ''}.npz",oof=oof,Y=Y,types=np.array(types))
# quick single-model tuned macro
ths=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
for k in range(6):
    best,bt=-1,0.5
    for t in grid:
        f=f1_score(Y[:,k],(oof[:,k]>=t).astype(int),zero_division=0)
        if f>best: best,bt=f,t
    ths[k]=bt
print(f"DONE {TAG} s{SEED} pw{POSW} macro_tuned={f1_score(Y,(oof>=ths).astype(int),average='macro',zero_division=0):.4f}",flush=True)
