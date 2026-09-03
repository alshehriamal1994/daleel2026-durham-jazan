import os
import json, numpy as np, torch
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16; FOLDS=5
CV_SEEDS=[42,1,2]; FULL_SEEDS=[42,1,2,3,4,5,6,7]
MAR=f"{W}/models/marbert-dapt-v2"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl",encoding="utf-8")]
bt1=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt.jsonl",encoding="utf-8")]
bt2=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt2.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt2.jsonl",encoding="utf-8")]
bt3=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt3.jsonl",encoding="utf-8")]+\
    [json.loads(l) for l in open(f"{W}/data/dev_task_1_bt3.jsonl",encoding="utf-8")]
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
    return tt,tl
tok=AutoTokenizer.from_pretrained(MAR)
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
def train_pred(seed,tr_idx,pred_texts):
    set_seed(seed); tt,tl=aug_train(tr_idx)
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dpr=Dataset.from_dict({"text":pred_texts}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(MAR,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/drs",per_device_train_batch_size=BS,per_device_eval_batch_size=32,
        learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=3000,seed=seed)
    t=Trainer(model=m,args=a,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dpr).predictions)).numpy(); del m,t; torch.cuda.empty_cache(); return p
strat=[f"{types[i]}_{int(Y[i,L2I['CO']])}{int(Y[i,L2I['ST']])}" for i in range(len(rows))]
c=Counter(strat); strat=[s if c[s]>=FOLDS else types[i] for i,s in enumerate(strat)]
oof=np.zeros((len(rows),6)); skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(texts,strat)):
    acc=np.zeros((len(va),6))
    for sd in CV_SEEDS: acc+=train_pred(sd*100+fold,tr,[texts[i] for i in va])
    oof[va]=acc/len(CV_SEEDS); print(f"[cv fold {fold}] done",flush=True)
np.save(f"{W}/oof/t1_marbert_oof.npy",oof)
# per-domain diagnosis vs CAMeLBERT OOF at matched thresholds (report only)
cam=np.load(f"{W}/oof/t1_recal_oof_closed.npy")
_c=f"{W}/oof/t1_recal_ths_closed.json"
if not os.path.exists(_c): _c=f"{W}/configs/t1_recal_ths_closed.json"
ths=np.array(json.load(open(_c))["ths"])
ed=np.array([t=="editorial" for t in types])
for name,o in [("CAM",cam),("MAR",oof),("MEAN",(cam+oof)/2)]:
    P=(o>=ths).astype(int)
    print(f"[{name}] OOF macro: overall {f1_score(Y,P,average='macro',zero_division=0):.4f} | ed {f1_score(Y[ed],P[ed],average='macro',zero_division=0):.4f} | db {f1_score(Y[~ed],P[~ed],average='macro',zero_division=0):.4f}",flush=True)
# full retrain on all data, save test probs
allidx=np.arange(len(rows)); tp=np.zeros((len(test),6))
for sd in FULL_SEEDS: tp+=train_pred(sd,allidx,[r["text"] for r in test]); print(f"[test s{sd}] done",flush=True)
tp/=len(FULL_SEEDS); np.save(f"{W}/oof/t1_marbert_test_probs.npy",tp)
print("saved t1_marbert_test_probs.npy",flush=True)
