import json, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; DAPT=f"{W}/models/camelbert-dapt"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
bt=[json.loads(l) for l in open(f"{W}/data/train_task_1_bt.jsonl",encoding="utf-8")]
def yv(r):
    y=[0.]*6
    for x in set(r["labels"]): y[L2I[x]]=1.
    return y
Y=np.array([yv(r) for r in rows],dtype=np.float32); texts=[r["text"] for r in rows]; btx=[r["text"] for r in bt]
def tp(model_path,seed,tr,pred_idx):
    set_seed(seed); tok=AutoTokenizer.from_pretrained(model_path)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    tt=[texts[i] for i in tr]+[btx[i] for i in tr]; tl=[Y[i].tolist() for i in tr]+[Y[i].tolist() for i in tr]
    dtr=Dataset.from_dict({"text":tt,"labels":tl}).map(enc,batched=True)
    dpr=Dataset.from_dict({"text":[texts[i] for i in pred_idx]}).map(enc,batched=True)
    m=AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=6,problem_type="multi_label_classification")
    a=TrainingArguments(output_dir=f"{W}/models/ec",per_device_train_batch_size=16,per_device_eval_batch_size=32,
        learning_rate=2e-5,num_train_epochs=8,warmup_ratio=0.1,weight_decay=0.01,fp16=True,report_to=[],
        save_strategy="no",eval_strategy="no",logging_steps=3000,seed=seed)
    t=Trainer(model=m,args=a,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train(); p=torch.sigmoid(torch.tensor(t.predict(dpr).predictions)).numpy(); del m,t; torch.cuda.empty_cache(); return p
def score(prob,tr_prob,tr,te):
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt_=-1,0.5
        for x in grid:
            f=f1_score(Y[tr,k],(tr_prob[:,k]>=x).astype(int),zero_division=0)
            if f>b: b,bt_=f,x
        th[k]=bt_
    return f1_score(Y[te],(prob>=th).astype(int),average="macro",zero_division=0)
camO=[]; mixO=[]
for ss in [11,22,33,44]:
    rng=np.random.RandomState(ss); idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    # CAMeLBERT-DAPT 2 seeds
    cte=np.zeros((217,6)); ctr=np.zeros((len(tr),6))
    for sd in [ss,ss+50]:
        cte+=tp(DAPT,sd,tr,te); ctr+=tp(DAPT,sd,tr,tr)
    cte/=2; ctr/=2
    camO.append(score(cte,ctr,tr,te))
    # + AraBERT 1 seed
    ate=tp("aubmindlab/bert-base-arabertv2",ss+200,tr,te); atr=tp("aubmindlab/bert-base-arabertv2",ss+200,tr,tr)
    mte=(cte*2+ate)/3; mtr=(ctr*2+atr)/3
    mixO.append(score(mte,mtr,tr,te))
    print(f"split {ss}: DAPT-only={camO[-1]:.3f}  +AraBERT={mixO[-1]:.3f}",flush=True)
print(f"\nDAPT-only   mean={np.mean(camO):.3f}")
print(f"+AraBERT    mean={np.mean(mixO):.3f}")
