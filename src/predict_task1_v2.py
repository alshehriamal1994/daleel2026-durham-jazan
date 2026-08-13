import os
import json, sys, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=8; LR=2e-5; BS=16
MNAME={"camelbert":"CAMeL-Lab/bert-base-arabic-camelbert-mix","arabertv2":"aubmindlab/bert-base-arabertv2",
       "camdapt":f"{W}/models/camelbert-dapt"}
# config: name -> list of (tag, seed, posw) and the OOF files to derive thresholds
CONFIGS={
 "blend":  {"specs":[("camelbert",42,1),("arabertv2",42,1)], "oof":["task1_camelbert_s42_pw","task1_arabertv2_s42_pw"]},
 "single": {"specs":[("camelbert",42,1)], "oof":["task1_camelbert_s42_pw"]},
 "robust": {"specs":[("camelbert",42,0),("camelbert",1,0),("camelbert",2,0),("camelbert",7,0),
                     ("arabertv2",42,0),("arabertv2",1,0),("arabertv2",2,0)],
            "ths_json":"task1_robust.json"},
 "dapt":   {"specs":[("camdapt",42,0),("camdapt",1,0),("camdapt",2,0),
                     ("camelbert",42,0),("camelbert",1,0),("camelbert",2,0),("camelbert",7,0),
                     ("arabertv2",42,0),("arabertv2",1,0),("arabertv2",2,0)],
            "ths_json":"task1_dapt.json"},
}
CFG=sys.argv[1] if len(sys.argv)>1 else "blend"
specs=CONFIGS[CFG]["specs"]; oof_names=CONFIGS[CFG].get("oof")

rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
pos=Y.sum(0); posw=np.clip(np.sqrt((len(Y)-pos)/np.maximum(pos,1)),1,6).astype(np.float32)
dev=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]

# thresholds: from robust json if given, else in-sample on blended OOF
if CONFIGS[CFG].get("ths_json"):
    ths=np.array(json.load(open(f"{W}/oof/{CONFIGS[CFG]['ths_json']}"))["ths"])
    print(f"[{CFG}] robust ths={ths.round(3).tolist()}")
else:
    oof_blend=np.mean([np.load(f"{W}/oof/{n}.npz")["oof"] for n in oof_names],axis=0)
    ths=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt=-1,0.5
        for t in grid:
            f=f1_score(Y[:,k],(oof_blend[:,k]>=t).astype(int),zero_division=0)
            if f>b: b,bt=f,t
        ths[k]=bt
    print(f"[{CFG}] CV macro={f1_score(Y,(oof_blend>=ths).astype(int),average='macro',zero_division=0):.4f} ths={ths.round(3).tolist()}")

class WTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,**kw):
        labels=inputs.pop("labels"); out=model(**inputs)
        pw=torch.tensor(posw,device=out.logits.device) if self.use_pw else None
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out.logits,labels,pos_weight=pw)
        return (loss,out) if return_outputs else loss

dev_probs=[]
for tag,seed,pw in specs:
    set_seed(seed); mname=MNAME[tag]; tok=AutoTokenizer.from_pretrained(mname)
    def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
    dtr=Dataset.from_dict({"text":[r["text"] for r in rows],"labels":[Y[i].tolist() for i in range(len(rows))]}).map(enc,batched=True)
    dde=Dataset.from_dict({"text":[r["text"] for r in dev]}).map(enc,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(mname,num_labels=6,problem_type="multi_label_classification")
    args=TrainingArguments(output_dir=f"{W}/models/full_{tag}_{seed}_{pw}",per_device_train_batch_size=BS,
        per_device_eval_batch_size=32,learning_rate=LR,num_train_epochs=EPOCHS,warmup_ratio=0.1,seed=seed,
        weight_decay=0.01,fp16=True,report_to=[],save_strategy="no",eval_strategy="no",logging_steps=200)
    tr=WTrainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok)); tr.use_pw=bool(pw)
    tr.train()
    dev_probs.append(torch.sigmoid(torch.tensor(tr.predict(dde).predictions)).numpy())
    del model,tr; torch.cuda.empty_cache(); print(f"[{tag} s{seed} pw{pw}] done",flush=True)

avg=np.mean(dev_probs,axis=0); P=(avg>=ths).astype(int)
out=f"{W}/preds/task1_dev_{CFG}.jsonl"
with open(out,"w",encoding="utf-8") as f:
    for i,r in enumerate(dev):
        labs=[LABELS[k] for k in range(6) if P[i,k]==1]
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":labs,"type":r["type"]},ensure_ascii=False)+"\n")
import collections; c=collections.Counter(LABELS[k] for i in range(len(dev)) for k in range(6) if P[i,k])
print(f"wrote {out}  counts={dict(c)} empty={int((P.sum(1)==0).sum())}")
