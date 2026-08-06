import json, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; MODEL="FreedomIntelligence/AceGPT-v2-8B"
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
Y=np.zeros((len(rows),6),dtype=np.float32)
for i,r in enumerate(rows):
    for l in set(r["labels"]): Y[i,L2I[l]]=1.0
texts=[r["text"] for r in rows]
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
def enc(b): return tok(b["text"],truncation=True,max_length=MAXLEN)
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
def build():
    m=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=6,problem_type="multi_label_classification",
        quantization_config=bnb,device_map={"":0})
    m.config.pad_token_id=tok.pad_token_id
    m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=True)
    lc=LoraConfig(task_type="SEQ_CLS",r=16,lora_alpha=32,lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        modules_to_save=["score"])
    return get_peft_model(m,lc)
def run(split_seed):
    set_seed(100+split_seed); rng=np.random.RandomState(split_seed)
    idx=rng.permutation(len(rows)); te=idx[:217]; tr=idx[217:]
    dtr=Dataset.from_dict({"text":[texts[i] for i in tr],"labels":[Y[i].tolist() for i in tr]}).map(enc,batched=True)
    dall=Dataset.from_dict({"text":[texts[i] for i in list(tr)+list(te)]}).map(enc,batched=True)
    model=build()
    args=TrainingArguments(output_dir=f"{W}/models/llm",per_device_train_batch_size=4,gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,learning_rate=1e-4,num_train_epochs=5,warmup_ratio=0.05,weight_decay=0.0,
        bf16=True,report_to=[],save_strategy="no",eval_strategy="no",logging_steps=20,seed=100+split_seed,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    t=Trainer(model=model,args=args,train_dataset=dtr,data_collator=DataCollatorWithPadding(tok))
    t.train()
    pred=t.predict(dall).predictions
    p=torch.sigmoid(torch.tensor(pred,dtype=torch.float32)).numpy()
    ntr=len(tr); ptr,pte=p[:ntr],p[ntr:]
    th=np.full(6,0.5); grid=np.arange(0.05,0.95,0.025)
    for k in range(6):
        b,bt=-1,0.5
        for tt in grid:
            f=f1_score(Y[tr,k],(ptr[:,k]>=tt).astype(int),zero_division=0)
            if f>b: b,bt=f,tt
        th[k]=bt
    macro=f1_score(Y[te],(pte>=th).astype(int),average="macro",zero_division=0)
    del model,t; torch.cuda.empty_cache()
    print(f"[split {split_seed}] AceGPT-v2-8B clean-holdout macro = {macro:.3f}",flush=True)
    return macro
res=[run(s) for s in [11,22]]
print(f"AceGPT-v2-8B QLoRA clean-holdout: {[round(r,3) for r in res]} mean={np.mean(res):.3f}  (CAMeLBERT baseline=0.619)")
