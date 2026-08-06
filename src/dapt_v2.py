import json, torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from datasets import Dataset
W="/home/amal/Desktop/daleel2026"
# v2 corpus: ALL task text incl. the released test paragraphs (labels never used) — per dapt.py's original plan
texts=[]
for f in ["data/train_task_1.jsonl","data/dev_in.jsonl","data/test_in.jsonl"]:
    texts+=[json.loads(l)["text"] for l in open(f"{W}/{f}",encoding="utf-8")]
print("DAPT-v2 corpus paragraphs:",len(texts))
for MODEL,OUT in [("CAMeL-Lab/bert-base-arabic-camelbert-mix",f"{W}/models/camelbert-dapt-v2"),
                  ("UBC-NLP/MARBERTv2",f"{W}/models/marbert-dapt-v2")]:
    tok=AutoTokenizer.from_pretrained(MODEL)
    ds=Dataset.from_dict({"text":texts}).map(lambda b: tok(b["text"],truncation=True,max_length=384),batched=True,remove_columns=["text"])
    model=AutoModelForMaskedLM.from_pretrained(MODEL)
    args=TrainingArguments(output_dir=f"{W}/models/dapt_tmp",per_device_train_batch_size=16,num_train_epochs=30,
        learning_rate=5e-5,warmup_ratio=0.05,weight_decay=0.01,fp16=True,report_to=[],save_strategy="no",
        logging_steps=200,dataloader_num_workers=4)
    coll=DataCollatorForLanguageModeling(tokenizer=tok,mlm=True,mlm_probability=0.15)
    Trainer(model=model,args=args,train_dataset=ds,data_collator=coll).train()
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved",OUT,flush=True)
    del model; torch.cuda.empty_cache()
