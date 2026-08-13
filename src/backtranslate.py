import os
import json, torch
from transformers import MarianMTModel, MarianTokenizer
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); dev=torch.device("cuda")
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
def load(name):
    tk=MarianTokenizer.from_pretrained(name); m=MarianMTModel.from_pretrained(name).to(dev).half().eval(); return tk,m
@torch.no_grad()
def translate(texts,tk,m,bs=16):
    out=[]
    for i in range(0,len(texts),bs):
        batch=texts[i:i+bs]
        enc=tk(batch,return_tensors="pt",padding=True,truncation=True,max_length=512).to(dev)
        gen=m.generate(**enc,max_length=512,num_beams=2)
        out+=tk.batch_decode(gen,skip_special_tokens=True)
        print(f"  {min(i+bs,len(texts))}/{len(texts)}",end="\r",flush=True)
    print()
    return out
texts=[r["text"] for r in rows]
print("AR->EN ..."); tk,m=load("Helsinki-NLP/opus-mt-ar-en"); en=translate(texts,tk,m); del m; torch.cuda.empty_cache()
print("EN->AR ..."); tk,m=load("Helsinki-NLP/opus-mt-en-ar"); ar=translate(en,tk,m); del m; torch.cuda.empty_cache()
with open(f"{W}/data/train_task_1_bt.jsonl","w",encoding="utf-8") as f:
    for r,bt in zip(rows,ar):
        f.write(json.dumps({"paragraph_id":r["paragraph_id"],"text":bt,"labels":r["labels"],"type":r["type"]},ensure_ascii=False)+"\n")
print("wrote back-translations:",len(ar))
# sample
print("ORIG:",texts[0][:90]); print("BT  :",ar[0][:90])
