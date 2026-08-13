import os
import json, torch
from transformers import MarianMTModel, MarianTokenizer
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); dev=torch.device("cuda")
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
def load(name):
    tk=MarianTokenizer.from_pretrained(name); m=MarianMTModel.from_pretrained(name).to(dev).half().eval(); return tk,m
@torch.no_grad()
def gen(texts,tk,m,sample,seed,bs=16):
    if sample: torch.manual_seed(seed)
    out=[]
    for i in range(0,len(texts),bs):
        enc=tk(texts[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=512).to(dev)
        kw=dict(max_length=512)
        if sample: kw.update(do_sample=True,top_p=0.95,temperature=1.1,num_beams=1)
        else: kw.update(num_beams=2)
        g=m.generate(**enc,**kw); out+=tk.batch_decode(g,skip_special_tokens=True)
        print(f"  {min(i+bs,len(texts))}/{len(texts)}",end="\r",flush=True)
    print(); return out
texts=[r["text"] for r in rows]
print("AR->EN (beam)"); tkae,mae=load("Helsinki-NLP/opus-mt-ar-en"); en=gen(texts,tkae,mae,False,0); del mae; torch.cuda.empty_cache()
# also a sampled EN to diversify the pivot
print("AR->EN (sample)"); tkae,mae=load("Helsinki-NLP/opus-mt-ar-en"); en2=gen(texts,tkae,mae,True,7); del mae; torch.cuda.empty_cache()
tkea,mea=load("Helsinki-NLP/opus-mt-en-ar")
for tag,src,seed in [("bt2",en,11),("bt3",en2,23)]:
    print(f"EN->AR sampled -> {tag}"); ar=gen(src,tkea,mea,True,seed)
    with open(f"{W}/data/train_task_1_{tag}.jsonl","w",encoding="utf-8") as f:
        for r,bt in zip(rows,ar):
            f.write(json.dumps({"paragraph_id":r["paragraph_id"],"text":bt,"labels":r["labels"],"type":r["type"]},ensure_ascii=False)+"\n")
    print(f"  wrote {tag}",len(ar))
print("done multi-BT")
