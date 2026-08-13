import os
import json, torch
from transformers import MarianMTModel, MarianTokenizer
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); dev=torch.device("cuda")
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]
rare_idx=[i for i,r in enumerate(rows) if ("ST" in r["labels"]) or ("CO" in r["labels"])]
print("rare examples:",len(rare_idx))
def load(n):
    tk=MarianTokenizer.from_pretrained(n); m=MarianMTModel.from_pretrained(n).to(dev).half().eval(); return tk,m
@torch.no_grad()
def gen(texts,tk,m,seed,bs=16):
    torch.manual_seed(seed); out=[]
    for i in range(0,len(texts),bs):
        enc=tk(texts[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=512).to(dev)
        g=m.generate(**enc,max_length=512,do_sample=True,top_p=0.92,temperature=1.2,num_beams=1)
        out+=tk.batch_decode(g,skip_special_tokens=True)
    return out
rtexts=[rows[i]["text"] for i in rare_idx]
tkae,mae=load("Helsinki-NLP/opus-mt-ar-en")
ens=[gen(rtexts,tkae,mae,s) for s in [101,202,303]]  # 3 diverse EN
del mae; torch.cuda.empty_cache()
tkea,mea=load("Helsinki-NLP/opus-mt-en-ar")
# 5 extra rare paraphrases
variants=[]
for vi,(en,seed) in enumerate(zip(ens+ens[:2],[11,22,33,44,55])):
    ar=gen(en,tkea,mea,seed); variants.append(ar); print(f"  rare variant {vi} done")
# save: map rare_idx -> list of paraphrases
out={str(rare_idx[j]):[variants[v][j] for v in range(len(variants))] for j in range(len(rare_idx))}
json.dump(out,open(f"{W}/data/rare_bt_extra.json","w"),ensure_ascii=False)
print("saved rare_bt_extra.json with",len(variants),"paraphrases each for",len(rare_idx),"rare examples")
