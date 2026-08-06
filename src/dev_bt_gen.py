import json, torch
from transformers import MarianMTModel, MarianTokenizer
W="/home/amal/Desktop/daleel2026"; dev=torch.device("cuda")
rows=[json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl",encoding="utf-8")]
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
# mirror train generation: bt = beam pivot + beam decode; bt2 = beam pivot + sampled decode (seed 11); bt3 = sampled pivot (seed 7) + sampled decode (seed 23)
print("AR->EN (beam)"); tkae,mae=load("Helsinki-NLP/opus-mt-ar-en"); en=gen(texts,tkae,mae,False,0)
print("AR->EN (sample)"); en2=gen(texts,tkae,mae,True,7); del mae; torch.cuda.empty_cache()
tkea,mea=load("Helsinki-NLP/opus-mt-en-ar")
print("EN->AR (beam) -> dev_bt"); bt=gen(en,tkea,mea,False,0)
print("EN->AR (sample) -> dev_bt2"); bt2=gen(en,tkea,mea,True,11)
print("EN->AR (sample) -> dev_bt3"); bt3=gen(en2,tkea,mea,True,23)
for tag,ar in [("bt",bt),("bt2",bt2),("bt3",bt3)]:
    with open(f"{W}/data/dev_task_1_{tag}.jsonl","w",encoding="utf-8") as f:
        for r,t in zip(rows,ar):
            f.write(json.dumps({"paragraph_id":r["paragraph_id"],"text":t,"labels":r["labels"],"type":r["type"]},ensure_ascii=False)+"\n")
    print(f"wrote dev_task_1_{tag}.jsonl",len(ar))
print("ORIG:",texts[0][:90]); print("BT  :",bt[0][:90])
