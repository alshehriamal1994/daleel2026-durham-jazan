import json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8
dev=torch.device("cuda")
MNAME={"camelbert":"CAMeL-Lab/bert-base-arabic-camelbert-mix","arabertv2":"aubmindlab/bert-base-arabertv2"}
cfg=json.load(open(f"{W}/oof/task2_best.json")); members=cfg["members"]; ths=np.array(cfg["ths"])
# members like camelbert, camelbert_pw, arabertv2, arabertv2_pw
specs=[]
for m in members:
    pw=1 if m.endswith("_pw") else 0; tag=m[:-3] if pw else m
    specs.append((tag,pw))

rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
devr=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]

class Net(nn.Module):
    def __init__(s,mname):
        super().__init__(); s.enc=AutoModel.from_pretrained(mname)
        s.drop=nn.Dropout(0.1); s.head=nn.Linear(s.enc.config.hidden_size,6)
    def forward(s,ids,am): return s.head(s.drop(s.enc(input_ids=ids,attention_mask=am).last_hidden_state))

def feats(rs,tok,train):
    out=[]
    for r in rs:
        e=tok(r["text"],truncation=True,max_length=MAXLEN,return_offsets_mapping=True)
        offs=e["offset_mapping"]; n=len(offs)
        tg=np.zeros((n,6),dtype=np.float32); mk=np.zeros(n,dtype=np.float32)
        for ti,(a,b) in enumerate(offs):
            if b<=a: continue
            mk[ti]=1.0
            if train:
                for lab in r["labels"]:
                    if min(b,lab["end_offset"])-max(a,lab["start_offset"])>0: tg[ti,L2I[lab["label"]]]=1.0
        out.append((e["input_ids"],e["attention_mask"],offs,tg,mk))
    return out
def collate(b):
    m=max(len(x[0]) for x in b)
    ids=torch.zeros(len(b),m,dtype=torch.long); am=torch.zeros(len(b),m,dtype=torch.long)
    tg=torch.zeros(len(b),m,6); mk=torch.zeros(len(b),m)
    for i,(I,A,O,T,M) in enumerate(b):
        L=len(I); ids[i,:L]=torch.tensor(I); am[i,:L]=torch.tensor(A); tg[i,:L]=torch.tensor(T); mk[i,:L]=torch.tensor(M)
    return ids,am,tg,mk

# accumulate char-level probs over dev
char_acc=[np.zeros((len(r["text"]),6),dtype=np.float32) for r in devr]
for tag,pw in specs:
    mname=MNAME[tag]; set_seed(42); tok=AutoTokenizer.from_pretrained(mname)
    tf=feats(rows,tok,True); df=feats(devr,tok,False)
    # token pos_weight
    if pw:
        tp=np.zeros(6); tot=0
        for _,_,_,T,M in tf: tp+=(T*M[:,None]).sum(0); tot+=M.sum()
        PW=torch.tensor(np.clip(np.sqrt((tot-tp)/np.maximum(tp,1)),1,8).astype(np.float32),device=dev)
    else: PW=None
    net=Net(mname).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
    dl=DataLoader(tf,batch_size=BS,shuffle=True,collate_fn=collate)
    sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*EPOCHS),len(dl)*EPOCHS)
    lossf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=PW); scaler=torch.amp.GradScaler(); net.train()
    for ep in range(EPOCHS):
        for ids,am,tg,mk in dl:
            ids,am,tg,mk=ids.to(dev),am.to(dev),tg.to(dev),mk.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l=lossf(net(ids,am),tg)*mk.unsqueeze(-1); loss=l.sum()/mk.sum()/6
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    net.eval()
    with torch.no_grad():
        for di,(I,A,O,T,M) in enumerate(df):
            ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
            with torch.amp.autocast("cuda"):
                p=torch.sigmoid(net(ids,am))[0].float().cpu().numpy()
            cp=np.zeros((len(devr[di]["text"]),6),dtype=np.float32)
            for ti,(a,b) in enumerate(O):
                if b>a: cp[a:b]=p[ti]
            char_acc[di]+=cp
    del net,opt; torch.cuda.empty_cache(); print(f"[{tag} pw{pw}] done",flush=True)

nm=len(specs)
def decode(cp,text):
    cp=cp/nm; spans=[]
    for k in range(6):
        on=cp[:,k]>=ths[k]; i=0; n=len(on)
        while i<n:
            if on[i]:
                j=i
                while j+1<n and on[j+1]: j+=1
                spans.append({"label":LABELS[k],"span_text":text[i:j+1],"start_offset":i,"end_offset":j+1}); i=j+1
            else: i+=1
    return spans
out=[]
for di,r in enumerate(devr):
    out.append({"paragraph_id":r["paragraph_id"],"labels":decode(char_acc[di],r["text"]),"type":r["type"]})
with open(f"{W}/preds/task2_dev.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
print("wrote task2_dev.jsonl n=",len(out)," counts=",dict(c)," avg/para=",round(sum(len(o['labels']) for o in out)/len(out),2))
