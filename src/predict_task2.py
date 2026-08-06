import json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
MODEL="CAMeL-Lab/bert-base-arabic-camelbert-mix"; TAG="camelbert"
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8
dev=torch.device("cuda"); tok=AutoTokenizer.from_pretrained(MODEL)
ths=np.load(f"{W}/oof/task2_{TAG}_pw_ths.npy")
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
devr=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]

def featurize(r,train=True):
    e=tok(r["text"],truncation=True,max_length=MAXLEN,return_offsets_mapping=True)
    offs=e["offset_mapping"]; n=len(offs)
    tgt=np.zeros((n,6),dtype=np.float32); mask=np.zeros(n,dtype=np.float32)
    for ti,(a,b) in enumerate(offs):
        if b<=a: continue
        mask[ti]=1.0
        if train:
            for lab in r["labels"]:
                if min(b,lab["end_offset"])-max(a,lab["start_offset"])>0: tgt[ti,L2I[lab["label"]]]=1.0
    return e["input_ids"],e["attention_mask"],offs,tgt,mask
feats=[featurize(r) for r in rows]
dfeats=[featurize(r,train=False) for r in devr]
# token-level pos_weight (class weighting), matches camelbert_pw CV config
_tp=np.zeros(6); _tot=0
for _,_,_,T,M in feats: _tp+=(T*M[:,None]).sum(0); _tot+=M.sum()
POS_WEIGHT=torch.tensor(np.clip(np.sqrt((_tot-_tp)/np.maximum(_tp,1)),1,8).astype(np.float32),device=dev)

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=AutoModel.from_pretrained(MODEL)
        s.drop=nn.Dropout(0.1); s.head=nn.Linear(s.enc.config.hidden_size,6)
    def forward(s,ids,am):
        return s.head(s.drop(s.enc(input_ids=ids,attention_mask=am).last_hidden_state))
def collate(b):
    m=max(len(x[0]) for x in b)
    ids=torch.zeros(len(b),m,dtype=torch.long); am=torch.zeros(len(b),m,dtype=torch.long)
    tg=torch.zeros(len(b),m,6); mk=torch.zeros(len(b),m)
    for i,(I,A,O,T,M) in enumerate(b):
        L=len(I); ids[i,:L]=torch.tensor(I); am[i,:L]=torch.tensor(A); tg[i,:L]=torch.tensor(T); mk[i,:L]=torch.tensor(M)
    return ids,am,tg,mk
def decode(probs,offs,ths,text):
    spans=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                s,e=offs[i][0],offs[j][1]
                spans.append({"label":LABELS[k],"span_text":text[s:e],"start_offset":int(s),"end_offset":int(e)}); i=j+1
            else: i+=1
    return spans

net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
dl=DataLoader(feats,batch_size=BS,shuffle=True,collate_fn=collate)
sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*EPOCHS),len(dl)*EPOCHS)
lossf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=POS_WEIGHT); scaler=torch.amp.GradScaler(); net.train()
for ep in range(EPOCHS):
    for ids,am,tg,mk in dl:
        ids,am,tg,mk=ids.to(dev),am.to(dev),tg.to(dev),mk.to(dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            l=lossf(net(ids,am),tg)*mk.unsqueeze(-1); loss=l.sum()/mk.sum()/6
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    print(f"epoch {ep} done",flush=True)
net.eval()
out=[]
with torch.no_grad():
    for r,(I,A,O,T,M) in zip(devr,dfeats):
        ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
        with torch.amp.autocast("cuda"):
            p=torch.sigmoid(net(ids,am))[0].float().cpu().numpy()
        sp=decode(p,O,ths,r["text"])
        out.append({"paragraph_id":r["paragraph_id"],"labels":sp,"type":r["type"]})
with open(f"{W}/preds/task2_dev.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
print("wrote task2_dev.jsonl  n=",len(out)," span label counts:",dict(c))
print("avg spans/para:",round(sum(len(o['labels']) for o in out)/len(out),2))
