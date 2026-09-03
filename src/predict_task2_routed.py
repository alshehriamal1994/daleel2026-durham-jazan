import os
import json, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8; dev=torch.device("cuda")
CAM="CAMeL-Lab/bert-base-arabic-camelbert-mix"; MAR="UBC-NLP/MARBERTv2"
camcfg,marcfg=[json.load(open(f"{W}/oof/{n}" if os.path.exists(f"{W}/oof/{n}") else f"{W}/configs/{n}")) for n in ("task2_camel_ens.json","task2_marbert_deb.json")]
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
devr=[json.loads(l) for l in open(f"{W}/data/dev_in.jsonl",encoding="utf-8")]
class Net(nn.Module):
    def __init__(s,m):
        super().__init__(); s.enc=AutoModel.from_pretrained(m); s.drop=nn.Dropout(0.1); s.head=nn.Linear(s.enc.config.hidden_size,6)
    def forward(s,ids,am): return s.head(s.drop(s.enc(input_ids=ids,attention_mask=am).last_hidden_state))
def collate(b):
    m=max(len(x[0]) for x in b); ids=torch.zeros(len(b),m,dtype=torch.long); am=torch.zeros(len(b),m,dtype=torch.long)
    tg=torch.zeros(len(b),m,6); mk=torch.zeros(len(b),m)
    for i,(I,A,O,T,M) in enumerate(b):
        L=len(I); ids[i,:L]=torch.tensor(I); am[i,:L]=torch.tensor(A); tg[i,:L]=torch.tensor(T); mk[i,:L]=torch.tensor(M)
    return ids,am,tg,mk
def feats(rs,tok,train):
    out=[]
    for r in rs:
        e=tok(r["text"],truncation=True,max_length=MAXLEN,return_offsets_mapping=True); offs=e["offset_mapping"]; n=len(offs)
        tg=np.zeros((n,6),dtype=np.float32); mk=np.zeros(n,dtype=np.float32)
        for ti,(a,b) in enumerate(offs):
            if b<=a: continue
            mk[ti]=1.0
            if train:
                for lab in r["labels"]:
                    if min(b,lab["end_offset"])-max(a,lab["start_offset"])>0: tg[ti,L2I[lab["label"]]]=1.0
        out.append((e["input_ids"],e["attention_mask"],offs,tg,mk))
    return out
def train_family(model_name,seeds):
    tok=AutoTokenizer.from_pretrained(model_name); tf=feats(rows,tok,True); df=feats(devr,tok,False)
    tp=np.zeros(6); tot=0
    for _,_,_,T,M in tf: tp+=(T*M[:,None]).sum(0); tot+=M.sum()
    PW=torch.tensor(np.clip(np.sqrt((tot-tp)/np.maximum(tp,1)),1,8).astype(np.float32),device=dev)
    acc=[np.zeros((len(O),6),dtype=np.float32) for (_,_,O,_,_) in df]
    for seed in seeds:
        set_seed(seed); net=Net(model_name).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
        dl=DataLoader(tf,batch_size=BS,shuffle=True,collate_fn=collate)
        sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*EPOCHS),len(dl)*EPOCHS)
        lossf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=PW); sc=torch.amp.GradScaler(); net.train()
        for ep in range(EPOCHS):
            for ids,am,tg,mk in dl:
                ids,am,tg,mk=ids.to(dev),am.to(dev),tg.to(dev),mk.to(dev); opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    l=lossf(net(ids,am),tg)*mk.unsqueeze(-1); loss=l.sum()/mk.sum()/6
                sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
        net.eval()
        with torch.no_grad():
            for di,(I,A,O,T,M) in enumerate(df):
                ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
                with torch.amp.autocast("cuda"): acc[di]+=torch.sigmoid(net(ids,am))[0].float().cpu().numpy()
        del net,opt; torch.cuda.empty_cache(); print(f"[{model_name.split('/')[-1]} s{seed}] done",flush=True)
    return [a/len(seeds) for a in acc],df
def decode(probs,offs,ths,text):
    spans=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                s,e=offs[i][0],offs[j][1]; spans.append({"label":LABELS[k],"span_text":text[s:e],"start_offset":int(s),"end_offset":int(e)}); i=j+1
            else: i+=1
    return spans
cam_probs,cam_df=train_family(CAM,camcfg["seeds"])
mar_probs,mar_df=train_family(MAR,marcfg["seeds"])
the=np.array(camcfg["ths_editorial"]); thd=np.array(marcfg["ths_debate"])
out=[]
for di,r in enumerate(devr):
    if r["type"]=="editorial": sp=decode(cam_probs[di],cam_df[di][2],the,r["text"])
    else: sp=decode(mar_probs[di],mar_df[di][2],thd,r["text"])
    out.append({"paragraph_id":r["paragraph_id"],"labels":sp,"type":r["type"]})
with open(f"{W}/preds/task2_dev_routed2.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
print("wrote task2_dev_routed2.jsonl n=",len(out)," spans=",dict(c)," avg/para=",round(sum(len(o['labels']) for o in out)/len(out),2))
