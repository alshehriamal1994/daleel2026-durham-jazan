import os
import json, pickle, os, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8; dev=torch.device("cuda")
CAM=f"{W}/models/camelbert-dapt-v2"; MAR=f"{W}/models/marbert-dapt-v2"
GE,GD,ML=400,5,25
FULL_SEEDS=[42,1,2,3,4,5,6,7]
_r=f"{W}/oof/t2_recal_ths.json"
if not os.path.exists(_r): _r=f"{W}/configs/t2_recal_ths.json"
cfg=json.load(open(_r)); THE=cfg["ths_editorial"]; THD=cfg["ths_debate"]
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_2_ref.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/synth2_all.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/synth2_v2_built.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/synth2_v3_built.jsonl",encoding="utf-8")]
testr=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
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
def train_one(model_name,seed,tf,pf):
    tp=np.zeros(6); tot=0
    for _,_,_,T,M in tf: tp+=(T*M[:,None]).sum(0); tot+=M.sum()
    PW=torch.tensor(np.clip(np.sqrt((tot-tp)/np.maximum(tp,1)),1,8).astype(np.float32),device=dev)
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
    net.eval(); out=[]
    with torch.no_grad():
        for (I,A,O,T,M) in pf:
            ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
            with torch.amp.autocast("cuda"): out.append(torch.sigmoid(net(ids,am))[0].float().cpu().numpy())
    del net,opt; torch.cuda.empty_cache(); return out
def decode(probs,offs,ths,is_ed):
    gap=GE if is_ed else GD
    spans=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0; raw=[]
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                raw.append([offs[i][0],offs[j][1]]); i=j+1
            else: i+=1
        if raw:
            cur=raw[0]
            for s,e in raw[1:]:
                if s-cur[1]<=gap: cur[1]=max(cur[1],e)
                else: spans.append((k,cur)); cur=[s,e]
            spans.append((k,cur))
    return [(k,se) for k,se in spans if se[1]-se[0]>=ML]
cam_tok=AutoTokenizer.from_pretrained(CAM); mar_tok=AutoTokenizer.from_pretrained(MAR)
tf_cam=feats(rows,cam_tok,True); tf_mar=feats(rows,mar_tok,True)
pf_cam=feats(testr,cam_tok,False); pf_mar=feats(testr,mar_tok,False)
def full_family(mname,tfx,pfx):
    acc=[np.zeros((len(p[2]),6),dtype=np.float32) for p in pfx]
    for sd in FULL_SEEDS:
        ps=train_one(mname,sd,tfx,pfx)
        for a,p in zip(acc,ps): a+=p
        print(f"[{os.path.basename(mname)} s{sd}] done",flush=True)
    return [a/len(FULL_SEEDS) for a in acc]
cam_probs=full_family(CAM,tf_cam,pf_cam)
mar_probs=full_family(MAR,tf_mar,pf_mar)
with open(f"{W}/oof/t2_test_probs_open_v7.pkl","wb") as f:
    pickle.dump({"cam":cam_probs,"mar":mar_probs,"cam_offs":[p[2] for p in pf_cam],"mar_offs":[p[2] for p in pf_mar]},f)
out=[]
for di,r in enumerate(testr):
    if r["type"]=="editorial": sp=decode(cam_probs[di],pf_cam[di][2],THE,True)
    else: sp=decode(mar_probs[di],pf_mar[di][2],THD,False)
    out.append({"paragraph_id":r["paragraph_id"],"labels":[{"label":LABELS[k],"span_text":r["text"][s:e],"start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
with open(f"{W}/preds/task2_test_open_v7_raw.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
print("wrote task2_test_open_v7_raw.jsonl spans=",dict(c)," avg/para=",round(sum(len(o['labels']) for o in out)/len(out),2),flush=True)
