import os
import json, numpy as np, torch, torch.nn as nn, sys
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
from collections import defaultdict
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import task2_scoring as T2
L=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(L)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8; MODEL="UBC-NLP/MARBERTv2"
dev=torch.device("cuda")
real=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
synth=[json.loads(l) for l in open(f"{W}/data/synth2_all.jsonl",encoding="utf-8")]
tok=AutoTokenizer.from_pretrained(MODEL)
def feat(r,train=True):
    e=tok(r["text"],truncation=True,max_length=MAXLEN,return_offsets_mapping=True)
    offs=e["offset_mapping"]; n=len(offs); tg=np.zeros((n,6),dtype=np.float32); mk=np.zeros(n,dtype=np.float32)
    for ti,(a,b) in enumerate(offs):
        if b<=a: continue
        mk[ti]=1.0
        for lab in r["labels"]:
            if min(b,lab["end_offset"])-max(a,lab["start_offset"])>0: tg[ti,L2I[lab["label"]]]=1.0
    return e["input_ids"],e["attention_mask"],offs,tg,mk
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=AutoModel.from_pretrained(MODEL); s.drop=nn.Dropout(0.1); s.head=nn.Linear(s.enc.config.hidden_size,6)
    def forward(s,ids,am): return s.head(s.drop(s.enc(input_ids=ids,attention_mask=am).last_hidden_state))
def collate(b):
    m=max(len(x[0]) for x in b); ids=torch.zeros(len(b),m,dtype=torch.long); am=torch.zeros(len(b),m,dtype=torch.long)
    tg=torch.zeros(len(b),m,6); mk=torch.zeros(len(b),m)
    for i,(I,A,O,T,M) in enumerate(b):
        Ln=len(I); ids[i,:Ln]=torch.tensor(I); am[i,:Ln]=torch.tensor(A); tg[i,:Ln]=torch.tensor(T); mk[i,:Ln]=torch.tensor(M)
    return ids,am,tg,mk
def decode(probs,offs,ths):
    sp=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                sp.append((L[k],[offs[i][0],offs[j][1]])); i=j+1
            else: i+=1
    return sp
def train(feats,seed):
    set_seed(seed); net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
    tp=np.zeros(6); tot=0
    for _,_,_,T,M in feats: tp+=(T*M[:,None]).sum(0); tot+=M.sum()
    PW=torch.tensor(np.clip(np.sqrt((tot-tp)/np.maximum(tp,1)),1,8).astype(np.float32),device=dev)
    dl=DataLoader(feats,batch_size=BS,shuffle=True,collate_fn=collate)
    sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*EPOCHS),len(dl)*EPOCHS)
    lossf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=PW); sc=torch.amp.GradScaler(); net.train()
    for ep in range(EPOCHS):
        for ids,am,tg,mk in dl:
            ids,am,tg,mk=ids.to(dev),am.to(dev),tg.to(dev),mk.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l=lossf(net(ids,am),tg)*mk.unsqueeze(-1); loss=l.sum()/mk.sum()/6
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
    net.eval(); return net
def predict(net,rows,feats):
    out=defaultdict(list)
    with torch.no_grad():
        for r,(I,A,O,T,M) in zip(rows,feats):
            ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
            with torch.amp.autocast("cuda"): p=torch.sigmoid(net(ids,am))[0].float().cpu().numpy()
            out[r["paragraph_id"]]=(p,O)
    return out
def score(net,trrows,trfeats,terows,tefeats):
    # tune thresholds on train (official scorer), eval on test
    tr_pred=predict(net,trrows,trfeats); te_pred=predict(net,terows,tefeats)
    gold_tr={r["paragraph_id"]:T2.sort_spans([(l["label"],[l["start_offset"],l["end_offset"]]) for l in r["labels"]]) for r in trrows}
    gold_te={r["paragraph_id"]:T2.sort_spans([(l["label"],[l["start_offset"],l["end_offset"]]) for l in r["labels"]]) for r in terows}
    def ev(pred,gold,ths):
        pd=defaultdict(list)
        for pid,(p,O) in pred.items(): pd[pid]=T2.sort_spans(decode(p,O,ths))
        return T2.score_per_span(gold,pd)[2]
    ths=np.full(6,0.5); grid=np.arange(0.2,0.85,0.05)
    for _ in range(2):
        for k in range(6):
            best,bt=-1,ths[k]
            for t in grid:
                ths[k]=t; f=ev(tr_pred,gold_tr,ths)
                if f>best: best,bt=f,t
            ths[k]=bt
    return ev(te_pred,gold_te,ths)
synth_feats=[feat(r) for r in synth]
rng=np.random.RandomState(0)
for tag in ["real-only","real+SYNTH"]:
    scores=[]
    for ss in [1,2]:
        idx=rng.permutation(len(real)); te=[real[i] for i in idx[:150]]; tr=[real[i] for i in idx[150:]]
        trf=[feat(r) for r in tr]; tef=[feat(r) for r in te]
        feats=trf+(synth_feats if tag=="real+SYNTH" else [])
        net=train(feats,100+ss)
        scores.append(score(net,tr,trf,te,tef)); del net; torch.cuda.empty_cache()
    print(f"{tag:14s} overlap-F1 mean={np.mean(scores):.4f} runs={[round(x,4) for x in scores]}",flush=True)
