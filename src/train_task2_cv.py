import json, sys, numpy as np, torch, torch.nn as nn
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
sys.path.insert(0,"/home/amal/Desktop/daleel2026/src")
import task2_scoring as T2

LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
MODEL=sys.argv[1] if len(sys.argv)>1 else "CAMeL-Lab/bert-base-arabic-camelbert-mix"
TAG=sys.argv[2] if len(sys.argv)>2 else "camelbert"
POSW=int(sys.argv[3]) if len(sys.argv)>3 else 0
SEED=int(sys.argv[4]) if len(sys.argv)>4 else 42
W="/home/amal/Desktop/daleel2026"; MAXLEN=384; EPOCHS=10; FOLDS=5; LR=3e-5; BS=8
dev=torch.device("cuda")
torch.manual_seed(SEED); np.random.seed(SEED)
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]
tok=AutoTokenizer.from_pretrained(MODEL)

def featurize(r):
    e=tok(r["text"],truncation=True,max_length=MAXLEN,return_offsets_mapping=True)
    offs=e["offset_mapping"]; n=len(offs)
    tgt=np.zeros((n,6),dtype=np.float32); mask=np.zeros(n,dtype=np.float32)
    for ti,(a,b) in enumerate(offs):
        if b<=a: continue  # special tokens
        mask[ti]=1.0
        for lab in r["labels"]:
            s,t=lab["start_offset"],lab["end_offset"]
            if min(b,t)-max(a,s)>0: tgt[ti,L2I[lab["label"]]]=1.0
    return e["input_ids"],e["attention_mask"],offs,tgt,mask

feats=[featurize(r) for r in rows]
# token-level class frequencies for pos_weight
_tp=np.zeros(6); _tot=0
for _,_,_,T,M in feats:
    _tp+=(T*M[:,None]).sum(0); _tot+=M.sum()
POS_WEIGHT=np.clip(np.sqrt((_tot-_tp)/np.maximum(_tp,1)),1,8).astype(np.float32) if POSW else None

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=AutoModel.from_pretrained(MODEL)
        s.drop=nn.Dropout(0.1); s.head=nn.Linear(s.enc.config.hidden_size,6)
    def forward(s,ids,am):
        h=s.enc(input_ids=ids,attention_mask=am).last_hidden_state
        return s.head(s.drop(h))

def collate(batch):
    m=max(len(b[0]) for b in batch)
    ids=torch.zeros(len(batch),m,dtype=torch.long); am=torch.zeros(len(batch),m,dtype=torch.long)
    tg=torch.zeros(len(batch),m,6); mk=torch.zeros(len(batch),m)
    for i,(I,A,O,T,M) in enumerate(batch):
        L=len(I); ids[i,:L]=torch.tensor(I); am[i,:L]=torch.tensor(A)
        tg[i,:L]=torch.tensor(T); mk[i,:L]=torch.tensor(M)
    return ids,am,tg,mk

def decode(probs,offs,ths):
    """probs [n,6] -> list of (label,[start,end]) by contiguous runs per label."""
    spans=[]
    for k in range(6):
        on=probs[:,k]>=ths[k]; i=0
        while i<len(on):
            if on[i] and offs[i][1]>offs[i][0]:
                j=i
                while j+1<len(on) and on[j+1] and offs[j+1][1]>offs[j+1][0]: j+=1
                spans.append((LABELS[k],[offs[i][0],offs[j][1]])); i=j+1
            else: i+=1
    return spans

strat=[f"{r['type']}" for r in rows]
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=42)
oof_probs=[None]*len(rows)
for fold,(tr,va) in enumerate(skf.split(rows,strat)):
    net=Net().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
    dl=DataLoader([feats[i] for i in tr],batch_size=BS,shuffle=True,collate_fn=collate)
    sch=get_linear_schedule_with_warmup(opt,int(0.1*len(dl)*EPOCHS),len(dl)*EPOCHS)
    _pw=torch.tensor(POS_WEIGHT,device=dev) if POS_WEIGHT is not None else None
    lossf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=_pw)
    scaler=torch.amp.GradScaler()
    net.train()
    for ep in range(EPOCHS):
        for ids,am,tg,mk in dl:
            ids,am,tg,mk=ids.to(dev),am.to(dev),tg.to(dev),mk.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                lo=net(ids,am); l=lossf(lo,tg)*mk.unsqueeze(-1)
                loss=l.sum()/mk.sum()/6
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    net.eval()
    with torch.no_grad():
        for i in va:
            I,A,O,T,M=feats[i]
            ids=torch.tensor([I]).to(dev); am=torch.tensor([A]).to(dev)
            with torch.amp.autocast("cuda"):
                p=torch.sigmoid(net(ids,am))[0].float().cpu().numpy()
            oof_probs[i]=(p,O)
    del net,opt; torch.cuda.empty_cache(); print(f"[fold {fold}] done",flush=True)

# save
SFX=f"{TAG}{'_pw' if POSW else ''}_s{SEED}"
np.save(f"{W}/oof/task2_{SFX}_probs.npy",np.array(oof_probs,dtype=object),allow_pickle=True)

# gold dict for scorer
gold=defaultdict(list)
for r in rows:
    gold[r["paragraph_id"]]=T2.sort_spans([(l["label"],[l["start_offset"],l["end_offset"]]) for l in r["labels"]])

def eval_ths(ths):
    pred=defaultdict(list)
    for i,r in enumerate(rows):
        p,offs=oof_probs[i]
        pred[r["paragraph_id"]]=T2.sort_spans(decode(p,offs,ths))
    return T2.score_per_span(gold,pred)[2]

print(f"\n=== {SFX} task2 ===")
print("F1 @0.5 :",round(eval_ths(np.full(6,0.5)),4))
# coordinate-ascent threshold tuning on official metric
ths=np.full(6,0.5); grid=np.arange(0.2,0.85,0.05)
for _ in range(3):
    for k in range(6):
        best,bt=-1,ths[k]
        for t in grid:
            ths[k]=t; f=eval_ths(ths)
            if f>best: best,bt=f,t
        ths[k]=bt
print("tuned ths:",dict(zip(LABELS,ths.round(2))))
print("F1 tuned :",round(eval_ths(ths),4))
np.save(f"{W}/oof/task2_{SFX}_ths.npy",ths)
