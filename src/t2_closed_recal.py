import os
import json, pickle, subprocess, tempfile, os, re, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, set_seed
from torch.utils.data import DataLoader
LABELS=["AS","AN","ST","TE","CO","OT"]; L2I={l:i for i,l in enumerate(LABELS)}
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=384; EPOCHS=10; LR=3e-5; BS=8; dev=torch.device("cuda")
CAM=f"{W}/models/camelbert-dapt-v2"; MAR=f"{W}/models/marbert-dapt-v2"
GE,GD,ML=400,5,25   # pp3 compaction, tuned on dev, test-confirmed (sub 868218 = 0.729)
CV_SEEDS=[42,1,2]; FULL_SEEDS=[42,1,2,3,4,5,6,7]; FOLDS=5
SCORER=os.environ.get("DALEEL_SCORER", "")
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_2_ref.jsonl",encoding="utf-8")]
testr=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
camcfg=json.load(open(f"{W}/oof/task2_camel_ens.json")); marcfg=json.load(open(f"{W}/oof/task2_marbert_deb.json"))
OLD_THE=list(camcfg["ths_editorial"]); OLD_THD=list(marcfg["ths_debate"])
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
def score_file(pred_rows,gold_path,typ):
    with tempfile.NamedTemporaryFile('w',suffix='.jsonl',delete=False,encoding='utf-8') as f:
        for r in pred_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n"); p=f.name
    r=subprocess.run(["python3",SCORER,"-g",gold_path,"-p",p,"-t",typ],capture_output=True,text=True); os.unlink(p)
    fs=re.findall(r"F1 = ([0-9.]+)",r.stdout)
    return float(fs[0]) if fs else 0.0
# ---------- Phase A: CV OOF ----------
print("Phase A: CV OOF",flush=True)
cam_tok=AutoTokenizer.from_pretrained(CAM); mar_tok=AutoTokenizer.from_pretrained(MAR)
tf_cam=feats(rows,cam_tok,True); tf_mar=feats(rows,mar_tok,True)
types=[r["type"] for r in rows]
idx_by_type={"editorial":[i for i,t in enumerate(types) if t=="editorial"],"debate":[i for i,t in enumerate(types) if t=="debate"]}
folds=[[] for _ in range(FOLDS)]
for t,idxs in idx_by_type.items():
    for j,i in enumerate(idxs): folds[j%FOLDS].append(i)
oof_cam=[None]*len(rows); oof_mar=[None]*len(rows)
for f_i,va in enumerate(folds):
    tr=[i for i in range(len(rows)) if i not in set(va)]
    for fam,tfx,oof in [("cam",tf_cam,oof_cam),("mar",tf_mar,oof_mar)]:
        tff=[tfx[i] for i in tr]; pff=[tfx[i] for i in va]
        acc=[np.zeros((len(tfx[i][2]),6),dtype=np.float32) for i in va]
        mname=CAM if fam=="cam" else MAR
        for sd in CV_SEEDS:
            ps=train_one(mname,sd*100+f_i,tff,pff)
            for a,p in zip(acc,ps): a+=p
        for a,i in zip(acc,va): oof[i]=a/len(CV_SEEDS)
    print(f"[fold {f_i}] done",flush=True)
with open(f"{W}/oof/t2_recal_oof.pkl","wb") as f:
    pickle.dump({"cam":oof_cam,"mar":oof_mar,"cam_offs":[t[2] for t in tf_cam],"mar_offs":[t[2] for t in tf_mar]},f)
# gold file for OOF scoring
gold_all=f"{W}/oof/t2_gold_all.jsonl"
with open(gold_all,"w",encoding="utf-8") as f:
    for i,r in enumerate(rows):
        f.write(json.dumps({"paragraph_id":100000+i,"labels":r["labels"],"type":r["type"]},ensure_ascii=False)+"\n")
def build_preds(ths_ed,ths_db):
    ed,db=[],[]
    for i,r in enumerate(rows):
        if r["type"]=="editorial":
            sp=decode(oof_cam[i],tf_cam[i][2],ths_ed,True)
            ed.append({"paragraph_id":100000+i,"labels":[{"label":LABELS[k],"span_text":"x","start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
        else:
            sp=decode(oof_mar[i],tf_mar[i][2],ths_db,False)
            db.append({"paragraph_id":100000+i,"labels":[{"label":LABELS[k],"span_text":"x","start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
    return ed,db
print("Phase A2: threshold search",flush=True)
grid=[round(0.1+0.05*i,2) for i in range(17)]
the=list(OLD_THE); thd=list(OLD_THD)
ed0,db0=build_preds(the,thd)
base_ed=score_file(ed0,gold_all,"E"); base_db=score_file(db0,gold_all,"D")
print(f"OOF with OLD ths: ed {base_ed:.4f} db {base_db:.4f}",flush=True)
for rnd in range(2):
    for k in range(6):
        best=score_file(build_preds(the,thd)[0],gold_all,"E"); bv=the[k]
        for v in grid:
            cand=list(the); cand[k]=v
            s=score_file(build_preds(cand,thd)[0],gold_all,"E")
            if s>best: best,bv=s,v
        the[k]=bv
    print(f"[ed round {rnd}] ths={the} F1={best:.4f}",flush=True)
for rnd in range(2):
    for k in range(6):
        best=score_file(build_preds(the,thd)[1],gold_all,"D"); bv=thd[k]
        for v in grid:
            cand=list(thd); cand[k]=v
            s=score_file(build_preds(the,cand)[1],gold_all,"D")
            if s>best: best,bv=s,v
        thd[k]=bv
    print(f"[db round {rnd}] ths={thd} F1={best:.4f}",flush=True)
new_ed=score_file(build_preds(the,thd)[0],gold_all,"E"); new_db=score_file(build_preds(the,thd)[1],gold_all,"D")
if new_ed<=base_ed: the=list(OLD_THE); print("editorial ths: keeping OLD (no OOF gain)",flush=True)
if new_db<=base_db: thd=list(OLD_THD); print("debate ths: keeping OLD (no OOF gain)",flush=True)
print(f"FINAL ths_ed={the} (OOF {max(new_ed,base_ed):.4f} vs old {base_ed:.4f}); ths_db={thd} (OOF {max(new_db,base_db):.4f} vs old {base_db:.4f})",flush=True)
json.dump({"ths_editorial":the,"ths_debate":thd,"oof_ed_old":base_ed,"oof_ed_new":new_ed,"oof_db_old":base_db,"oof_db_new":new_db},open(f"{W}/oof/t2_recal_ths.json","w"))
# ---------- Phase B: full retrain, save TEST PROBS, decode ----------
print("Phase B: full retrain + test probs",flush=True)
pf_cam=feats(testr,cam_tok,False); pf_mar=feats(testr,mar_tok,False)
def full_family(mname,tfx,pfx,seeds):
    acc=[np.zeros((len(p[2]),6),dtype=np.float32) for p in pfx]
    for sd in seeds:
        ps=train_one(mname,sd,tfx,pfx)
        for a,p in zip(acc,ps): a+=p
        print(f"[{os.path.basename(mname)} s{sd}] done",flush=True)
    return [a/len(seeds) for a in acc]
cam_probs=full_family(CAM,tf_cam,pf_cam,FULL_SEEDS)
mar_probs=full_family(MAR,tf_mar,pf_mar,FULL_SEEDS)
with open(f"{W}/oof/t2_test_probs_v5.pkl","wb") as f:
    pickle.dump({"cam":cam_probs,"mar":mar_probs,"cam_offs":[p[2] for p in pf_cam],"mar_offs":[p[2] for p in pf_mar]},f)
out=[]
for di,r in enumerate(testr):
    if r["type"]=="editorial": sp=decode(cam_probs[di],pf_cam[di][2],the,True)
    else: sp=decode(mar_probs[di],pf_mar[di][2],thd,False)
    out.append({"paragraph_id":r["paragraph_id"],"labels":[{"label":LABELS[k],"span_text":r["text"][s:e],"start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
with open(f"{W}/preds/task2_test_closed_v5.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
print("wrote task2_test_closed_v5.jsonl spans=",dict(c)," avg/para=",round(sum(len(o['labels']) for o in out)/len(out),2),flush=True)
