import os
import json, pickle, subprocess, tempfile, os, re, numpy as np
LABELS=["AS","AN","ST","TE","CO","OT"]
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORER=os.environ.get("DALEEL_SCORER", "")
GE,GD,ML=400,5,25
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_2_ref.jsonl",encoding="utf-8")]
d=pickle.load(open(f"{W}/oof/t2_recal_oof.pkl","rb"))
_r=f"{W}/oof/t2_recal_ths.json"
if not os.path.exists(_r): _r=f"{W}/configs/t2_recal_ths.json"
cfg=json.load(open(_r)); THE=np.array(cfg["ths_editorial"]); THD=np.array(cfg["ths_debate"])
gold_all=f"{W}/oof/t2_gold_all.jsonl"
def charprobs(probs,offs,n):
    cp=np.zeros((n,6),dtype=np.float32)
    for (a,b),p in zip(offs,probs):
        if b>a: cp[a:b]=np.maximum(cp[a:b],p)
    return cp
def decode_char(cp,ths,is_ed):
    gap=GE if is_ed else GD
    spans=[]
    for k in range(6):
        on=cp[:,k]>=ths[k]; i=0; raw=[]
        while i<len(on):
            if on[i]:
                j=i
                while j+1<len(on) and on[j+1]: j+=1
                raw.append([i,j+1]); i=j+2
            else: i+=1
        if raw:
            cur=raw[0]
            for s,e in raw[1:]:
                if s-cur[1]<=gap: cur[1]=max(cur[1],e)
                else: spans.append((k,cur)); cur=[s,e]
            spans.append((k,cur))
    return [(k,se) for k,se in spans if se[1]-se[0]>=ML]
CP_CAM=[charprobs(d["cam"][i],d["cam_offs"][i],len(rows[i]["text"])) for i in range(len(rows))]
CP_MAR=[charprobs(d["mar"][i],d["mar_offs"][i],len(rows[i]["text"])) for i in range(len(rows))]
def build(w_ed,w_db):
    ed,db=[],[]
    for i,r in enumerate(rows):
        is_ed=r["type"]=="editorial"
        w=w_ed if is_ed else w_db
        cp=w*CP_CAM[i]+(1-w)*CP_MAR[i]
        sp=decode_char(cp,THE if is_ed else THD,is_ed)
        rec={"paragraph_id":100000+i,"labels":[{"label":LABELS[k],"span_text":"x","start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]}
        (ed if is_ed else db).append(rec)
    return ed,db
def score(recs,typ):
    with tempfile.NamedTemporaryFile('w',suffix='.jsonl',delete=False,encoding='utf-8') as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n"); p=f.name
    r=subprocess.run(["python3",SCORER,"-g",gold_all,"-p",p,"-t",typ],capture_output=True,text=True); os.unlink(p)
    fs=re.findall(r"F1 = ([0-9.]+)",r.stdout)
    return float(fs[0]) if fs else 0.0
# reference: token-level single-family with recal ths scored 0.7289 (ed) / 0.7333 (db) on this OOF
ref_ed,ref_db=build(1.0,0.0)   # char-level pure cam (ed) / pure mar (db) — measures char-decode cost alone
print("char-level pure: ed %.4f db %.4f (token-level refs: 0.7289 / 0.7333)"%(score(ref_ed,"E"),score(ref_db,"D")),flush=True)
print("editorial blend sweep (w=weight on CAM):",flush=True)
for w in [0.8,0.65,0.5,0.35,0.2]:
    ed,_=build(w,0.0); print("  w_ed=%.2f ed=%.4f"%(w,score(ed,"E")),flush=True)
print("debate blend sweep (w=weight on CAM):",flush=True)
for w in [0.2,0.35,0.5,0.65]:
    _,db=build(0.0,w); print("  w_db=%.2f db=%.4f"%(w,score(db,"D")),flush=True)
