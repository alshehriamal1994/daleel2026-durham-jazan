import os
import json, pickle, subprocess, tempfile, os, re, numpy as np
LABELS=["AS","AN","ST","TE","CO","OT"]
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORER=os.environ.get("DALEEL_SCORER", "")
GE,GD,ML=400,5,25; W_ED,W_DB=0.5,0.35
rows=[json.loads(l) for l in open(f"{W}/data/train_task_2.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_2_ref.jsonl",encoding="utf-8")]
d=pickle.load(open(f"{W}/oof/t2_recal_oof.pkl","rb"))
cfg=json.load(open(f"{W}/oof/t2_recal_ths.json")); THE=list(cfg["ths_editorial"]); THD=list(cfg["ths_debate"])
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
BLEND=[None]*len(rows)
for i,r in enumerate(rows):
    n=len(r["text"]); is_ed=r["type"]=="editorial"; w=W_ED if is_ed else W_DB
    BLEND[i]=w*charprobs(d["cam"][i],d["cam_offs"][i],n)+(1-w)*charprobs(d["mar"][i],d["mar_offs"][i],n)
def build(ths_ed,ths_db,typ):
    recs=[]
    for i,r in enumerate(rows):
        is_ed=r["type"]=="editorial"
        if (typ=="E")!=is_ed: continue
        sp=decode_char(BLEND[i],ths_ed if is_ed else ths_db,is_ed)
        recs.append({"paragraph_id":100000+i,"labels":[{"label":LABELS[k],"span_text":"x","start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
    return recs
def score(recs,typ):
    with tempfile.NamedTemporaryFile('w',suffix='.jsonl',delete=False,encoding='utf-8') as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n"); p=f.name
    r=subprocess.run(["python3",SCORER,"-g",gold_all,"-p",p,"-t",typ],capture_output=True,text=True); os.unlink(p)
    fs=re.findall(r"F1 = ([0-9.]+)",r.stdout)
    return float(fs[0]) if fs else 0.0
grid=[round(0.1+0.05*i,2) for i in range(17)]
base_ed=score(build(THE,THD,"E"),"E"); base_db=score(build(THE,THD,"D"),"D")
print(f"blended with recal ths: ed {base_ed:.4f} db {base_db:.4f}",flush=True)
the=list(THE); thd=list(THD)
for rnd in range(2):
    for k in range(6):
        best=score(build(the,thd,"E"),"E"); bv=the[k]
        for v in grid:
            cand=list(the); cand[k]=v
            s=score(build(cand,thd,"E"),"E")
            if s>best: best,bv=s,v
        the[k]=bv
    print(f"[ed round {rnd}] {the} F1={best:.4f}",flush=True)
for rnd in range(2):
    for k in range(6):
        best=score(build(the,thd,"D"),"D"); bv=thd[k]
        for v in grid:
            cand=list(thd); cand[k]=v
            s=score(build(the,cand,"D"),"D")
            if s>best: best,bv=s,v
        thd[k]=bv
    print(f"[db round {rnd}] {thd} F1={best:.4f}",flush=True)
fin_ed=score(build(the,thd,"E"),"E"); fin_db=score(build(the,thd,"D"),"D")
print(f"FINAL blended OOF: ed {fin_ed:.4f} (recal-single was 0.7289) db {fin_db:.4f} (0.7333)",flush=True)
json.dump({"w_ed":W_ED,"w_db":W_DB,"ths_editorial":the,"ths_debate":thd,"oof_ed":fin_ed,"oof_db":fin_db},open(f"{W}/oof/t2_blend_cfg.json","w"))
# build TEST submissions from saved probs (open + closed)
testr=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
for tag,pklf in [("open","t2_test_probs_open_v5.pkl"),("closed","t2_test_probs_v5.pkl")]:
    td=pickle.load(open(f"{W}/oof/{pklf}","rb"))
    out=[]
    for i,r in enumerate(testr):
        n=len(r["text"]); is_ed=r["type"]=="editorial"; w=W_ED if is_ed else W_DB
        cp=w*charprobs(td["cam"][i],td["cam_offs"][i],n)+(1-w)*charprobs(td["mar"][i],td["mar_offs"][i],n)
        sp=decode_char(cp,the if is_ed else thd,is_ed)
        out.append({"paragraph_id":r["paragraph_id"],"labels":[{"label":LABELS[k],"span_text":r["text"][s:e],"start_offset":int(s),"end_offset":int(e)} for k,(s,e) in sp],"type":r["type"]})
    fn=f"{W}/preds/task2_test_{tag}_v6blend.jsonl"
    with open(fn,"w",encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
    import collections; c=collections.Counter(s["label"] for o in out for s in o["labels"])
    print("wrote",fn,"spans=",dict(c),flush=True)
