import os
import json, sys, re, numpy as np, torch
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MODEL=sys.argv[1]; MODE=sys.argv[2] if len(sys.argv)>2 else "holdout"   # holdout | full
W=os.environ.get("DALEEL_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); MAXLEN=896; EPOCHS=3; LR=2e-4; ACC=8
L=["AS","AN","ST","TE","CO","OT"]
DEF=("AS (افتراض): افتراضات أو استنتاجات أو آراء أو أحكام أو ادعاءات تحتاج إلى دعم. "
     "AN (واقعة): دليل عبر تجربة شخصية أو قصة أو حدث واقعي أو مثال ملموس. "
     "ST (إحصائية): دليل كمي أو دراسات أو نسب وأرقام، ولا يُشترط ذكر المصدر. "
     "TE (شهادة): اقتباس أو استشهاد بخبراء أو جهات أو مصادر محددة؛ وفي المناظرات، إعادة ذكر ادعاءات الفريق الخصم تُعد شهادة. "
     "CO (مسلّمة): معرفة متفق عليها أو حقيقة بديهية أو شرح موضوعي لكيفية عمل إجراء ما. "
     "OT (أخرى): لا يسهم إسهاماً حقيقياً في الخطاب الحجاجي (تحيات، تنظيم، انتقالات).")
def prompt(r):
    return [{"role":"user","content":
        f"صنّف الفقرة التالية من نوع ({'مقال افتتاحي' if r['type']=='editorial' else 'مناظرة'}) حسب أنواع الأدلة الحجاجية الموجودة فيها.\n"
        f"التعريفات: {DEF}\n"
        f"أخرج فقط قائمة الرموز الموجودة مفصولة بفواصل، بدون أي شرح.\n\nالفقرة:\n{r['text']}"}]
def target(r): return ", ".join([l for l in L if l in set(r["labels"])])
rows=[json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl",encoding="utf-8")]+\
     [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl",encoding="utf-8")]
test=[json.loads(l) for l in open(f"{W}/data/test_in.jsonl",encoding="utf-8")]
rng=np.random.RandomState(42)
strat=[f"{r['type']}_{int('ST' in r['labels'] or 'CO' in r['labels'])}" for r in rows]
idx=np.arange(len(rows)); val=[]
for s in sorted(set(strat)):   # deterministic split across runs (set order is hash-randomized)
    ii=[i for i in idx if strat[i]==s]; rng.shuffle(ii); val+=ii[:max(1,len(ii)//4)]
val=set(val); tr_idx=[i for i in idx if i not in val]; va_idx=sorted(val)
if MODE=="full": tr_idx=list(idx); va_idx=[]
print(f"train {len(tr_idx)} val {len(va_idx)}",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)
def build(r,with_target):
    msgs=prompt(r)
    ptxt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pids=tok(ptxt,add_special_tokens=False)["input_ids"]
    if not with_target: return pids
    cids=tok(target(r)+tok.eos_token,add_special_tokens=False)["input_ids"]
    ids=(pids+cids)[:MAXLEN]
    labels=([-100]*len(pids)+cids)[:MAXLEN]
    return {"input_ids":ids,"labels":labels}
train_ds=[build(rows[i],True) for i in tr_idx]
class Coll:
    def __call__(s,feats):
        m=max(len(f["input_ids"]) for f in feats)
        ids=torch.full((len(feats),m),tok.pad_token_id or tok.eos_token_id,dtype=torch.long)
        lab=torch.full((len(feats),m),-100,dtype=torch.long); am=torch.zeros(len(feats),m,dtype=torch.long)
        for i,f in enumerate(feats):
            n=len(f["input_ids"]); ids[i,:n]=torch.tensor(f["input_ids"]); lab[i,:n]=torch.tensor(f["labels"]); am[i,:n]=1
        return {"input_ids":ids,"labels":lab,"attention_mask":am}
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
set_seed(42)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,attn_implementation="sdpa",torch_dtype=torch.bfloat16,device_map={"":0})
# manual kbit prep: skip peft's fp32 upcast (OOMs at 14B on 16GB); bf16 layernorms are fine with bf16 compute
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache=False
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()
args=TrainingArguments(output_dir=f"{W}/models/llm_tmp",per_device_train_batch_size=1,gradient_accumulation_steps=ACC,
    num_train_epochs=EPOCHS,learning_rate=LR,lr_scheduler_type="cosine",warmup_ratio=0.03,bf16=True,logging_steps=50,
    save_strategy="no",report_to=[],gradient_checkpointing=True,optim="paged_adamw_8bit")
Trainer(model=model,args=args,train_dataset=train_ds,data_collator=Coll()).train()
model.eval()
@torch.no_grad()
def predict(rs):
    P=np.zeros((len(rs),6))
    for i,r in enumerate(rs):
        pids=build(r,False)
        ids=torch.tensor([pids]).to(model.device)
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        txt=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
        for k,l in enumerate(L):
            if re.search(rf"\b{l}\b",txt): P[i,k]=1
        if i%50==0: print(f"  gen {i}/{len(rs)}",flush=True)
    return P
if MODE=="holdout":
    va=[rows[i] for i in va_idx]
    P=predict(va)
    Y=np.array([[1. if l in set(r["labels"]) else 0. for l in L] for r in va])
    ed=np.array([r["type"]=="editorial" for r in va])
    def mac(y,p): return f1_score(y,p,average="macro",zero_division=0)
    print(f"LLM holdout macro: overall {mac(Y,P):.4f} | ed {mac(Y[ed],P[ed]):.4f} | db {mac(Y[~ed],P[~ed]):.4f}",flush=True)
    cam=np.load(f"{W}/oof/t1_recal_oof_closed.npy")
    _c=f"{W}/oof/t1_recal_ths_closed.json"
    if not os.path.exists(_c): _c=f"{W}/configs/t1_recal_ths_closed.json"
    ths=np.array(json.load(open(_c))["ths"])
    CP=(cam[va_idx]>=ths).astype(int)
    print(f"CAM same rows      : overall {mac(Y,CP):.4f} | ed {mac(Y[ed],CP[ed]):.4f} | db {mac(Y[~ed],CP[~ed]):.4f}",flush=True)
    np.save(f"{W}/oof/t1_llm_holdout_pred.npy",P); np.save(f"{W}/oof/t1_llm_holdout_idx.npy",np.array(va_idx))
else:
    P=predict(test)
    np.save(f"{W}/oof/t1_llm_test_pred.npy",P)
    from collections import Counter
    with open(f"{W}/preds/task1_test_closed_llm.jsonl","w",encoding="utf-8") as f:
        for i,r in enumerate(test):
            f.write(json.dumps({"paragraph_id":r["paragraph_id"],"labels":[L[k] for k in range(6) if P[i,k]],"type":r["type"]},ensure_ascii=False)+"\n")
    print("wrote task1_test_closed_llm.jsonl counts=",dict(Counter(L[k] for i in range(len(test)) for k in range(6) if P[i,k])),flush=True)
