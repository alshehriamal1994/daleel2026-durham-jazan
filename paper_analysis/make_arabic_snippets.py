import json, subprocess, re, os

W = "/home/amal/Desktop/daleel2026"
OUT = f"{W}/paper/arabic"
os.makedirs(OUT, exist_ok=True)

gold = {json.loads(l)["paragraph_id"]: json.loads(l)
        for l in open(f"{W}/data/dev_task_2_ref.jsonl", encoding="utf-8")}
dev = {json.loads(l)["paragraph_id"]: json.loads(l)
       for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
synth2 = [json.loads(l) for l in open(f"{W}/data/synth2_all.jsonl", encoding="utf-8")]

COL = {"AS": "#cfe0f5", "AN": "#d5efce", "ST": "#ffdfb0", "TE": "#f6d5d5",
       "CO": "#ecd6f0", "OT": "#e4e4e4"}

CSS = """
@page { size: 95mm 120mm; margin: 2mm; }
body { width: 86mm; margin: 0; font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 10pt; }
.ar { direction: rtl; text-align: right; font-family: 'Noto Naskh Arabic'; font-size: 10.5pt; line-height: 1.75; }
.tag { font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 7pt; font-weight: bold;
       vertical-align: super; padding: 0 1px; direction: ltr; unicode-bidi: embed; }
.sp { padding: 1px 1px; border-radius: 2px; }
.cap { font-size: 8.5pt; margin-top: 2pt; }
"""

def mark(text, spans, shift=0):
    """Wrap [start,end,label] spans (absolute offsets minus shift) in colored marks."""
    spans = sorted(spans, key=lambda s: s[0])
    out, pos = [], 0
    for st, en, lab in spans:
        st, en = st - shift, en - shift
        out.append(text[pos:st])
        out.append(f'<span class="sp" style="background:{COL[lab]}">{text[st:en]}'
                   f'<span class="tag">{lab}</span></span>')
        pos = en
    out.append(text[pos:])
    return "".join(out)

def render(name, body_html):
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body_html}</body></html>"
    hp = f"{OUT}/{name}.html"
    open(hp, "w", encoding="utf-8").write(html)
    pdf = f"{OUT}/{name}_raw.pdf"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu",
                    f"--print-to-pdf={pdf}", "--no-pdf-header-footer", hp],
                   check=True, capture_output=True)
    # tight-crop with ghostscript bbox
    r = subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", pdf],
                       capture_output=True, text=True)
    m = re.search(r"%%HiResBoundingBox: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", r.stderr)
    x0, y0, x1, y1 = map(float, m.groups())
    pad = 2
    w, h = x1 - x0 + 2*pad, y1 - y0 + 2*pad
    subprocess.run(["gs", "-o", f"{OUT}/{name}.pdf", "-sDEVICE=pdfwrite",
                    f"-dDEVICEWIDTHPOINTS={w:.1f}", f"-dDEVICEHEIGHTPOINTS={h:.1f}",
                    "-dFIXEDMEDIA", "-c", f"<</PageOffset [{-(x0-pad):.1f} {-(y0-pad):.1f}]>> setpagedevice",
                    "-f", pdf], check=True, capture_output=True)
    print(name, f"{w:.0f}x{h:.0f}pt")

# --- 1. Background worked example: dev editorial paragraph 956 ---
r = gold[956]
txt = dev[956]["text"]
spans = [(s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]]
render("bg_example", f'<div class="ar">{mark(txt, spans)}</div>')

# --- 2. Fragmentation example: paragraph 186, gold AS [154,316) vs fragments ---
txt = dev[186]["text"]
lo, hi = 154, 322
seg = txt[lo:hi]
g = f'<div class="ar">{mark(seg, [(154,316,"AS")], shift=lo)}</div>'
frags = [(185,197,"AS"), (200,204,"AS"), (211,316,"AS"), (319,322,"AS")]
p = f'<div class="ar">{mark(seg, frags, shift=lo)}</div>'
render("frag_example", f'<div class="cap"><b>Gold:</b></div>{g}<div class="cap"><b>Predicted (raw decode):</b></div>{p}')

# --- 3. Rare-class example: paragraph 95, gold CO predicted AS ---
txt = dev[95]["text"]
render("co_example", f'<div class="ar">{mark(txt, [(0, len(txt), "CO")])}</div>')

# --- 4. Synthetic Task 2 paragraph (4 segments) ---
s = min((r for r in synth2), key=lambda r: abs(len(r["text"])-263))
# use the exact record found earlier: 4 segments CO/AS/ST/AS about e-commerce
for r in synth2:
    labs = [x["label"] for x in r["labels"]]
    if labs == ["CO","AS","ST","AS"] and "التجارة" in r["text"]:
        s = r; break
spans = [(x["start_offset"], x["end_offset"], x["label"]) for x in s["labels"]]
render("synth_example", f'<div class="ar">{mark(s["text"], spans)}</div>')
