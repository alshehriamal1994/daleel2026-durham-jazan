"""Regenerates all Arabic example figures for the paper, in one consistent style.

Style system: pastel span fill + 2px class-colored underline + small dark chip
badge carrying the label code (identity survives grayscale via the chip text).
Arabic set in Noto Naskh (pinned via @font-face); Latin in Times to match the paper.
Requires: google-chrome (headless), ghostscript, the task data under ../../data.
"""
import json, subprocess, re, os

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # daleel2026/
OUT = os.path.dirname(os.path.abspath(__file__))

PASTEL = {"AS": "#e6eef9", "AN": "#eaf5e5", "ST": "#fdf0da", "TE": "#faeaea", "CO": "#f5ebf7", "OT": "#f0f0f0"}
DARK   = {"AS": "#2b5e9e", "AN": "#3f7d33", "ST": "#a85e00", "TE": "#a33a3a", "CO": "#7c3f8c", "OT": "#5a5a5a"}

NASKH = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
NASKHB = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"

CSS = f"""
@font-face {{ font-family: 'PaperNaskh'; src: url('file://{NASKH}'); }}
@font-face {{ font-family: 'PaperNaskh'; src: url('file://{NASKHB}'); font-weight: bold; }}
@page {{ size: 95mm 200mm; margin: 2mm; }}
body {{ width: 86mm; margin: 0; font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 10pt; }}
.ar {{ direction: rtl; text-align: right; font-family: 'PaperNaskh'; font-size: 10.5pt; line-height: 2.45; }}
.chip {{ font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 5.8pt; font-weight: bold;
        color: #ffffff; padding: 0.5px 2.5px 1px 2.5px; border-radius: 2.5px;
        vertical-align: 2px; margin: 0 1.5px; direction: ltr; unicode-bidi: embed;
        letter-spacing: 0.3px; }}
.sp {{ padding: 1px 2.5px 0.5px 2.5px; border-radius: 2px; margin: 0 1.5px;
      -webkit-box-decoration-break: clone; box-decoration-break: clone; }}
.hd {{ font-variant: small-caps; font-weight: bold; font-size: 9pt; margin: 4pt 0 0.5pt 0;
      border-bottom: 0.5px solid #cccccc; padding-bottom: 1pt; }}
.lg {{ direction: ltr; text-align: center; font-size: 7.2pt; color: #333333;
      margin-top: 5pt; padding-top: 3pt; border-top: 0.5px solid #dddddd; line-height: 1.9; }}
.hd .sc {{ font-variant: normal; font-weight: normal; font-size: 8pt; color: #666666; }}
"""

def chip(lab):
    return f'<span class="chip" style="background:{DARK[lab]}">{lab}</span>'

def span_html(text, lab, pad_bottom=None):
    extra = f" padding-bottom:{pad_bottom}px;" if pad_bottom is not None else ""
    return (f'<span class="sp" style="background:{PASTEL[lab]}; '
            f'border-bottom:2px solid {DARK[lab]};{extra}">{text}{chip(lab)}</span>')

def mark(text, spans, shift=0):
    spans = sorted(spans, key=lambda s: s[0])
    out, pos = [], 0
    for st, en, lab in spans:
        st, en = st - shift, en - shift
        out.append(text[pos:st])
        out.append(span_html(text[st:en], lab))
        pos = en
    out.append(text[pos:])
    return "".join(out)

def render(name, body_html):
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body_html}</body></html>"
    hp = f"{OUT}/{name}.html"
    open(hp, "w", encoding="utf-8").write(html)
    pdf = f"{OUT}/{name}_raw.pdf"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf}", "--no-pdf-header-footer", hp], check=True, capture_output=True)
    r = subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", pdf], capture_output=True, text=True)
    x0, y0, x1, y1 = map(float, re.search(r"%%HiResBoundingBox: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", r.stderr).groups())
    pad = 2
    w, h = x1 - x0 + 2*pad, y1 - y0 + 2*pad
    subprocess.run(["gs", "-o", f"{OUT}/{name}.pdf", "-sDEVICE=pdfwrite",
                    f"-dDEVICEWIDTHPOINTS={w:.1f}", f"-dDEVICEHEIGHTPOINTS={h:.1f}", "-dFIXEDMEDIA",
                    "-c", f"<</PageOffset [{-(x0-pad):.1f} {-(y0-pad):.1f}]>> setpagedevice",
                    "-f", pdf], check=True, capture_output=True)
    os.remove(pdf)
    print(name, f"{w:.0f}x{h:.0f}pt")

# ---------------- data ----------------
def load2(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        out[r["paragraph_id"]] = [(s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]]
    return out

gold = load2(f"{W}/data/dev_task_2_ref.jsonl")
dev = {json.loads(l)["paragraph_id"]: json.loads(l) for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}
cam = load2(f"{W}/preds/task2_dev_camelbert.jsonl")
mar = load2(f"{W}/preds/task2_dev_marbert.jsonl")
synth2 = [json.loads(l) for l in open(f"{W}/data/synth2_all.jsonl", encoding="utf-8")]

# 1. bg_example — worked gold example (Figure 1 of the paper, carries the label key)
NAMES = [("AS", "assumption"), ("AN", "anecdote"), ("ST", "statistics"),
         ("TE", "testimony"), ("CO", "common ground"), ("OT", "other")]
legend = ('<div class="lg">' + ' &nbsp;&nbsp; '.join(
    f'<span style="white-space:nowrap">{chip(l)}&nbsp;{n.replace(" ", "&nbsp;")}</span>'
    for l, n in NAMES) + '</div>')
render("bg_example", f'<div class="ar">{mark(dev[956]["text"], gold[956])}</div>{legend}')

# 2. synth_example — 4-segment synthetic paragraph (e-commerce)
s = next(r for r in synth2
         if [x["label"] for x in r["labels"]] == ["CO", "AS", "ST", "AS"] and "التجارة" in r["text"])
sp = [(x["start_offset"], x["end_offset"], x["label"]) for x in s["labels"]]
render("synth_example", f'<div class="ar">{mark(s["text"], sp)}</div>')

# 3. nested_example — ST inside AN (editorial 962)
t = dev[962]["text"]
an_s, an_e, st_s, st_e = 1, 214, 91, 159
nested = (t[:an_s]
          + f'<span class="sp" style="background:{PASTEL["AN"]}; border-bottom:2px solid {DARK["AN"]}; padding-bottom:3.5px;">'
          + t[an_s:st_s] + span_html(t[st_s:st_e], "ST") + t[st_e:an_e]
          + chip("AN") + "</span>" + t[an_e:])
render("nested_example", f'<div class="ar">{nested}</div>')

# 4. routing_example — gold vs CAM vs MAR (debate 390)
t = dev[390]["text"].replace("\n", " ")
rows = [("Gold", gold[390], ""),
        ("CAMeLBERT-mix", cam[390], "paragraph F1 0.42"),
        ("MARBERTv2", mar[390], "paragraph F1 0.94")]
body = "".join(
    f'<div class="hd">{name} <span class="sc">{sc}</span></div><div class="ar">{mark(t, sp)}</div>'
    for name, sp, sc in rows)
render("routing_example", body)

# 5. frag_example — gold span vs fragmented decode (debate 186)
t = dev[186]["text"]
lo, hi = 154, 322
frags = sorted((s, e, l) for s, e, l in mar[186] if l == "AS" and s >= 180 and e <= 322)
body = (f'<div class="hd">Gold</div><div class="ar">{mark(t[lo:hi], [(154, 316, "AS")], shift=lo)}</div>'
        f'<div class="hd">Raw decode <span class="sc">before compaction</span></div>'
        f'<div class="ar">{mark(t[lo:hi], frags, shift=lo)}</div>')
render("frag_example", body)

# 6. co_example — CO misread as AS (editorial 95)
t = dev[95]["text"]
render("co_example", f'<div class="ar">{mark(t, [(0, len(t), "CO")])}</div>')
