import json, subprocess, re, os

W = "/home/amal/Desktop/daleel2026"
OUT = f"{W}/paper/arabic"

dev = {json.loads(l)["paragraph_id"]: json.loads(l) for l in open(f"{W}/data/dev_in.jsonl", encoding="utf-8")}

COL = {"AS": "#cfe0f5", "AN": "#d5efce", "ST": "#ffdfb0", "TE": "#f6d5d5",
       "CO": "#ecd6f0", "OT": "#e4e4e4"}
EDGE = {"AN": "#5a9e4b"}

CSS = """
@page { size: 95mm 160mm; margin: 2mm; }
body { width: 86mm; margin: 0; font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 10pt; }
.ar { direction: rtl; text-align: right; font-family: 'Noto Naskh Arabic'; font-size: 10.5pt; line-height: 1.8; }
.tag { font-family: 'Times New Roman','Nimbus Roman',serif; font-size: 7pt; font-weight: bold;
       vertical-align: super; padding: 0 1px; direction: ltr; unicode-bidi: embed; }
.sp { padding: 1px 1px; border-radius: 2px; }
.cap { font-size: 8.5pt; margin: 3pt 0 1pt 0; }
"""

def mark(text, spans, shift=0):
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
    os.remove(pdf)
    print(name, f"{w:.0f}x{h:.0f}pt")

# ---- nested spans: pid 962, gold AN [1,214) containing ST [91,159) ----
t = dev[962]["text"]
an_s, an_e, st_s, st_e = 1, 214, 91, 159
nested = (
    t[:an_s]
    + f'<span class="sp" style="background:{COL["AN"]}; border-bottom:2.5px solid {EDGE["AN"]};">'
    + t[an_s:st_s]
    + f'<span class="sp" style="background:{COL["ST"]}">{t[st_s:st_e]}<span class="tag">ST</span></span>'
    + t[st_e:an_e]
    + '<span class="tag">AN</span></span>'
    + t[an_e:]
)
render("nested_example", f'<div class="ar">{nested}</div>')

# ---- routing: pid 390, gold vs CAM vs MAR ----
t = dev[390]["text"].replace("\n", " ")
gold = [(1,39,"TE"), (42,97,"TE"), (137,177,"AS"), (186,243,"AS")]
camp = [(2,25,"TE"), (28,31,"TE"), (89,92,"AS"), (94,177,"AS"), (178,202,"OT"), (218,237,"OT")]
marp = [(0,39,"TE"), (42,97,"TE"), (98,243,"AS")]
body = (
    f'<div class="cap"><b>Gold:</b></div><div class="ar">{mark(t, gold)}</div>'
    f'<div class="cap"><b>CAMeLBERT-mix (paragraph F1 0.42):</b></div><div class="ar">{mark(t, camp)}</div>'
    f'<div class="cap"><b>MARBERTv2 (paragraph F1 0.94):</b></div><div class="ar">{mark(t, marp)}</div>'
)
render("routing_example", body)
