# System pipeline diagram (paper appendix). Solid = Closed track,
# dashed blue = Open-track additions.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
})

INK, GRAY, BLUE, LIGHT = "#1a1a1a", "#8a8a8a", "#0072B2", "#f2f2f2"
FS = 6.8

fig, ax = plt.subplots(figsize=(3.03, 3.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 152)
ax.axis("off")

def box(x, y, w, h, text, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                                facecolor=LIGHT,
                                edgecolor=BLUE if dashed else GRAY,
                                linestyle="--" if dashed else "-", linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS, color=INK, linespacing=1.25)

def arrow(x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=7,
                                 linestyle=(0, (4, 3)) if dashed else "-",
                                 linewidth=0.8, color=BLUE if dashed else GRAY,
                                 shrinkA=1, shrinkB=1))

def band(y, label):
    ax.text(1, y, label, fontsize=7.2, fontweight="bold", color=INK,
            ha="left", va="center")

# ---- data band -------------------------------------------------------------
band(148, "Data and pretraining")
box(2, 128, 45, 15, "612 train (+217 dev, eval);\nBT ×1; rare-class BT ×3")
box(52, 128, 46, 15, "LLM-synthetic 291 T1 /\n158 T2 (Open; training only)", dashed=True)
box(27, 108, 46, 13, "DAPT-v2 encoders\n(MLM on 1,042 paragraphs)")
arrow(24, 128, 40, 122)

# ---- task 2 band -----------------------------------------------------------
band(102, "Task 2 — spans")
box(2, 82, 45, 14, "CAMeLBERT-mix ×8\n(editorials)")
box(52, 82, 46, 14, "MARBERTv2 ×8\n(debates)")
arrow(40, 108, 26, 97)
arrow(60, 108, 73, 97)
box(2, 62, 60, 13, "route by provided genre;\nper-domain thresholds")
box(67, 62, 31, 13, "char-level\nblend (Open)", dashed=True)
arrow(24, 82, 27, 76)
arrow(75, 82, 40, 76)
arrow(67, 68, 63, 68, dashed=True)
box(2, 44, 96, 11, "span compaction ($G_{ed}$=400, $G_{db}$=5, $L$=25)  →  labelled spans")
arrow(32, 62, 32, 56)

# ---- task 1 band -----------------------------------------------------------
band(38, "Task 1 — paragraph labels")
box(2, 18, 45, 14, "Qwen3-32B QLoRA ×3,\nlabel-set union (editorials)")
box(52, 18, 46, 14, "encoder ensemble, robust\nmedian thresholds (debates)")
box(2, 2, 60, 11, "route by genre  →  paragraph labels")
box(67, 2, 31, 11, "span fusion\n(debates, Open)", dashed=True)
arrow(24, 18, 28, 14)
arrow(75, 18, 40, 14)
arrow(67, 7, 63, 7, dashed=True)
arrow(96, 44, 95, 14, dashed=True)   # compacted spans feed the Open fusion

fig.tight_layout(pad=0.1)
fig.savefig("/home/amal/Desktop/daleel2026/paper/figs/pipeline.pdf")
print("saved")
