# Full-width system diagram (paper, Appendix A).
# Data and pretraining across the top, then the two task pipelines side by
# side. Solid = Closed track, dashed blue = Open-track additions.
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
})

INK, GRAY, BLUE, FILL = "#1a1a1a", "#9a9a9a", "#0072B2", "#f4f4f4"
FS = 8.3

fig, ax = plt.subplots(figsize=(6.3, 3.05))
ax.set_xlim(-7, 207)
ax.set_ylim(-14, 92)
ax.axis("off")


def box(x, y, w, h, text, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.4",
                                facecolor="white" if dashed else FILL,
                                edgecolor=BLUE if dashed else GRAY,
                                linestyle=(0, (3.5, 2.5)) if dashed else "-",
                                linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS, color=INK, linespacing=1.3, zorder=3)


def arrow(p1, p2, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7,
                                 linestyle=(0, (3.5, 2.5)) if dashed else "-",
                                 linewidth=0.9, color=BLUE if dashed else GRAY,
                                 shrinkA=0.5, shrinkB=0.5, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def band(y, label, x=0, ha="left"):
    ax.text(x, y, label, fontsize=8.8, fontweight="bold", color=INK,
            ha=ha, va="center")


# ---------------- data band (full width) ----------------
band(86, "Data and pretraining")
box(0, 61, 60, 16, "612 train paragraphs\n(+217 dev in the eval phase)")
box(68, 61, 60, 16, "back-translation $\\times$1\nrare-class BT $\\times$3")
box(136, 61, 64, 16, "LLM-synthetic\n291 T1 / 158 T2 (Open)", dashed=True)
box(36, 42, 128, 13, "DAPT-v2 pretraining (MLM on 1,042 paragraphs), then task training")
arrow((30, 61), (70, 55.5))
arrow((98, 61), (100, 55.5))
arrow((168, 61), (130, 55.5), dashed=True)

# ---------------- task 2 (left) ----------------
band(36, "Task 2 — argumentative spans")
box(0, 14, 44, 15, "CAMeLBERT-mix $\\times$8\n(editorials)")
box(50, 14, 44, 15, "MARBERTv2 $\\times$8\n(debates)")
arrow((70, 42), (24, 29.5))
arrow((100, 42), (70, 29.5))
box(0, -2, 60, 13, "route by genre, then\nspan compaction")
box(66, -1, 28, 11, "char blend\n(Open)", dashed=True)
arrow((24, 14), (28, 11.5))
arrow((70, 14), (44, 11.5))
arrow((66, 4.5), (60.5, 4.5), dashed=True)

# ---------------- task 1 (right) ----------------
box(106, 14, 44, 15, "Qwen3-32B QLoRA\n$\\times$3 (editorials)")
box(156, 14, 44, 15, "encoder ensemble\n(debates)")
arrow((120, 42), (128, 29.5))
arrow((140, 42), (176, 29.5))
box(106, -2, 60, 13, "route by genre, then\nrobust thresholds")
box(172, -1, 28, 11, "span fusion\n(Open)", dashed=True)
arrow((128, 14), (130, 11.5))
arrow((176, 14), (150, 11.5))
arrow((172, 4.5), (166.5, 4.5), dashed=True)

arrow((30, -2), (30, -6.5))
arrow((136, -2), (136, -6.5))
ax.text(30, -9.5, "labelled spans", fontsize=FS+0.2, style="italic",
        color=INK, ha="center", va="center")
ax.text(136, -9.5, "paragraph labels", fontsize=FS+0.2, style="italic",
        color=INK, ha="center", va="center")
band(36, "Task 1 — paragraph labels", x=200, ha="right")

fig.tight_layout(pad=0.15)
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(FIGS, exist_ok=True)
fig.savefig(os.path.join(FIGS, "pipeline.pdf"))
print("saved")
