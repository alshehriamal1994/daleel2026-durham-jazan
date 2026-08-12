# Full-width system diagram (paper, Appendix A).
# Data and pretraining across the top, then the two task pipelines side by
# side. Solid = Closed track, dashed blue = Open-track additions.
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
FS = 7.2

fig, ax = plt.subplots(figsize=(6.3, 2.72))
ax.set_xlim(-7, 207)
ax.set_ylim(-13, 92)
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
    ax.text(x, y, label, fontsize=7.6, fontweight="bold", color=INK,
            ha=ha, va="center")


# ---------------- data band (full width) ----------------
band(85, "Data and pretraining")
box(0, 62, 60, 14, "612 train paragraphs\n(+217 dev in the eval phase)")
box(68, 62, 60, 14, "back-translation $\\times$1\nrare-class BT $\\times$3")
box(136, 62, 64, 14, "LLM-synthetic\n291 T1 / 158 T2 (Open)", dashed=True)
box(40, 44, 120, 12, "DAPT-v2 pretraining (MLM on 1,042 paragraphs), then task training")
arrow((30, 62), (70, 56.5))
arrow((98, 62), (100, 56.5))
arrow((168, 62), (130, 56.5), dashed=True)

# ---------------- task 2 (left) ----------------
band(38, "Task 2 — argumentative spans")
box(0, 16, 44, 13, "CAMeLBERT-mix $\\times$8\n(editorials)")
box(50, 16, 44, 13, "MARBERTv2 $\\times$8\n(debates)")
arrow((70, 44), (24, 29.5))
arrow((100, 44), (70, 29.5))
box(0, 0, 60, 12, "route by genre, then\nspan compaction")
box(66, 2, 28, 9, "char blend\n(Open)", dashed=True)
arrow((24, 16), (28, 12.5))
arrow((70, 16), (44, 12.5))
arrow((66, 6.5), (60.5, 6.5), dashed=True)

# ---------------- task 1 (right) ----------------
box(106, 16, 44, 13, "Qwen3-32B QLoRA\n$\\times$3 (editorials)")
box(156, 16, 44, 13, "encoder ensemble\n(debates)")
arrow((120, 44), (128, 29.5))
arrow((140, 44), (176, 29.5))
box(106, 0, 60, 12, "route by genre, then\nrobust thresholds")
box(172, 2, 28, 9, "span fusion\n(Open)", dashed=True)
arrow((128, 16), (130, 12.5))
arrow((176, 16), (150, 12.5))
arrow((172, 6.5), (166.5, 6.5), dashed=True)

arrow((30, 0), (30, -6))
arrow((136, 0), (136, -6))
ax.text(30, -9.5, "labelled spans", fontsize=FS, style="italic",
        color=INK, ha="center", va="center")
ax.text(136, -9.5, "paragraph labels", fontsize=FS, style="italic",
        color=INK, ha="center", va="center")
band(38, "Task 1 — paragraph labels", x=200, ha="right")

fig.tight_layout(pad=0.15)
fig.savefig("/home/amal/Desktop/daleel2026/paper/figs/pipeline.pdf")
print("saved")
