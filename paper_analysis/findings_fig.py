# Two-panel figure for the paper's two post-submission findings (Appendix C/D).
# Left: per-label editorial advantage against the editorial share of that
# label's training spans. Right: threshold recalibration measured in sample
# and on held-out data over 500 random splits.
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#8a8a8a"
W = "/home/amal/Desktop/daleel2026"
LABELS = ["AS", "AN", "ST", "TE", "CO", "OT"]

# ---------------- left panel data (from genre_gap.py) ----------------
SHARE = [29, 57, 79, 33, 50, 5]
DIFF = [-0.136, +0.079, +0.229, +0.012, -0.143, -0.602]
OFFS = {"AS": (0, 9), "AN": (0, 9), "ST": (0, 9), "TE": (0, 9),
        "CO": (0, -14), "OT": (7, 6)}

# ---------------- right panel data (recompute, fast) ----------------
BASE = [.425, .25, .2, .3, .125, .275]
GRID = np.arange(0.05, 0.96, 0.025)
P = np.load(f"{W}/oof/t1_recal_oof_closed.npy")
rows = [json.loads(l) for l in open(f"{W}/data/train_task_1.jsonl", encoding="utf-8")]
rows += [json.loads(l) for l in open(f"{W}/data/dev_task_1_ref.jsonl", encoding="utf-8")]
Y = np.zeros((len(rows), 6))
for i, r in enumerate(rows):
    for l in r["labels"]:
        Y[i, LABELS.index(l)] = 1


def cf1(p, g, t):
    pr = p >= t
    tp = np.sum(pr & g); fp = np.sum(pr & ~g); fn = np.sum(~pr & g)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def macro(idx, ths):
    return float(np.mean([cf1(P[idx, k], Y[idx, k] > 0, ths[k]) for k in range(6)]))


rng = np.random.RandomState(0)
ins, out = [], []
for _ in range(500):
    perm = rng.permutation(len(P))
    A, B = perm[:len(P) // 4], perm[len(P) // 4:]
    ths = [max(GRID, key=lambda t: cf1(P[A, k], Y[A, k] > 0, t)) for k in range(6)]
    ins.append(macro(A, ths) - macro(A, BASE))
    out.append(macro(B, ths) - macro(B, BASE))
ins, out = np.array(ins), np.array(out)

# ---------------- draw ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.35))

ax1.axhline(0, lw=0.6, color="#cccccc", zorder=1)
m, b = np.polyfit(SHARE, DIFF, 1)
xs = np.array([0, 85])
ax1.plot(xs, m * xs + b, ls=(0, (4, 3)), lw=0.8, color=GRAY, zorder=2)
for lab, s, d in zip(LABELS, SHARE, DIFF):
    col = VERM if d < -0.05 else BLUE
    ax1.scatter(s, d, s=34, color=col, edgecolors="white", linewidths=0.7, zorder=3)
    dx, dy = OFFS[lab]
    ax1.annotate(lab, (s, d), textcoords="offset points", xytext=(dx, dy),
                 fontsize=7.5, ha="center", color="#1a1a1a", zorder=4)
ax1.set_xlim(-4, 88)
ax1.set_ylim(-0.72, 0.33)
ax1.set_xlabel("editorial share of the label's training spans (%)")
ax1.set_ylabel("editorial advantage (F1)")
ax1.set_title("Where a label is learned decides where it works", fontsize=8, pad=4)
ax1.text(52, -0.55, f"$r = {np.corrcoef(SHARE, DIFF)[0,1]:.2f}$", fontsize=7.5, color=GRAY)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(length=2.5)

bins = np.linspace(-0.06, 0.08, 40)
ax2.hist(ins, bins=bins, color=BLUE, alpha=0.8, lw=0)
ax2.hist(out, bins=bins, color=VERM, alpha=0.8, lw=0)
ax2.axvline(0, lw=0.8, color="#555555")
ax2.set_xlabel("change in macro-F1 after recalibration")
ax2.set_ylabel("splits")
ax2.set_ylim(0, 100)
ax2.set_title("Threshold gains do not leave the tuning set", fontsize=8, pad=6)
ax2.annotate("on held-out\ndata", (-0.021, 74), color=VERM, fontsize=7.5,
             ha="center", va="bottom", linespacing=1.25)
ax2.annotate("on the\ntuning half", (0.036, 70), color=BLUE, fontsize=7.5,
             ha="center", va="bottom", linespacing=1.25)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(length=2.5)

fig.tight_layout(pad=0.3, w_pad=1.6)
fig.savefig(f"{W}/paper/figs/findings.pdf")
print(f"saved. in-sample {ins.mean():+.4f}, held-out {out.mean():+.4f}, "
      f"held-out<=0 {100*np.mean(out<=0):.0f}%")
