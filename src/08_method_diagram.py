"""Methodology diagram (left to right) for the paper.
Writes results/figures/fig0_methodology.{pdf,png,svg}."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8984"
FILL = "#f6f5f2"
ACCENT = {"data": "#2a78d6", "prep": "#eb6834", "model": "#1baf7a", "eval": "#4a3aa7"}

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(22, 5.8))
ax.set_xlim(0, 43)
ax.set_ylim(2.1, 12.4)
ax.axis("off")


def box(x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=FILL, ec=EDGE, lw=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle="square,pad=0",
                                fc=ACCENT[kind], ec="none", zorder=3))
    ax.text(x + 0.22, y + h - 0.45, title, fontsize=10.5, fontweight="bold", color=INK, va="top", zorder=4)
    ax.text(x + 0.22, y + h - 1.05, "\n".join(lines), fontsize=8.6, color=INK2, va="top",
            linespacing=1.45, zorder=4)
    return (x, y, w, h)


def arrow(a, b, side_a="r", side_b="l", rad=0.0):
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    p1 = (ax_ + aw, ay + ah / 2) if side_a == "r" else (ax_ + aw / 2, ay)
    p2 = (bx, by + bh / 2) if side_b == "l" else (bx + bw / 2, by + bh)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=INK2,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def header(x, w, n, text):
    ax.text(x + w / 2, 11.95, f"{n}", fontsize=10, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.28", fc=INK, ec="none"))
    ax.text(x + w / 2, 11.35, text, fontsize=10, fontweight="bold", color=INK, ha="center", va="center")


W = 5.4
X = [0.2 + i * 6.1 for i in range(7)]
titles = ["Dataset", "Labels & groups", "Preprocessing", "Data split",
          "Classifiers", "Evaluation", "Analysis"]
for i, (x, t) in enumerate(zip(X, titles), 1):
    header(x, W, i, t)

b1 = box(X[0], 3.2, W, 6.8, "BanglishRev", [
    "Daraz e-commerce reviews", "119,859 reviews", "Bangla, English, Romanized,", "and code-mixed text", "",
    "− 11 corrupted rows (#NAME?)", "→ 119,848 reviews"], "data")

b2 = box(X[1], 3.2, W, 6.8, "Star rating → label", [
    "1–3★ → negative (17.7%)", "4–5★ → positive (82.3%)", "",
    "Script-based groups:", "• Bangla script      36.1%", "• English              37.9%",
    "• Romanized Bangla  5.7%", "• Romanized + Eng. 15.5%", "• Mixed script        4.7%"], "data")

b3a = box(X[2], 6.75, W, 3.25, "A. Minimal cleaning", [
    "NFC, strip HTML", "URL → <URL>,  @user → <USER>", "keep hashtags, emojis,", "negation, English words"], "prep")
b3b = box(X[2], 3.2, W, 3.25, "B. Normalized Banglish", [
    "A + lowercase, Bangla digits,", "elongation (sooo → soo),", "138 spelling variants → 67 forms",
    "(vlo, bhalo → valo)"], "prep")

b4 = box(X[3], 3.2, W, 6.8, "Stratified 80 / 20", [
    "stratified by label × group", "seed 42, same for all models", "",
    "Train      86,290  (72%)", "Validation  9,588  (8%)", "Test       23,970  (20%)", "",
    "validation: tuning &", "early stopping only"], "data")

b5a = box(X[4], 6.75, W, 3.25, "TF-IDF baselines", [
    "word 1–2 or char 2–5-grams", "× LR or linear SVM", "class_weight = balanced",
    "C tuned on validation"], "model")
b5b = box(X[4], 3.2, W, 3.25, "Fine-tuned transformers", [
    "XLM-R-base (3 seeds)", "BanglaBERT (1 seed)", "own tokenizer, ≤ 128 tokens",
    "weighted CE, best epoch on val."], "model")

b6 = box(X[5], 3.2, W, 6.8, "20% test set", [
    "Macro-F1 (primary)", "per-class P / R / F1", "confusion matrix", "accuracy (supplementary)", "",
    "per language group", "mean ± std over seeds", "paired bootstrap", "(1,000 resamples)"], "eval")

b7 = box(X[6], 3.2, W, 6.8, "What we examine", [
    "Model comparison", "per language group", "", "Normalization effect", "(B − A) per group", "",
    "Tokenizer fertility", "", "Error analysis &", "rating-label noise"], "eval")
arrow(b1, b2)
arrow(b2, b3a, rad=-0.15)
arrow(b2, b3b, rad=0.15)
arrow(b3a, b4, rad=0.15)
arrow(b3b, b4, rad=-0.15)
arrow(b4, b5a, rad=-0.15)
arrow(b4, b5b, rad=0.15)
arrow(b5a, b6, rad=0.15)
arrow(b5b, b6, rad=-0.15)
arrow(b6, b7)

ax.text(X[2] + W / 2, 2.6, "every model is trained and tested on both A and B",
        fontsize=8.5, color=INK2, ha="center", style="italic")
ax.text(X[4] + W / 2, 2.6, "same split, same metrics for all runs", fontsize=8.5, color=INK2,
        ha="center", style="italic")

for ext in ["pdf", "png", "svg"]:
    fig.savefig(OUT / f"fig0_methodology.{ext}", bbox_inches="tight", facecolor="white")
print("saved", OUT)
