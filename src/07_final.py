"""Step 7: final evaluation tables, paper figures and error analysis.

Tables  (results/step7/tables.md, *.csv, *.tex)
  T1 overall metrics per model/text (mean ± std over seeds)
  T2 macro-F1 per language group
  T3 normalization effect (norm - min) per group
Figures (results/step7/fig_*.png + .pdf)
  F1 per-group macro-F1 (dot plot)          F2 normalization effect per group
  F3 confusion matrix of the best model     F4 error rate by star rating
Error analysis (results/step7/error_analysis.md)
  error rates by group / star rating / length / contrast & negation markers,
  consensus errors (wrong in every run) as a label-noise proxy,
  annotation_sample.csv for a manual label-noise check.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import GROUP_ORDER, ROOT, SEED

R = ROOT / "results"
OUT = R / "step7"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_LABEL = {"bangla_script": "Bangla script", "english": "English", "romanized": "Romanized Bangla",
               "romanized_codemixed": "Romanized + English", "mixed_script": "Mixed script"}
MODEL_LABEL = {"word_tfidf_svm": "Word TF-IDF + SVM", "char_tfidf_svm": "Char TF-IDF + SVM",
               "word_tfidf_lr": "Word TF-IDF + LR", "char_tfidf_lr": "Char TF-IDF + LR",
               "xlmr": "XLM-R-base", "banglabert": "BanglaBERT"}
# validated categorical slots 1-3 (dataviz reference palette, light mode) + distinct markers
STYLE = {"word_tfidf_svm": ("#2a78d6", "o"), "xlmr": ("#eb6834", "s"), "banglabert": ("#1baf7a", "D")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BEST = ("banglabert", "norm")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans"})

RUN_RE = re.compile(r"^(?P<model>[a-z]+)_text_(?P<text>min|norm)_seed(?P<seed>\d+)$")


def collect_runs():
    runs = []
    for step in ["step5", "step6"]:
        for d in sorted((R / step).glob("*")):
            m = RUN_RE.match(d.name)
            if d.is_dir() and m and (d / "metrics.json").exists():
                runs.append((m["model"], m["text"], int(m["seed"]), d))
    for text, step in [("min", "step3"), ("norm", "step4")]:
        for model in ["word_tfidf_lr", "word_tfidf_svm", "char_tfidf_lr", "char_tfidf_svm"]:
            runs.append((model, text, 0, R / step / model))
    return runs


def metrics_frame(runs):
    rows = []
    for model, text, seed, d in runs:
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        pc = m["per_class"]
        rows.append({"model": model, "text": text, "seed": seed, "macro_f1": m["macro_f1"],
                     "accuracy": m["accuracy"],
                     "neg_p": pc["negative"]["precision"], "neg_r": pc["negative"]["recall"], "neg_f1": pc["negative"]["f1"],
                     "pos_p": pc["positive"]["precision"], "pos_r": pc["positive"]["recall"], "pos_f1": pc["positive"]["f1"],
                     **{g: m["per_group"][g]["macro_f1"] for g in GROUP_ORDER}})
    return pd.DataFrame(rows)


def mean_std(df, cols):
    g = df.groupby(["model", "text"])
    mu, sd, n = g[cols].mean(), g[cols].std(), g["seed"].nunique()
    return mu, sd, n


def fmt_cell(mu, sd):
    return f"{mu:.3f}" if pd.isna(sd) else f"{mu:.3f} ± {sd:.3f}"


def write_tables(df):
    order = [("word_tfidf_lr", "min"), ("word_tfidf_svm", "min"), ("char_tfidf_lr", "min"), ("char_tfidf_svm", "min"),
             ("word_tfidf_svm", "norm"), ("char_tfidf_svm", "norm"),
             ("xlmr", "min"), ("xlmr", "norm"), ("banglabert", "min"), ("banglabert", "norm")]
    cols1 = ["macro_f1", "neg_p", "neg_r", "neg_f1", "pos_p", "pos_r", "pos_f1", "accuracy"]
    mu, sd, n = mean_std(df, cols1 + GROUP_ORDER)
    idx = [k for k in order if k in mu.index]
    names = [f"{MODEL_LABEL[m]} ({'normalized' if t == 'norm' else 'minimal'})" for m, t in idx]

    t1 = pd.DataFrame({c: [fmt_cell(mu.loc[k, c], sd.loc[k, c]) for k in idx] for c in cols1}, index=names)
    t1.insert(0, "seeds", [int(n.loc[k]) if k[0] in ("xlmr", "banglabert") else "-" for k in idx])
    t1.columns = ["Seeds", "Macro-F1", "Neg P", "Neg R", "Neg F1", "Pos P", "Pos R", "Pos F1", "Accuracy"]
    t2 = pd.DataFrame({GROUP_LABEL[g]: [fmt_cell(mu.loc[k, g], sd.loc[k, g]) for k in idx] for g in GROUP_ORDER},
                      index=names)
    test_n = pd.read_csv(R / "step3" / "per_group_macro_f1.csv", index_col=0).loc["n_test"]
    t2.loc["n (test)"] = [f"{int(test_n[g]):,}" for g in GROUP_ORDER]

    deltas = []
    for model in ["word_tfidf_svm", "xlmr", "banglabert"]:
        a = df[(df.model == model) & (df.text == "min")].set_index("seed")[["macro_f1"] + GROUP_ORDER]
        b = df[(df.model == model) & (df.text == "norm")].set_index("seed")[["macro_f1"] + GROUP_ORDER]
        d = (b - a).dropna()
        deltas.append((model, d.mean(), d.std(), len(d)))
    t3 = pd.DataFrame({("Overall" if c == "macro_f1" else GROUP_LABEL[c]):
                       [(f"{m_[c]:+.3f}" if pd.isna(s_[c]) else f"{m_[c]:+.3f} ± {s_[c]:.3f}") for _, m_, s_, _ in deltas]
                       for c in ["macro_f1"] + GROUP_ORDER},
                      index=[f"{MODEL_LABEL[m]} ({k} seed{'s' if k > 1 else ''})" for m, _, _, k in deltas])

    for name, t in [("T1_overall", t1), ("T2_per_group", t2), ("T3_normalization", t3)]:
        t.to_csv(OUT / f"{name}.csv")
        (OUT / f"{name}.tex").write_text(t.to_latex(escape=True), encoding="utf-8")
    md = ["# Step 7: final tables (20% test set, 23,970 reviews)\n",
          "Transformer rows: mean ± std over seeds. Macro-F1 is the primary metric; accuracy is supplementary.\n",
          "## T1. Overall results\n", t1.to_markdown(), "",
          "## T2. Macro-F1 per language group\n", t2.to_markdown(), "",
          "## T3. Effect of Banglish normalization (normalized − minimal, macro-F1)\n", t3.to_markdown(), ""]
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    return mu, sd, deltas


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_per_group(mu, sd):
    models = [("word_tfidf_svm", "min"), ("xlmr", "norm"), ("banglabert", "norm")]
    labels = {("word_tfidf_svm", "min"): "Word TF-IDF + SVM", ("xlmr", "norm"): "XLM-R (norm., 3 seeds)",
              ("banglabert", "norm"): "BanglaBERT (norm.)"}
    groups = ["ALL"] + GROUP_ORDER
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(groups))[::-1]
    offs = [0.22, 0, -0.22]
    for (k, off) in zip(models, offs):
        col, mk = STYLE[k[0]]
        vals = [mu.loc[k, "macro_f1" if g == "ALL" else g] for g in groups]
        errs = [0 if pd.isna(sd.loc[k, "macro_f1" if g == "ALL" else g]) else sd.loc[k, "macro_f1" if g == "ALL" else g]
                for g in groups]
        ax.errorbar(vals, y + off, xerr=errs, fmt=mk, ms=6, color=col, ecolor=col, elinewidth=1.2, capsize=2,
                    mec="white", mew=0.8, label=labels[k], zorder=3)
    ax.set_yticks(y, ["Overall"] + [GROUP_LABEL[g] for g in GROUP_ORDER], color=INK)
    ax.set_xlabel("Macro-F1 (test)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(y[0] - 0.5, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handletextpad=0.3)
    save(fig, "fig1_per_group_macro_f1")


def fig_norm_effect(deltas):
    groups = ["macro_f1"] + GROUP_ORDER
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    y = np.arange(len(groups))[::-1]
    offs = [0.22, 0, -0.22]
    for (model, m_, s_, k), off in zip(deltas, offs):
        col, mk = STYLE[model]
        errs = [0 if pd.isna(s_[g]) else s_[g] for g in groups]
        lab = f"{MODEL_LABEL[model]} ({k} seed{'s' if k > 1 else ''})"
        ax.errorbar([m_[g] for g in groups], y + off, xerr=errs, fmt=mk, ms=6, color=col, ecolor=col,
                    elinewidth=1.2, capsize=2, mec="white", mew=0.8, label=lab, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, ["Overall"] + [GROUP_LABEL[g] for g in GROUP_ORDER], color=INK)
    ax.set_xlabel("Δ macro-F1 (normalized − minimal text)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(y[0] - 0.5, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handletextpad=0.3)
    save(fig, "fig2_normalization_effect")


def fig_confusion(best_dir):
    m = json.loads((best_dir / "metrics.json").read_text(encoding="utf-8"))
    cm = np.array(m["confusion_matrix"]["rows_true_cols_pred"])
    pct = cm / cm.sum(1, keepdims=True)
    blues = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(pct, cmap=blues, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * pct[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (star rating)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig3_confusion_best")


def load_preds(runs):
    """uid-aligned table: text, stars, group, gold + a prediction column per run."""
    clean = pd.read_csv(ROOT / "data" / "clean.csv", usecols=["uid", "label_raw", "text_min"])
    base, cols = None, []
    for model, text, seed, d in runs:
        p = pd.read_csv(d / "predictions.csv")
        name = f"{model}|{text}|{seed}"
        p = p.rename(columns={"pred": name, "score_positive": f"score|{name}"})
        keep = ["uid", name] + ([f"score|{name}"] if f"score|{name}" in p else [])
        if base is None:
            base = p[["uid", "group", "label"] + keep[1:]]
        else:
            base = base.merge(p[keep], on="uid")
        cols.append(name)
    return base.merge(clean, on="uid"), cols


CONTRAST = r"\b(but|however|although|though|kintu|kinto|tobe|jodio)\b|কিন্তু|তবে|যদিও"
NEGATION = r"\b(not|no|never|don'?t|didn'?t|isn'?t|wasn'?t|na|nai|nei|ni|nah)\b|(?<!\S)(না|নাই|নেই|নি)(?!\S)"


def length_bucket(n):
    return "1-3" if n <= 3 else "4-7" if n <= 7 else "8-15" if n <= 15 else "16-30" if n <= 30 else "31+"


def error_analysis(runs):
    df, cols = load_preds(runs)
    best = f"{BEST[0]}|{BEST[1]}|42"
    df["err"] = df[best] != df["label"]
    df["n_tok"] = df["text_min"].str.split().str.len()
    df["len_bucket"] = pd.Categorical(df["n_tok"].map(length_bucket), ["1-3", "4-7", "8-15", "16-30", "31+"])
    low = df["text_min"].str.lower()
    df["has_contrast"] = low.str.contains(CONTRAST, regex=True)
    df["has_negation"] = low.str.contains(NEGATION, regex=True)
    wrong_all = np.all([df[c] != df["label"] for c in cols], axis=0)
    df["consensus_error"] = wrong_all

    def rate(by):
        t = df.groupby(by, observed=True).agg(n=("err", "size"), errors=("err", "sum"))
        t["error_%"] = (100 * t["errors"] / t["n"]).round(1)
        return t

    by_group = rate("group").reindex(GROUP_ORDER)
    by_group.index = [GROUP_LABEL[g] for g in by_group.index]
    by_star = rate("label_raw")
    by_star.index.name = "stars"
    by_len = rate("len_bucket")
    by_contrast = rate("has_contrast")
    by_neg = rate("has_negation")
    by_gold = rate("label")

    # F4 error rate by stars
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = by_star.index.astype(int)
    ax.bar(x, by_star["error_%"], width=0.6, color="#2a78d6", edgecolor="white", linewidth=2, zorder=3)
    for xi, v in zip(x, by_star["error_%"]):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{i}★" for i in x])
    ax.set_ylabel("Error rate of best model (%)")
    ax.set_xlabel("Star rating (1–3 = negative, 4–5 = positive)")
    ax.axvline(3.5, color=INK2, lw=1, ls="--")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylim(0, max(by_star["error_%"]) * 1.18)
    save(fig, "fig4_error_by_stars")

    cons = df[df["consensus_error"]]
    cons_star = cons["label_raw"].value_counts().sort_index()
    cons_group = cons["group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)

    # confident errors of the best model
    sc = df[f"score|{best}"]
    conf_fp = df[(df["label"] == "negative") & (sc > 0.95)]
    conf_fn = df[(df["label"] == "positive") & (sc < 0.05)]

    rng = np.random.default_rng(SEED)
    samp = cons.sample(min(200, len(cons)), random_state=SEED)[["uid", "group", "label_raw", "label", "text_min"]]
    samp = samp.assign(model_pred=np.where(samp["label"] == "positive", "negative", "positive"),
                       annot_true_sentiment="", annot_label_is_wrong="", annot_note="")
    samp.to_csv(OUT / "annotation_sample.csv", index=False, encoding="utf-8-sig")

    ex_lines = []
    for g in GROUP_ORDER:
        sub = df[(df["group"] == g) & df["err"]]
        ex_lines.append(f"\n### {GROUP_LABEL[g]}\n")
        for _, r in sub.sample(min(6, len(sub)), random_state=SEED).iterrows():
            ex_lines.append(f"- [{r.label_raw}★ gold={r.label}, pred={r[best]}] {r.text_min[:160]}")
    cons_lines = [f"- [{r.label_raw}★ gold={r.label}] {r.text_min[:160]}"
                  for _, r in cons.sample(min(25, len(cons)), random_state=SEED + 1).iterrows()]

    df[["uid", "group", "label_raw", "label", best, "err", "consensus_error", "has_contrast", "has_negation",
        "n_tok", "text_min"]].to_csv(OUT / "best_model_test_predictions.csv", index=False, encoding="utf-8-sig")

    n_runs = len(cols)
    rep = [
        f"# Step 7: error analysis — best model = BanglaBERT (normalized), test set n={len(df):,}\n",
        f"Best-model errors: {int(df['err'].sum()):,} ({100 * df['err'].mean():.1f}%).\n",
        "## Error rate by gold class\n", by_gold.to_markdown(), "",
        "## Error rate by language group\n", by_group.to_markdown(), "",
        "## Error rate by star rating\n", by_star.to_markdown(), "",
        "## Error rate by review length (whitespace tokens)\n", by_len.to_markdown(), "",
        "## Error rate with / without a contrast marker (but, kintu, tobe, কিন্তু, তবে …)\n", by_contrast.to_markdown(), "",
        "## Error rate with / without a standalone negation token (not, na, nai, না, নাই, নেই …)\n", by_neg.to_markdown(), "",
        f"## Consensus errors (wrong in all {n_runs} runs: 8 TF-IDF, 6 XLM-R, 2 BanglaBERT)\n",
        f"{len(cons):,} reviews ({100 * len(cons) / len(df):.1f}% of test; "
        f"{100 * len(cons) / df['err'].sum():.1f}% of best-model errors). "
        "These are the strongest candidates for label noise (rating disagrees with text).\n",
        "By star rating:\n", cons_star.rename("n").to_frame().to_markdown(), "",
        "By group:\n", cons_group.rename("n").to_frame().to_markdown(), "",
        f"Confident best-model errors: gold negative but P(pos) > 0.95: {len(conf_fp):,}; "
        f"gold positive but P(pos) < 0.05: {len(conf_fn):,}.\n",
        "`annotation_sample.csv` has 200 random consensus errors with empty columns for a manual check.\n",
        "## Random consensus errors\n", *cons_lines, "",
        "## Random best-model errors per group", *ex_lines, "",
    ]
    (OUT / "error_analysis.md").write_text("\n".join(rep), encoding="utf-8")
    return df, cons


def main():
    runs = collect_runs()
    df = metrics_frame(runs)
    df.round(4).to_csv(OUT / "all_runs_metrics.csv", index=False)
    mu, sd, deltas = write_tables(df)
    fig_per_group(mu, sd)
    fig_norm_effect(deltas)
    fig_confusion(R / "step6" / "banglabert_text_norm_seed42")
    error_analysis(runs)
    print((OUT / "tables.md").read_text(encoding="utf-8"))
    print((OUT / "error_analysis.md").read_text(encoding="utf-8").split("## Random consensus")[0])


if __name__ == "__main__":
    main()
