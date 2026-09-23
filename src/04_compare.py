"""Step 4: minimal vs normalized text, same models and split.

Paired bootstrap over the test set (same resampled rows for both versions):
delta = macro-F1(text_norm) - macro-F1(text_min), with 95% CI and a two-sided
p-value, overall and per language group.

Usage: python src/04_compare.py [--a DIR_A --b DIR_B --out DIR]
Defaults compare results/step3 (text_min) with results/step4 (text_norm).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import GROUP_ORDER, ROOT, SEED

N_BOOT = 1000


def load(d, model):
    p = pd.read_csv(d / model / "predictions.csv")
    return p.sort_values("uid").reset_index(drop=True)


def boot_delta(y, pa, pb, rng):
    n = len(y)
    base = f1_score(y, pb, average="macro") - f1_score(y, pa, average="macro")
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        deltas[i] = f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return base, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "results" / "step3"))
    ap.add_argument("--b", default=str(ROOT / "results" / "step4"))
    ap.add_argument("--out", default=str(ROOT / "results" / "step4"))
    ap.add_argument("--models", nargs="*", default=["word_tfidf_lr", "word_tfidf_svm", "char_tfidf_lr", "char_tfidf_svm"])
    args = ap.parse_args()
    from pathlib import Path
    A, B, OUT = Path(args.a), Path(args.b), Path(args.out)

    rng = np.random.default_rng(SEED)
    rows = []
    for model in args.models:
        a, b = load(A, model), load(B, model)
        assert (a["uid"] == b["uid"]).all()
        y = (a["label"] == "positive").to_numpy().astype(int)
        pa = (a["pred"] == "positive").to_numpy().astype(int)
        pb = (b["pred"] == "positive").to_numpy().astype(int)
        for g in ["ALL"] + GROUP_ORDER:
            m = np.ones(len(y), bool) if g == "ALL" else (a["group"] == g).to_numpy()
            fa = f1_score(y[m], pa[m], average="macro")
            fb = f1_score(y[m], pb[m], average="macro")
            d, lo, hi, p = boot_delta(y[m], pa[m], pb[m], rng)
            rows.append({"model": model, "group": g, "n": int(m.sum()),
                         "f1_min": round(fa, 4), "f1_norm": round(fb, 4),
                         "delta": round(d, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                         "p": round(p, 3), "significant": p < 0.05})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "min_vs_norm.csv", index=False)

    wide = res.pivot(index="model", columns="group", values="delta")[["ALL"] + GROUP_ORDER]
    sig = res.pivot(index="model", columns="group", values="significant")[["ALL"] + GROUP_ORDER]
    marked = wide.map(lambda v: f"{v:+.4f}") + sig.map(lambda s: " *" if s else "")
    report = [
        "# Step 4: minimal vs normalized text (TF-IDF baselines)\n",
        f"delta = macro-F1(text_norm) - macro-F1(text_min) on the 20% test set. "
        f"Paired bootstrap, {N_BOOT} resamples; * = p < 0.05.\n",
        marked.to_markdown(), "",
        "## Full table\n", res.set_index(["model", "group"]).to_markdown(), "",
    ]
    (OUT / "min_vs_norm.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
