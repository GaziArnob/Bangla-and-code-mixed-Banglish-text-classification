"""Step 3 (and 4): TF-IDF baselines.

Features x classifiers:
  word TF-IDF (1-2 grams)        x  Logistic Regression, Linear SVM
  char TF-IDF (2-5 grams, char_wb) x  Logistic Regression, Linear SVM

class_weight="balanced" for the 82/18 imbalance. C is tuned on the validation
set by macro-F1; the model is trained on the train part only and scored once
on the 20% test set.

Usage:
  python src/03_baselines.py --text text_min   (step 3)
  python src/03_baselines.py --text text_norm  (step 4)
"""
import argparse
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from common import GROUP_ORDER, ROOT, SEED, evaluate, load_data, macro_f1

# Tokens = runs of anything except whitespace/punctuation. sklearn's default \w-based
# pattern splits Bengali words at vowel signs, so it is not used.
WORD_TOKEN = r"[^\s.,!?;:।৷'\"()\[\]{}\-~*/|]+"

FEATURES = {
    "word": lambda: TfidfVectorizer(token_pattern=WORD_TOKEN, lowercase=True, ngram_range=(1, 2),
                                    min_df=2, max_df=0.9, sublinear_tf=True),
    "char": lambda: TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(2, 5),
                                    min_df=3, max_df=0.9, sublinear_tf=True, max_features=500_000),
}
CLASSIFIERS = {
    "lr": (lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                        solver="liblinear", random_state=SEED),
           [0.5, 1, 2, 4, 8, 16]),
    "svm": (lambda C: LinearSVC(C=C, class_weight="balanced", random_state=SEED),
            [0.03, 0.1, 0.3, 1]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="text_min", choices=["text_min", "text_norm"])
    args = ap.parse_args()

    step = "step3" if args.text == "text_min" else "step4"
    out = ROOT / "results" / step
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.text)
    tr, va, te = data["train"], data["val"], data["test"]
    print(f"text={args.text}  train={len(tr):,} val={len(va):,} test={len(te):,}")

    summary, groups, tuning = [], [], []
    for feat_name, make_vec in FEATURES.items():
        t0 = time.time()
        vec = make_vec()
        Xtr = vec.fit_transform(tr["text"])
        Xva, Xte = vec.transform(va["text"]), vec.transform(te["text"])
        print(f"[{feat_name}] {Xtr.shape[1]:,} features ({time.time() - t0:.0f}s)")

        for clf_name, (make_clf, grid) in CLASSIFIERS.items():
            name = f"{feat_name}_tfidf_{clf_name}"
            best = (-1, None)
            for C in grid:
                clf = make_clf(C).fit(Xtr, tr["y"])
                f = macro_f1(va["y"], clf.predict(Xva))
                tuning.append({"model": name, "C": C, "val_macro_f1": round(f, 4)})
                print(f"  {name} C={C:<5} val macro-F1={f:.4f}")
                if f > best[0]:
                    best = (f, C, clf)
            val_f1, C, clf = best
            scores = clf.decision_function(Xte)
            m = evaluate(te, clf.predict(Xte), out / name, f"{name} ({args.text})", scores=scores)
            summary.append({
                "model": name, "text": args.text, "best_C": C, "n_features": Xtr.shape[1],
                "val_macro_f1": round(val_f1, 4),
                "test_macro_f1": round(m["macro_f1"], 4),
                "test_accuracy": round(m["accuracy"], 4),
                "neg_precision": round(m["per_class"]["negative"]["precision"], 4),
                "neg_recall": round(m["per_class"]["negative"]["recall"], 4),
                "neg_f1": round(m["per_class"]["negative"]["f1"], 4),
                "pos_precision": round(m["per_class"]["positive"]["precision"], 4),
                "pos_recall": round(m["per_class"]["positive"]["recall"], 4),
                "pos_f1": round(m["per_class"]["positive"]["f1"], 4),
            })
            groups.append({"model": name, **{g: round(m["per_group"][g]["macro_f1"], 4) for g in GROUP_ORDER}})
            print(f"  -> best C={C}: TEST macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")

    summary = pd.DataFrame(summary).set_index("model")
    groups = pd.DataFrame(groups).set_index("model")
    n_test = {g: int((te["group"] == g).sum()) for g in GROUP_ORDER}
    groups.loc["n_test"] = n_test
    summary.to_csv(out / "summary.csv")
    groups.to_csv(out / "per_group_macro_f1.csv")
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    allrows = pd.concat([tr, va, te], ignore_index=True)
    split_stats = pd.crosstab(allrows["group"], allrows["split80"])
    split_stats = split_stats[["train", "val", "test"]]
    report = [
        f"# {step}: TF-IDF baselines on `{args.text}`\n",
        "Split: 80% train / 20% test (stratified by label x group, seed 42); "
        "10% of train held out as validation for choosing C.\n",
        split_stats.to_markdown(), "",
        "## Test results (20% test set)\n", summary.drop(columns=["text"]).to_markdown(), "",
        "## Test macro-F1 per language group\n", groups.to_markdown(), "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
