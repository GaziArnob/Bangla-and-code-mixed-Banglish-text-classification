"""Step 5 summary: XLM-R (min / norm) vs the TF-IDF baselines.

Paired bootstrap (same resampled test rows) for:
  - XLM-R text_norm vs XLM-R text_min        (normalization effect for a transformer)
  - XLM-R text_min  vs word TF-IDF SVM min   (transformer vs best baseline)
  - XLM-R text_norm vs word TF-IDF SVM norm
Writes results/step5/report.md and comparisons.csv.
"""
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import GROUP_ORDER, ROOT, SEED

spec = importlib.util.spec_from_file_location("cmp", ROOT / "src" / "04_compare.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

R = ROOT / "results"
RUNS = {
    "word_tfidf_svm (min)": R / "step3" / "word_tfidf_svm",
    "word_tfidf_svm (norm)": R / "step4" / "word_tfidf_svm",
    "char_tfidf_svm (min)": R / "step3" / "char_tfidf_svm",
    "xlmr (min)": R / "step5" / "xlmr_text_min_seed42",
    "xlmr (norm)": R / "step5" / "xlmr_text_norm_seed42",
}
PAIRS = [
    ("xlmr (min)", "xlmr (norm)"),
    ("word_tfidf_svm (min)", "xlmr (min)"),
    ("word_tfidf_svm (norm)", "xlmr (norm)"),
]


def load_pred(d):
    p = pd.read_csv(d / "predictions.csv").sort_values("uid").reset_index(drop=True)
    return p


def main():
    out = R / "step5"
    rows, groups = [], []
    for name, d in RUNS.items():
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        pc = m["per_class"]
        rows.append({"model": name, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "neg_P": pc["negative"]["precision"], "neg_R": pc["negative"]["recall"],
                     "neg_F1": pc["negative"]["f1"], "pos_F1": pc["positive"]["f1"]})
        groups.append({"model": name, **{g: m["per_group"][g]["macro_f1"] for g in GROUP_ORDER}})
    overall = pd.DataFrame(rows).set_index("model").round(4)
    per_group = pd.DataFrame(groups).set_index("model").round(4)

    rng = np.random.default_rng(SEED)
    comps = []
    for a_name, b_name in PAIRS:
        a, b = load_pred(RUNS[a_name]), load_pred(RUNS[b_name])
        assert (a["uid"] == b["uid"]).all()
        y = (a["label"] == "positive").to_numpy().astype(int)
        pa = (a["pred"] == "positive").to_numpy().astype(int)
        pb = (b["pred"] == "positive").to_numpy().astype(int)
        for g in ["ALL"] + GROUP_ORDER:
            msk = np.ones(len(y), bool) if g == "ALL" else (a["group"] == g).to_numpy()
            d, lo, hi, p = cmp.boot_delta(y[msk], pa[msk], pb[msk], rng)
            comps.append({"comparison": f"{b_name} - {a_name}", "group": g, "n": int(msk.sum()),
                          "delta": round(d, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                          "p": round(p, 3), "sig": "*" if p < 0.05 else ""})
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "comparisons.csv", index=False)
    wide = comps.assign(cell=comps["delta"].map(lambda v: f"{v:+.4f}") + comps["sig"]) \
                .pivot(index="comparison", columns="group", values="cell")[["ALL"] + GROUP_ORDER]

    runs = {k: json.loads((RUNS[k] / "run.json").read_text(encoding="utf-8")) for k in ["xlmr (min)", "xlmr (norm)"]}
    tok = pd.DataFrame({k: {**{f"fertility_{g}": v["tokenizer"]["fertility_test"][g] for g in GROUP_ORDER},
                            "truncated_test_%": v["tokenizer"]["truncated_test_%"],
                            "unk_rate_test_%": v["tokenizer"]["unk_rate_test_%"],
                            "best_val_macro_f1": round(v["best_val_macro_f1"], 4),
                            "val_f1_by_epoch": ", ".join(f"{h['eval_macro_f1']:.4f}" for h in v["val_history"]),
                            "train_minutes": v["train_minutes"], "gpu": v["gpu"]}
                        for k, v in runs.items()})

    report = [
        "# Step 5: XLM-R-base vs TF-IDF baselines\n",
        "All numbers on the same 20% test set (23,970 reviews).\n",
        "## Overall\n", overall.to_markdown(), "",
        "## Macro-F1 per language group\n", per_group.to_markdown(), "",
        "## Paired bootstrap differences (1000 resamples, * = p < 0.05)\n", wide.to_markdown(), "",
        "## XLM-R run details\n", tok.to_markdown(), "",
        "## Full comparison table\n", comps.set_index(["comparison", "group"]).to_markdown(), "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
