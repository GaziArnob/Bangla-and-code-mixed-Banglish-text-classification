"""Step 6 summary: BanglaBERT vs XLM-R vs baselines, and XLM-R seed variance.

- Collects every transformer run in results/step5 and results/step6
  (folders <model>_<text>_seed<N>) plus the TF-IDF baselines.
- Reports mean +- std over seeds (overall and per language group).
- Normalization effect per model and seed, with paired bootstrap per seed.
- Paired bootstrap: BanglaBERT vs XLM-R (seed 42) and vs the best baseline.
Writes results/step6/report.md, all_runs.csv, comparisons.csv.
"""
import importlib.util
import json
import re

import numpy as np
import pandas as pd

from common import GROUP_ORDER, ROOT, SEED

spec = importlib.util.spec_from_file_location("cmp", ROOT / "src" / "04_compare.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

R = ROOT / "results"
RUN_RE = re.compile(r"^(?P<model>[a-z]+)_(?P<text>text_min|text_norm)_seed(?P<seed>\d+)$")


def collect():
    runs = []
    for step in ["step5", "step6"]:
        for d in sorted((R / step).glob("*")):
            m = RUN_RE.match(d.name)
            if d.is_dir() and m and (d / "metrics.json").exists():
                runs.append({"model": m["model"], "text": m["text"].replace("text_", ""),
                             "seed": int(m["seed"]), "dir": d})
    for text, step in [("min", "step3"), ("norm", "step4")]:
        runs.append({"model": "word_tfidf_svm", "text": text, "seed": 0, "dir": R / step / "word_tfidf_svm"})
    return runs


def preds(d):
    p = pd.read_csv(d / "predictions.csv").sort_values("uid").reset_index(drop=True)
    return p, (p["label"] == "positive").to_numpy().astype(int), (p["pred"] == "positive").to_numpy().astype(int)


def compare(a_dir, b_dir, label, rng):
    pa_df, y, pa = preds(a_dir)
    pb_df, _, pb = preds(b_dir)
    assert (pa_df["uid"] == pb_df["uid"]).all()
    rows = []
    for g in ["ALL"] + GROUP_ORDER:
        m = np.ones(len(y), bool) if g == "ALL" else (pa_df["group"] == g).to_numpy()
        d, lo, hi, p = cmp.boot_delta(y[m], pa[m], pb[m], rng)
        rows.append({"comparison": label, "group": g, "delta": round(d, 4), "ci95_lo": round(lo, 4),
                     "ci95_hi": round(hi, 4), "p": round(p, 3), "sig": "*" if p < 0.05 else ""})
    return rows


def main():
    out = R / "step6"
    out.mkdir(exist_ok=True)
    runs = collect()

    rows = []
    for r in runs:
        met = json.loads((r["dir"] / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"model": r["model"], "text": r["text"], "seed": r["seed"],
                     "macro_f1": met["macro_f1"], "accuracy": met["accuracy"],
                     "neg_f1": met["per_class"]["negative"]["f1"], "pos_f1": met["per_class"]["positive"]["f1"],
                     **{g: met["per_group"][g]["macro_f1"] for g in GROUP_ORDER}})
    allr = pd.DataFrame(rows).sort_values(["model", "text", "seed"])
    allr.round(4).to_csv(out / "all_runs.csv", index=False)

    cols = ["macro_f1", "accuracy", "neg_f1", "pos_f1"] + GROUP_ORDER
    agg = allr.groupby(["model", "text"])[cols].agg(["mean", "std"])
    n_seeds = allr.groupby(["model", "text"])["seed"].nunique()

    def fmt(row, c):
        mu, sd = row[(c, "mean")], row[(c, "std")]
        return f"{mu:.4f}" if pd.isna(sd) else f"{mu:.4f} ± {sd:.4f}"

    table = pd.DataFrame({c: agg.apply(lambda r: fmt(r, c), axis=1) for c in cols})
    table.insert(0, "seeds", n_seeds)

    # normalization effect per model / seed
    rng = np.random.default_rng(SEED)
    norm_rows, comps = [], []
    by = {(r["model"], r["text"], r["seed"]): r["dir"] for r in runs}
    for (model, text, seed), d in sorted(by.items()):
        if text != "min" or (model, "norm", seed) not in by:
            continue
        res = compare(d, by[(model, "norm", seed)], f"{model} seed{seed}: norm - min", rng)
        comps += res
        norm_rows.append({"model": model, "seed": seed,
                          **{r_["group"]: f"{r_['delta']:+.4f}{r_['sig']}" for r_ in res}})
    norm_tbl = pd.DataFrame(norm_rows).set_index(["model", "seed"])
    mean_delta = (allr[allr.text == "norm"].set_index(["model", "seed"])[["macro_f1"] + GROUP_ORDER]
                  - allr[allr.text == "min"].set_index(["model", "seed"])[["macro_f1"] + GROUP_ORDER]
                  ).dropna().groupby("model").agg(["mean", "std"]).round(4)

    # model comparisons (seed 42 transformers)
    for text in ["min", "norm"]:
        pairs = [(("xlmr", text, 42), ("banglabert", text, 42)),
                 (("word_tfidf_svm", text, 0), ("banglabert", text, 42))]
        for a, b in pairs:
            if a in by and b in by:
                comps += compare(by[a], by[b], f"{b[0]} - {a[0]} ({text})", rng)
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "comparisons.csv", index=False)
    model_cmp = comps[~comps["comparison"].str.contains("norm - min")].copy()
    model_cmp["cell"] = [f"{d:+.4f}{s}" for d, s in zip(model_cmp["delta"], model_cmp["sig"])]
    model_wide = (model_cmp.pivot(index="comparison", columns="group", values="cell")[["ALL"] + GROUP_ORDER]
                  if len(model_cmp) else pd.DataFrame())

    tok = {}
    for (model, text, seed), d in by.items():
        rj = d / "run.json"
        if seed == 42 and rj.exists():
            t = json.loads(rj.read_text(encoding="utf-8"))["tokenizer"]
            tok[f"{model} ({text})"] = {**{g: t["fertility_test"][g] for g in GROUP_ORDER},
                                        "unk_%": t["unk_rate_test_%"], "truncated_%": t["truncated_test_%"]}
    tok = pd.DataFrame(tok).T

    report = [
        "# Step 6: BanglaBERT vs XLM-R vs TF-IDF\n",
        "20% test set (23,970 reviews). Transformers: mean ± std over seeds where >1 seed.\n",
        "## Results (macro-F1 unless noted)\n", table.to_markdown(), "",
        "## Normalization effect (norm - min), paired bootstrap per seed, * = p < 0.05\n",
        norm_tbl.to_markdown(), "",
        "Mean ± std of the per-seed deltas:\n", mean_delta.to_markdown(), "",
        "## Model comparisons (seed 42), paired bootstrap\n", model_wide.to_markdown(), "",
        "## Tokenizer: subword pieces per word (fertility), UNK and truncation on test\n",
        tok.to_markdown(), "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
