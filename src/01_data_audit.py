"""Step 1: data audit and script-based language groups for BanglishRev sentiment.

Groups are derived from the characters actually present in `text_raw`
(Bengali block U+0980-U+09FF vs Latin letters) combined with the dataset's
`is_codemixed` flag. The `is_romanized` flag alone is not used because it
misses many Latin-script Bangla reviews.

Outputs:
  data/groups.csv                 uid, split, label, group
  results/step1/*.csv / *.md      statistics tables for the paper
  results/step1/group_examples.txt
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "banglishrev_data" / "banglishrev.csv"
OUT = ROOT / "results" / "step1"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

GROUP_ORDER = ["bangla_script", "english", "romanized", "romanized_codemixed", "mixed_script", "other"]
SPLIT_ORDER = ["train", "val", "test"]

BN_RE = r"[ঀ-৿]"
LAT_RE = r"[A-Za-z]"


def assign_group(row) -> str:
    if row.has_bn and row.has_lat:
        return "mixed_script"          # Bengali script + Latin letters in one review
    if row.has_bn:
        return "bangla_script"
    if row.has_lat:
        if row.is_codemixed:
            return "romanized_codemixed"   # Latin-only, Bangla and English words mixed
        if row.is_romanized:
            return "romanized"             # Latin-only Bangla, no English switching
        return "english"
    return "other"                          # emoji / digits / punctuation only


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown()


def main():
    df = pd.read_csv(DATA)
    df["has_bn"] = df["text_raw"].str.contains(BN_RE, regex=True)
    df["has_lat"] = df["text_raw"].str.contains(LAT_RE, regex=True)
    df["group"] = df.apply(assign_group, axis=1)
    df["group"] = pd.Categorical(df["group"], GROUP_ORDER)
    df["split"] = pd.Categorical(df["split"], SPLIT_ORDER)

    df[["uid", "split", "label", "label_raw", "group"]].to_csv(ROOT / "data" / "groups.csv", index=False)

    report = ["# Step 1: Dataset audit\n", f"Total rows: {len(df):,}\n"]

    # 1. Group x label
    gl = pd.crosstab(df["group"], df["label"], margins=True, margins_name="total")
    gl["neg_%"] = (100 * gl["negative"] / gl["total"]).round(1)
    gl["share_%"] = (100 * gl["total"] / len(df)).round(1)
    gl.to_csv(OUT / "group_by_label.csv")
    report += ["## Language group x label\n", md_table(gl), ""]

    # 2. Group x split
    gs = pd.crosstab(df["group"], df["split"], margins=True, margins_name="total")
    gs.to_csv(OUT / "group_by_split.csv")
    report += ["## Language group x split\n", md_table(gs), ""]

    # 3. Split x label (check stratification)
    sl = pd.crosstab(df["split"], df["label"], margins=True, margins_name="total")
    sl["neg_%"] = (100 * sl["negative"] / sl["total"]).round(1)
    sl.to_csv(OUT / "split_by_label.csv")
    report += ["## Split x label\n", md_table(sl), ""]

    # 4. Star rating -> label mapping
    sr = pd.crosstab(df["label_raw"], df["label"])
    sr.to_csv(OUT / "stars_by_label.csv")
    report += ["## Star rating -> label\n", md_table(sr), ""]

    # 5. Length stats per group
    ln = df.groupby("group", observed=True)["n_tokens"].describe(percentiles=[.5, .95])[
        ["mean", "50%", "95%", "max"]].round(1)
    ln.to_csv(OUT / "length_by_group.csv")
    report += ["## Tokens per review\n", md_table(ln), ""]

    # 6. Flag vs script-based group disagreement
    fl = pd.crosstab(df["group"], [df["is_romanized"], df["is_codemixed"]])
    fl.columns = [f"rom={r},cm={c}" for r, c in fl.columns]
    fl.to_csv(OUT / "group_vs_flags.csv")
    report += ["## Script-based group vs dataset flags\n", md_table(fl), ""]

    # 7. Data quality: empty text, duplicates, cross-split leakage
    key = df["text"].str.strip().str.lower()
    empty = int((key == "").sum() + df["text"].isna().sum())
    dup_rows = int(key.duplicated().sum())
    per_text_splits = df.assign(k=key).groupby("k")["split"].nunique()
    leaked_texts = per_text_splits[per_text_splits > 1].index
    leak = df[key.isin(leaked_texts)]
    test_leak = int(((leak["split"] == "test") & key[leak.index].isin(
        set(key[df["split"] == "train"]))).sum())
    conflict = df.assign(k=key).groupby("k")["label"].nunique()
    n_conflict = int((conflict > 1).sum())
    top_dups = key.value_counts().head(10)

    # 8. Noise that step 2 will handle
    raw = df["text_raw"].fillna("")
    noise = pd.Series({
        "html_tags": raw.str.contains(r"<[^>]+>", regex=True).sum(),
        "html_entities": raw.str.contains(r"&[a-z]+;|&#\d+;", regex=True).sum(),
        "urls": raw.str.contains(r"https?://|www\.", regex=True).sum(),
        "mentions": raw.str.contains(r"(?<!\w)@\w+", regex=True).sum(),
        "hashtags": raw.str.contains(r"(?<!\w)#\w+", regex=True).sum(),
        "newlines": raw.str.contains(r"\n").sum(),
        "has_emoji": (df["n_emoji"] > 0).sum(),
    }).astype(int)
    noise.to_csv(OUT / "noise_counts.csv", header=["rows"])

    report += [
        "## Data quality\n",
        f"- Empty texts: {empty}",
        f"- Duplicate rows (same normalized text): {dup_rows:,} ({100 * dup_rows / len(df):.1f}%)",
        f"- Distinct texts appearing in more than one split: {len(leaked_texts):,}",
        f"- Test rows whose exact text also appears in train: {test_leak:,}",
        f"- Distinct texts with conflicting labels: {n_conflict:,}",
        "",
        "Most frequent texts:\n",
        md_table(top_dups.rename("count").to_frame()),
        "",
        "## Noise present in text_raw (rows)\n",
        md_table(noise.rename("rows").to_frame()),
        "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    # Examples per group for manual verification
    with open(OUT / "group_examples.txt", "w", encoding="utf-8") as f:
        for g in GROUP_ORDER:
            sub = df[df["group"] == g]
            if sub.empty:
                continue
            f.write(f"===== {g} ({len(sub):,}) =====\n")
            for _, r in sub.sample(min(15, len(sub)), random_state=42).iterrows():
                f.write(f"[{r.label}] {r.text_raw[:150]!r}\n")
            f.write("\n")

    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
