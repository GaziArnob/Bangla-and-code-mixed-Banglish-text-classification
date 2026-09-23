"""Step 2: build two text versions from `text_raw`.

text_min  (minimal cleaning)
  - Unicode NFC, drop zero-width space / BOM (ZWJ/ZWNJ kept: needed by Bengali and emoji)
  - unescape HTML entities, remove HTML tags
  - URLs -> <URL>, @mentions -> <USER>
  - hashtags: keep the word, drop '#', '_' -> space
  - collapse all whitespace (incl. newlines) to single spaces
  Case, punctuation, digits, emojis, English words and negation are untouched.

text_norm (normalized Banglish) = text_min +
  - lowercase Latin letters
  - Bengali digits -> ASCII digits; broken vowel sign 'অা' -> 'আ'; '৷' -> '।'
  - runs of the same punctuation mark -> one mark (except between digits: '7..8')
  - space after sentence punctuation glued to the next word ("good.thanks" -> "good. thanks")
  - letter elongation: 3+ identical letters -> 2 ("sooooo" -> "soo")
  - Romanized Bangla spelling variants -> one canonical spelling (lexicon below)
  Emojis and negation words are kept. No stopword removal, no stemming.

Also drops the 11 rows whose text is the spreadsheet error "#NAME?".

Outputs:
  data/clean.csv                    uid, split, label, label_raw, group, text_min, text_norm
  data/romanized_lexicon.tsv        variant -> canonical, with corpus counts
  results/step2/report.md, examples.txt
"""
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step2"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_ORDER = ["bangla_script", "english", "romanized", "romanized_codemixed", "mixed_script"]

# ---------------------------------------------------------------- lexicon
# canonical: variants. Canonical = most frequent spelling in the corpus.
# Only spelling variants of the same word form; different verb forms / dialect
# words are not merged. Ambiguous variants that are also common English or
# different Bangla words (e.g. "ase" = comes/has, "basi" = stale) are excluded.
LEXICON = {
    "valo": ["vlo", "bhalo", "balo", "vhalo", "bhlo", "valow", "bhalow"],
    "valoi": ["vloi", "bhaloi", "baloi", "vhaloi"],
    "onek": ["onk", "onak", "anek", "onnek"],
    "khub": ["kub", "khob", "khuv", "khoob"],
    "khubi": ["kubi", "khobi"],
    "sundor": ["shundor", "sondor", "sundar", "shundar", "sundr"],
    "chilo": ["cilo", "silo", "celo", "selo", "chilo"],
    "ekta": ["akta"],
    "ekti": ["akti"],
    "ektu": ["aktu", "akto"],
    "ekdom": ["akdom"],
    "kintu": ["kinto"],
    "beshi": ["besi"],
    "amar": ["amr"],
    "amake": ["amk"],
    "jonno": ["jnno"],
    "thik": ["tik"],
    "thikthak": ["tiktak", "thiktak", "tikthak"],
    "dhonnobad": ["donnobad", "dhonyobad", "dhonnyobad", "dhonnobaad"],
    "kichu": ["kisu", "kicu", "kicchu"],
    "jinis": ["jinish"],
    "ache": ["ace", "asey"],
    "motamuti": ["motamoti", "mutamuti", "mutamoti"],
    "jemon": ["jmn", "jamon"],
    "temon": ["tmn", "tamon"],
    "kemon": ["kmn", "kamon"],
    "jemonta": ["jmnta", "jamonta"],
    "temoni": ["tmni", "tamoni"],
    "sobai": ["shobai", "sobay", "shobay"],
    "sob": ["shob"],
    "sathe": ["shathe"],
    "hisebe": ["hisabe", "hishebe", "hishabe"],
    "onujayi": ["onujai", "onujaye"],
    "bebohar": ["babohar", "byabohar", "bebohar"],
    "pochondo": ["posondo", "pocondo", "pochhondo"],
    "osadharon": ["oshadharon", "osadaron", "oshadharon", "osadharon"],
    "kharap": ["karap"],
    "khushi": ["khusi"],
    "ekhon": ["akhon", "akhn", "ekhn"],
    "ekhono": ["akhono", "akhno"],
    "abar": ["abr"],
    "emon": ["amon"],
    "keu": ["kew", "keo"],
    "hoy": ["hoi"],
    "jay": ["jai"],
    "gulo": ["gula"],
    "eta": ["ata"],
    "eita": ["aita", "ayta"],
    "ei": ["ai"],
    "peyechi": ["peyeci", "peyesi", "peyachi"],
    "paisi": ["paichi", "paici"],
    "lagse": ["lagce", "lagche"],
    "hoise": ["hoice", "hoiche"],
    "hoyeche": ["hoyese", "hoyeci"],
    "geche": ["gese", "gece"],
    "dise": ["dice", "diche"],
    "hocche": ["hosse", "hocce"],
    "kache": ["kase"],
    "chaisi": ["chaichi", "caisi"],
    "chaile": ["caile"],
    "vai": ["bhai"],
    "vaiya": ["bhaiya"],
    "na": ["nah", "naa", "nh"],
    "but": ["bt"],
    "thanks": ["thnx", "tnx", "thx", "thanx", "tnq", "thnq", "tnx"],
    "seller": ["sellar", "selar"],
    "daraz": ["daraj"],
}
VARIANT2CANON = {v: c for c, vs in LEXICON.items() for v in vs if v != c}
KNOWN = set(VARIANT2CANON) | set(LEXICON)

# ---------------------------------------------------------------- regexes
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION_RE = re.compile(r"(?<![\w@])@\w+")
# not \w: in Python it stops at Bengali vowel signs
HASHTAG_RE = re.compile(r"(?<![\w#])#([^\s#.,!?।;:()\[\]{}\"']+)")
TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
WS_RE = re.compile(r"\s+")
ZW_RE = re.compile("[​﻿]")

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
PUNCT_RUN_RE = re.compile(r"([.,!?;:।\-~*])\1+")
GLUED_RE = re.compile(r"([.,!?।])(?=[^\s\d.,!?।<])")
ELONG_RE = re.compile(r"([a-z])\1{2,}")
LATIN_TOKEN_RE = re.compile(r"[a-z]+")


def clean_minimal(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = ZW_RE.sub("", t)
    t = html.unescape(t)
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub(" <URL> ", t)
    t = MENTION_RE.sub(" <USER> ", t)
    t = HASHTAG_RE.sub(lambda m: m.group(1).replace("_", " "), t)
    return WS_RE.sub(" ", t).strip()


def _collapse_punct(m: re.Match) -> str:
    # keep ranges like "7..8" intact; otherwise one mark
    s, e, txt = m.start(), m.end(), m.string
    if 0 < s and e < len(txt) and txt[s - 1].isdigit() and txt[e].isdigit():
        return m.group(0)
    return m.group(1)


def _norm_token(m: re.Match) -> str:
    w = m.group(0)
    if w not in KNOWN:
        # "valoo" / "naa" -> try the fully de-elongated form, only if it is a lexicon word
        short = re.sub(r"([a-z])\1+", r"\1", w)
        if short in KNOWN:
            w = short
    return VARIANT2CANON.get(w, w)


def normalize(t: str, counter: Counter | None = None) -> str:
    # keep placeholders intact while lowercasing
    t = t.replace("<URL>", "\x00URL\x00").replace("<USER>", "\x00USER\x00")
    t = t.lower()
    t = t.translate(BN_DIGITS).replace("অা", "আ").replace("৷", "।")
    t = PUNCT_RUN_RE.sub(_collapse_punct, t)
    t = GLUED_RE.sub(r"\1 ", t)
    t = ELONG_RE.sub(r"\1\1", t)
    if counter is not None:
        for w in LATIN_TOKEN_RE.findall(t):
            if w in VARIANT2CANON:
                counter[w] += 1
    t = LATIN_TOKEN_RE.sub(_norm_token, t)
    t = t.replace("\x00url\x00", "<URL>").replace("\x00user\x00", "<USER>")
    return WS_RE.sub(" ", t).strip()


def main():
    df = pd.read_csv(ROOT / "banglishrev_data" / "banglishrev.csv")
    groups = pd.read_csv(ROOT / "data" / "groups.csv")[["uid", "group"]]
    df = df.merge(groups, on="uid", how="left")

    bad = df["text_raw"].str.strip() == "#NAME?"
    n_bad = int(bad.sum())
    df = df[~bad].copy()

    df["text_min"] = df["text_raw"].map(clean_minimal)
    var_counts = Counter()
    df["text_norm"] = [normalize(t, var_counts) for t in df["text_min"]]

    empty = (df["text_min"] == "") | (df["text_norm"] == "")
    assert not empty.any(), df.loc[empty, "text_raw"].head()

    cols = ["uid", "split", "label", "label_raw", "group", "text_min", "text_norm"]
    df[cols].to_csv(ROOT / "data" / "clean.csv", index=False)

    # lexicon table with corpus counts + frequency in the English group (collision check)
    eng_tokens = Counter(w for s in df.loc[df["group"] == "english", "text_min"].str.lower()
                         for w in LATIN_TOKEN_RE.findall(s))
    lex = pd.DataFrame([{"variant": v, "canonical": c, "count": var_counts.get(v, 0),
                         "count_in_english_group": eng_tokens.get(v, 0)}
                        for v, c in VARIANT2CANON.items()]).sort_values("count", ascending=False)
    lex.to_csv(ROOT / "data" / "romanized_lexicon.tsv", sep="\t", index=False)

    # ------------------------------------------------------------ report
    # split on whitespace/punctuation; \w would break Bengali words at vowel signs
    word_re = re.compile(r"[^\s.,!?;:।'\"()\[\]{}\-~*/]+")

    def words(text):
        return word_re.findall(text.lower())

    rows = []
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        wm = [words(t) for t in sub["text_min"]]
        wn = [words(t) for t in sub["text_norm"]]
        latin = sum(1 for x in wm for w in x if LATIN_TOKEN_RE.fullmatch(w))
        spell = sum(sum(1 for w in LATIN_TOKEN_RE.findall(t.lower()) if w in VARIANT2CANON)
                    for t in sub["text_min"])
        vm = len({w for x in wm for w in x})
        vn = len({w for x in wn for w in x})
        rows.append({
            "group": g,
            "rows": len(sub),
            "reviews_changed_by_min_%": round(100 * (sub["text_raw"] != sub["text_min"]).mean(), 1),
            "reviews_changed_by_norm_%": round(100 * (sub["text_min"].str.lower() != sub["text_norm"]).mean(), 1),
            "latin_word_tokens": latin,
            "spelling_variants_replaced": spell,
            "replaced_%_of_latin_tokens": round(100 * spell / max(latin, 1), 1),
            "word_vocab_min": vm,
            "word_vocab_norm": vn,
            "vocab_reduction_%": round(100 * (vm - vn) / vm, 1),
        })
    stats = pd.DataFrame(rows).set_index("group")
    stats.to_csv(OUT / "normalization_stats.csv")

    raw = df["text_raw"]

    def has(series, rx):
        rx = re.compile(rx) if isinstance(rx, str) else rx
        return series.map(lambda t: bool(rx.search(t)))

    rules = pd.Series({
        "html_tags_removed": has(raw, TAG_RE).sum(),
        "html_entities_unescaped": has(raw, r"&[a-z]+;|&#\d+;").sum(),
        "urls_to_<URL>": has(raw, URL_RE).sum(),
        "mentions_to_<USER>": has(raw, MENTION_RE).sum(),
        "hashtags_kept_as_words": has(raw, HASHTAG_RE).sum(),
        "whitespace_collapsed": has(raw, r"\s{2,}|\n").sum(),
        "bengali_digits_converted": has(df["text_min"], "[০-৯]").sum(),
        "punct_runs_collapsed": has(df["text_min"], PUNCT_RUN_RE).sum(),
        "elongations_reduced": has(df["text_min"].str.lower(), ELONG_RE).sum(),
        "lexicon_replacements(tokens)": sum(var_counts.values()),
    }).astype(int)
    rules.to_csv(OUT / "rule_counts.csv", header=["rows"])

    report = [
        "# Step 2: Preprocessing\n",
        f"- Rows in: {len(df) + n_bad:,}; dropped `#NAME?` rows: {n_bad}; rows out: {len(df):,}",
        f"- Lexicon: {len(LEXICON)} canonical words, {len(VARIANT2CANON)} variants\n",
        "## How often each rule fired\n", rules.rename("rows").to_frame().to_markdown(), "",
        "## Effect of normalization per language group\n", stats.to_markdown(), "",
        "## Top 25 lexicon replacements\n",
        lex.head(25).set_index("variant").to_markdown(), "",
        "## Variants that also occur in the English-only group (collision check)\n",
        lex[lex["count_in_english_group"] > 0].set_index("variant").to_markdown(), "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    with open(OUT / "examples.txt", "w", encoding="utf-8") as f:
        for g in GROUP_ORDER:
            sub = df[(df["group"] == g) & (df["text_min"].str.lower() != df["text_norm"])]
            f.write(f"===== {g} =====\n")
            for _, r in sub.sample(min(8, len(sub)), random_state=7).iterrows():
                f.write(f"RAW : {r.text_raw[:160]!r}\nMIN : {r.text_min[:160]}\nNORM: {r.text_norm[:160]}\n\n")
        f.write("===== rows touched by URL / mention / hashtag / HTML rules =====\n")
        special = df[has(raw, URL_RE) | has(raw, MENTION_RE)
                     | has(raw, HASHTAG_RE) | has(raw, TAG_RE)]
        for _, r in special.sample(min(12, len(special)), random_state=1).iterrows():
            f.write(f"RAW : {r.text_raw[:160]!r}\nMIN : {r.text_min[:160]}\n\n")

    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
