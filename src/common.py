"""Shared data split and evaluation used by every model (steps 3-7)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean.csv"
SPLIT = ROOT / "data" / "split_80_20.csv"

SEED = 42
LABELS = ["negative", "positive"]
GROUP_ORDER = ["bangla_script", "english", "romanized", "romanized_codemixed", "mixed_script"]


def make_split(force: bool = False) -> pd.DataFrame:
    """80% train / 20% test, stratified by label x language group.
    10% of the train part is held out as validation (tuning / early stopping only).
    Written once to data/split_80_20.csv and reused by every model."""
    if SPLIT.exists() and not force:
        return pd.read_csv(SPLIT)
    df = pd.read_csv(CLEAN, usecols=["uid", "label", "group"])
    strat = df["label"] + "|" + df["group"]
    trval, test = train_test_split(df, test_size=0.20, stratify=strat, random_state=SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=strat[trval.index], random_state=SEED)
    split = pd.concat([train.assign(split80="train"), val.assign(split80="val"), test.assign(split80="test")])
    split = split[["uid", "split80"]].sort_index()
    split.to_csv(SPLIT, index=False)
    return split


def load_data(text_col: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(CLEAN).merge(make_split(), on="uid")
    df["text"] = df[text_col]
    df["y"] = (df["label"] == "positive").astype(int)
    return {s: df[df["split80"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")


def evaluate(df: pd.DataFrame, y_pred, out_dir: Path, title: str, scores=None) -> dict:
    """Full evaluation on a split; writes metrics.json, predictions.csv, confusion.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y = df["y"].to_numpy()
    y_pred = np.asarray(y_pred)
    p, r, f, n = precision_recall_fscore_support(y, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    per_group = {}
    for g in GROUP_ORDER:
        m = (df["group"] == g).to_numpy()
        gp, gr, gf, gn = precision_recall_fscore_support(y[m], y_pred[m], labels=[0, 1], zero_division=0)
        per_group[g] = {"n": int(m.sum()), "macro_f1": float(gf.mean()),
                        "neg_f1": float(gf[0]), "pos_f1": float(gf[1]),
                        "accuracy": float(accuracy_score(y[m], y_pred[m]))}

    metrics = {
        "model": title,
        "n": int(len(y)),
        "macro_f1": float(f.mean()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i, lab in enumerate(LABELS)},
        "confusion_matrix": {"labels": LABELS, "rows_true_cols_pred": cm.tolist()},
        "per_group": per_group,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred = df[["uid", "group", "label"]].copy()
    pred["pred"] = np.where(y_pred == 1, "positive", "negative")
    if scores is not None:
        pred["score_positive"] = scores
    pred.to_csv(out_dir / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / cm[i].sum():.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=150)
    plt.close(fig)
    return metrics
