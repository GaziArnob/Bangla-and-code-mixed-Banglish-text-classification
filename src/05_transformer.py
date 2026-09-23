"""Steps 5-6: fine-tune a pretrained transformer (XLM-R-base, BanglaBERT) for sentiment.

- Each model's own tokenizer, max 128 subword tokens.
- Same 80/20 split as the baselines; the validation part (10% of train) picks the
  best epoch by macro-F1; the 20% test set is scored once with that checkpoint.
- Class-weighted cross-entropy ("balanced" weights, as in the baselines).
- Evaluation via common.evaluate (identical metrics / files to steps 3-4).

Usage:
  python src/05_transformer.py --model xlm-roberta-base --text text_min
  python src/05_transformer.py --model xlm-roberta-base --text text_norm
  python src/05_transformer.py --model csebuetnlp/banglabert --text text_min
Local smoke test on a small GPU:
  python src/05_transformer.py --limit 2000 --epochs 1 --batch 8 --freeze_embeddings
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from common import GROUP_ORDER, ROOT, evaluate, load_data, macro_f1

SHORT = {"xlm-roberta-base": "xlmr", "csebuetnlp/banglabert": "banglabert"}


class TextDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(outputs.logits, labels,
                                           weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def fertility(tok, texts: pd.Series) -> float:
    """Mean subword tokens per whitespace word (higher = words split into more pieces)."""
    words = texts.str.split().str.len().sum()
    pieces = sum(len(tok.tokenize(t)) for t in texts)
    return pieces / words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--text", default="text_min", choices=["text_min", "text_norm"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="subsample each split (smoke test only)")
    ap.add_argument("--freeze_embeddings", action="store_true", help="for GPUs with <=4 GB (smoke test)")
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    step = "step6" if "bangla" in args.model.lower() else "step5"
    name = f"{SHORT.get(args.model, args.model.split('/')[-1])}_{args.text}_seed{args.seed}"
    if args.limit:
        name = "SMOKE_" + name
    out = ROOT / "results" / step / name
    out.mkdir(parents=True, exist_ok=True)

    data = load_data(args.text)
    if args.limit:
        data = {k: v.sample(min(args.limit, len(v)), random_state=args.seed).reset_index(drop=True)
                for k, v in data.items()}
    tr, va, te = data["train"], data["val"], data["test"]

    tok = AutoTokenizer.from_pretrained(args.model)
    # BanglaBERT expects its own normalizer; plain text is fine for XLM-R.
    if "banglabert" in args.model:
        try:
            from normalizer import normalize as bb_norm
            for d in (tr, va, te):
                d["text"] = d["text"].map(lambda t: bb_norm(t) or t)
            print("applied csebuetnlp normalizer")
        except ImportError:
            print("WARNING: csebuetnlp normalizer not installed; using text as is")

    def enc(df):
        return tok(df["text"].tolist(), truncation=True, max_length=args.max_len)

    t0 = time.time()
    etr, eva, ete = enc(tr), enc(va), enc(te)
    lens = np.array([len(x) for x in ete["input_ids"]])
    tok_stats = {
        "truncated_test_%": round(100 * float((lens >= args.max_len).mean()), 2),
        "mean_len_test": round(float(lens.mean()), 1),
        "fertility_test": {g: round(float(fertility(tok, te.loc[te["group"] == g, "text"])), 3) for g in GROUP_ORDER},
        "unk_rate_test_%": round(100 * float(np.mean([i == tok.unk_token_id for x in ete["input_ids"] for i in x])), 3),
    }
    print(f"tokenized in {time.time() - t0:.0f}s; {tok_stats}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    if args.freeze_embeddings:
        model.base_model.embeddings.word_embeddings.weight.requires_grad = False

    counts = np.bincount(tr["y"], minlength=2)
    class_weights = torch.tensor(len(tr) / (2 * counts), dtype=torch.float)
    print("class weights", class_weights.tolist())

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        warmup_steps=int(0.06 * args.epochs * np.ceil(len(tr) / args.batch)),  # 6% warmup
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )

    def compute_metrics(p):
        return {"macro_f1": macro_f1(p.label_ids, p.predictions.argmax(-1))}

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=TextDS(etr, tr["y"].to_numpy()),
        eval_dataset=TextDS(eva, va["y"].to_numpy()),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    val_f1 = trainer.evaluate()["eval_macro_f1"]
    logits = trainer.predict(TextDS(ete, te["y"].to_numpy())).predictions
    prob = torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()
    m = evaluate(te, logits.argmax(-1), out, f"{name}", scores=prob)

    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    run = {"args": vars(args), "best_val_macro_f1": val_f1, "train_minutes": round(train_min, 1),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "val_history": history, "tokenizer": tok_stats,
           "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"]}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if args.save_model:
        trainer.save_model(str(out / "model"))
    import shutil
    shutil.rmtree(out / "ckpt", ignore_errors=True)

    print(f"\n{name}: val macro-F1={val_f1:.4f}  TEST macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
          f"({train_min:.1f} min)")
    print({g: round(v["macro_f1"], 4) for g, v in m["per_group"].items()})


if __name__ == "__main__":
    main()
