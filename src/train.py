"""
Fine-tune DeBERTa on Jigsaw toxicity data.
Run: python -m src.train
"""

import torch, evaluate
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
import transformers
from .config import *
print("Transformers version:", transformers.__version__)
from sklearn.metrics import roc_auc_score
import numpy as np

import evaluate
print(evaluate.__version__)              # 0.4.x
if "roc_auc" in evaluate.list_evaluation_modules():
    print("roc_auc")
else:
    print("roc_auc not found in this evaluate build")

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        """
        enc = tok(batch["comment_text"], truncation=True, padding="max_length",
            max_length=MAX_LEN)
        for col in LABELS:
            enc[col] = batch[col]
        return enc"""
        enc = tok(batch["comment_text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        labels_batch = []
        num_items_in_batch = len(batch[LABELS[0]])

        for i in range(num_items_in_batch):
            current_item_labels = [float(batch[label_col][i]) for label_col in LABELS]
            labels_batch.append(current_item_labels)
        enc["labels"] = labels_batch
        return enc


    ds = load_dataset("jigsaw_toxicity_pred", data_dir="data/jigsaw")
    ds = ds.map(tokenize, batched=True, remove_columns=LABELS)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    ds_split = ds["train"].train_test_split(0.1, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        auc = roc_auc_score(labels, probs, average="macro")
        return {"roc_auc": auc}


    args = TrainingArguments(
        output_dir="model",
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        fp16=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(model, args,
    train_dataset=ds_split["train"],
    eval_dataset=ds_split["test"],
    tokenizer=tok,
    compute_metrics=compute_metrics)

    trainer.train()
    results = trainer.evaluate()
    print(f"\nFinal macro ROC-AUC: {results['eval_roc_auc']:.4f}\n")

    model.save_pretrained("model")
    tok.save_pretrained("model")

if __name__ == "__main__":
    main()