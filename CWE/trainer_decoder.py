from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rapidfuzz import fuzz
import numpy as np
from evaluate import load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from dataset_tokenizer import fix_tokenizer_padding, tokenize_dataset
from dataset_map import load_and_split_dataset

bleu_metric = load("bleu")

def run_experiment(model_name, dataset_name, text_column):

    print(f"\nRunning {model_name} | {dataset_name}")

    dataset = load_and_split_dataset(dataset_name)

    # model_name = f"./pretrained/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    def build_cwe_target(example):

        cwe_id = example.get("cwe_id", "")
        cwe_desc = example.get("cwe_description", "")

        if isinstance(cwe_id, list):
            cwe_id = ", ".join(cwe_id)

        if isinstance(cwe_desc, list):
            cwe_desc = ", ".join(cwe_desc)

        example["cwe_target"] = f"{cwe_id}: {cwe_desc}"

        return example

    dataset = dataset.map(build_cwe_target)

    def tokenize(example):

        prompt = f"""
    ### Vulnerable code:
    {example[text_column]}

    ### Identify the CWE:
    """

        target = example["cwe_target"]

        full_text = prompt + target

        tokens = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    tokenizer, model = fix_tokenizer_padding(tokenizer, model)
    tokenized_dataset = tokenize_dataset(tokenize, dataset, dataset_name)

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred

        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id
        )

        decoded_preds = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        def extract_fixed(text):

            marker = "### Identify the CWE:"

            if marker in text:
                text = text.split(marker, 1)[1]

            return text.strip()

        decoded_preds = [
            extract_fixed(x)
            for x in decoded_preds
        ]

        decoded_labels = [
            extract_fixed(x)
            for x in decoded_labels
        ]

        pred_cwe_ids = []
        label_cwe_ids = []

        pred_descriptions = []
        label_descriptions = []

        for pred, label in zip(decoded_preds, decoded_labels):

            if ":" in pred:
                pred_id, pred_desc = pred.split(":", 1)
            else:
                pred_id = pred.strip()
                pred_desc = ""

            if ":" in label:
                label_id, label_desc = label.split(":", 1)
            else:
                label_id = label.strip()
                label_desc = ""

            pred_cwe_ids.append(
                pred_id.strip().upper()
            )

            label_cwe_ids.append(
                label_id.strip().upper()
            )

            pred_descriptions.append(
                pred_desc.strip()
            )

            label_descriptions.append(
                label_desc.strip()
            )

        unique_ids = sorted(
            list(
                set(label_cwe_ids)
                | set(pred_cwe_ids)
            )
        )

        label_map = {
            cwe: idx
            for idx, cwe in enumerate(unique_ids)
        }

        y_true = [
            label_map[x]
            for x in label_cwe_ids
        ]

        y_pred = [
            label_map[x]
            for x in pred_cwe_ids
        ]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        acc = accuracy_score(y_true, y_pred)

        bleu_result = bleu_metric.compute(
            predictions=pred_descriptions,
            references=[[x] for x in label_descriptions]
        )

        exact_match = np.mean([
            p.lower() == l.lower()
            for p, l in zip(
                pred_descriptions,
                label_descriptions
            )
        ])

        similarity = np.mean([
            fuzz.ratio(p, l)
            for p, l in zip(pred_descriptions, label_descriptions)
        ])

        for i in range(min(3, len(pred_cwe_ids))):
            print("\nPRED CWE:")
            print(pred_cwe_ids[i])

            print("LABEL CWE:")
            print(label_cwe_ids[i])

            print("\nPRED DESC:")
            print(pred_descriptions[i])

            print("LABEL DESC:")
            print(label_descriptions[i])

        return {
            "bleu": bleu_result["bleu"],
            "exact_match": exact_match,
            "similarity": similarity,
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    training_args = TrainingArguments(

        output_dir="./tmp",

        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,

        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none"
    )

    def preprocess_logits_for_metrics(logits, labels):

        if isinstance(logits, tuple):
            logits = logits[0]

        return logits.argmax(dim=-1)

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    predictions = trainer.predict(tokenized_dataset["test"])
    trainer.save_model(f"./pretrained/{model_name}")

    return predictions.metrics