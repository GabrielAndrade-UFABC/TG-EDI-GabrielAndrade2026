import numpy as np
from datasets import load_dataset
from peft import LoraConfig
from codebleu import calc_codebleu
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from dataset_tokenizer import fix_tokenizer_padding, tokenize_dataset
from dataset_map import load_and_split_dataset

def run_experiment(model_name, dataset_name, text_column):

    print(f"\nRunning {model_name} | {dataset_name}")

    dataset = load_and_split_dataset(dataset_name)

    # model_name = f"./pretrained/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize(example):

        prompt = f"""
    ### Buggy code:
    {example[text_column]}

    ### Fixed code:
    """

        # hardcoded fix
        match dataset_name:
            case "bstee615/bigvul":
                target = example["func_after"]
            case "hitoshura25/cvefixes":
                target = example["fixed_code"]
            case "hitoshura25/megavul":
                target = example["fixed_code"]
            case "hitoshura25/crossvul":
                target = example["fixed_code"]
            case _:
                raise ValueError("Unidentified dataset.")

        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=False
        )

        target_tokens = tokenizer(
            target,
            add_special_tokens=False
        )

        input_ids = (
            prompt_tokens["input_ids"]
            + target_tokens["input_ids"]
        )

        attention_mask = (
            prompt_tokens["attention_mask"]
            + target_tokens["attention_mask"]
        )

        labels = (
            [-100] * len(prompt_tokens["input_ids"])
            + target_tokens["input_ids"]
        )

        max_length = 256

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

        pad_len = max_length - len(input_ids)

        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    tokenizer, model = fix_tokenizer_padding(tokenizer, model)
    tokenized_dataset = tokenize_dataset(tokenize, dataset, dataset_name)

    def extract_fixed(text):

        marker = "### Fixed code:"

        if marker in text:
            text = text.split(marker, 1)[1]

        return text.strip()

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

        decoded_preds = [
            extract_fixed(x)
            for x in decoded_preds
        ]

        decoded_labels = [
            extract_fixed(x)
            for x in decoded_labels
        ]

        for i in range(3):

            print("\nPRED:\n")
            print(decoded_preds[i])

            print("\nLABEL:\n")
            print(decoded_labels[i])

        result = calc_codebleu(
            references=[[x] for x in decoded_labels],
            predictions=decoded_preds,
            lang="c"
        )

        return {
            "codebleu": result["codebleu"],
            "ngram_match_score": result["ngram_match_score"],
            "weighted_ngram_match_score": result["weighted_ngram_match_score"],
            "syntax_match_score": result["syntax_match_score"],
            "dataflow_match_score": result["dataflow_match_score"]
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