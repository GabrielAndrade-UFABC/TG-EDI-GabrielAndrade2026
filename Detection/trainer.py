from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from metrics import compute_metrics
from tokenizer import fix_tokenizer_padding, tokenize_dataset


def run_experiment(model_name, dataset_name, text_column):

    print(f"\nRunning {model_name} | {dataset_name}")

    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    def tokenize(example):

        return tokenizer(
            example[text_column],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenizer, model = fix_tokenizer_padding(tokenizer, model)
    tokenized_dataset = tokenize_dataset(tokenize, dataset, dataset_name)
    
    training_args = TrainingArguments(

        output_dir="./tmp",

        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        seed=42,

        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(tokenized_dataset["test"])

    # trainer.save_model(f"./pretrained/{model_name}")

    return predictions.metrics