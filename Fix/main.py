import os
import gc
import torch
import traceback
import importlib
from datetime import datetime

from csv_manipulator import write_csv_header, append_csv


# Read files

MODELS_FILE = "models.txt"
DATASETS_FILE = "datasets.txt"

TRAINERS = [
    "trainer_encoder",
    "trainer_decoder"
]

MODELS = []
DATASETS = []
TEXT_COLUMNS = {}

with open(MODELS_FILE, 'r') as f:
    MODELS = f.read().splitlines()

with open(DATASETS_FILE, 'r') as f:
    TEXT_COLUMNS = {line.split(': ', 1)[0].strip(): line.split(': ', 1)[1].strip() 
                 for line in f if ': ' in line}
    DATASETS = list(TEXT_COLUMNS.keys())

# Hardcoded

OUTPUT_DIR = "./results"
CSV_FILE = "experiment_results.csv"
# os.environ["HF_TOKEN"] = TOKEN

# Main

os.makedirs(OUTPUT_DIR, exist_ok=True)

write_csv_header(OUTPUT_DIR, CSV_FILE)

# Select trainer

print("\nAvailable trainers:")

for i, trainer in enumerate(TRAINERS):
    print(f"{i}: {trainer}.py")

try:
    trainer_index = int(input("Enter trainer index: "))

    if not 0 <= trainer_index < len(TRAINERS):
        raise IndexError

    trainer_name = TRAINERS[trainer_index]

    trainer_module = importlib.import_module(trainer_name)
    run_experiment = trainer_module.run_experiment

    print(f"You chose: {trainer_name}.py")

except (ValueError, IndexError):
    print("Invalid trainer index.")
    exit(1)

# Select dataset

for dataset_name in DATASETS:
    print(f"{DATASETS.index(dataset_name)}: {dataset_name}")
    
dataset_index = int(input("Enter dataset index: "))

try:
    dataset_name = DATASETS[dataset_index]
    print(f"You chose: {dataset_name}")

    for model_name in MODELS:

        start_time = datetime.now()
        text_column = TEXT_COLUMNS[dataset_name]

        try:

            metrics = run_experiment(
                model_name,
                dataset_name,
                text_column
            )

            append_csv([
                datetime.now(),
                model_name,
                dataset_name,
                datetime.now() - start_time,
                metrics["test_loss"],
                metrics["test_codebleu"],
                metrics["test_ngram_match_score"],
                metrics["test_weighted_ngram_match_score"],
                metrics["test_syntax_match_score"],
                metrics["test_dataflow_match_score"],
                metrics["test_runtime"],
                "success",
                ""
            ],
            OUTPUT_DIR, CSV_FILE)

        except Exception as e:

            print("ERROR:", e)
            print(traceback.format_exc())

            append_csv([
                datetime.now(),
                model_name,
                dataset_name,
                datetime.now() - start_time,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "error",
                str(e)
            ],
            OUTPUT_DIR, CSV_FILE)

        finally:

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll experiments completed.")

except (ValueError, IndexError):
    print("Invalid index.")