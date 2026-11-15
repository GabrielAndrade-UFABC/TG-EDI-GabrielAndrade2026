import os
import csv

def write_csv_header(output_dir, csv_file):

    path = os.path.join(output_dir, csv_file)

    if not os.path.exists(path):

        with open(path, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "model",
                "dataset",
                "runtime",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "test_runtime",
                "status",
                "error"
            ])


def append_csv(row, output_dir, csv_file):

    path = os.path.join(output_dir, csv_file)

    with open(path, "a", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(row)