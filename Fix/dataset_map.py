from datasets import load_dataset, DatasetDict

def map_BigVul(tokenized_dataset):
    # tokenized_dataset = tokenized_dataset.rename_column("vul","labels")
    tokenized_dataset = tokenized_dataset.rename_column("func_after","labels")
    return tokenized_dataset

def map_DiverseVul_PrimeVul(tokenized_dataset):
    def create_label(example):
        cwe = example["cwe"]

        if cwe is None:
            return 0

        if isinstance(cwe, list):
            return int(len(cwe) > 0)

        if isinstance(cwe, str):
            cwe = cwe.strip()
            
            if cwe == "" or cwe == "[]":
                return 0
            return 1

        return 0

    tokenized_dataset = tokenized_dataset.map(
        lambda x: {
            "labels": create_label(x)
        }
    )

    return tokenized_dataset


def load_and_split_dataset(dataset_name):

    dataset = load_dataset(dataset_name)

    if (
        dataset_name == "hitoshura25/cvefixes" or 
        dataset_name == "hitoshura25/megavul" or 
        dataset_name == "hitoshura25/crossvul"
        ):

        if dataset_name == "hitoshura25/cvefixes":
            MAX_CHARS = 20000

            total = len(dataset["train"])

            dataset = dataset.filter(
                lambda x:
                    len(x["vulnerable_code"]) <= MAX_CHARS
                    and
                    len(x["fixed_code"]) <= MAX_CHARS
            )

            print(
                f"Removed {total - len(dataset['train'])} examples "
                f"({100*(total-len(dataset['train']))/total:.2f}%)"
            )

        # 80% train, 20% temp
        split_1 = dataset["train"].train_test_split(
            test_size=0.2,
            seed=42,
            shuffle=True
        )

        # 10% validation, 10% test
        split_2 = split_1["test"].train_test_split(
            test_size=0.5,
            seed=42,
            shuffle=True
        )

        dataset = DatasetDict({
            "train": split_1["train"],
            "validation": split_2["train"],
            "test": split_2["test"]
        })

    return dataset