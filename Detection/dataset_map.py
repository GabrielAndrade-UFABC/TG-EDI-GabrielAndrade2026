def map_BigVul(tokenized_dataset):
    tokenized_dataset = tokenized_dataset.rename_column("vul","labels")
    # tokenized_dataset = tokenized_dataset.rename_column("func_after","labels")
    return tokenized_dataset

def map_DiverseVul_PrimeVul(tokenized_dataset):
    # def create_label(example):
    #     cwe = example["cwe"]

    #     if cwe is None:
    #         return 0

    #     if isinstance(cwe, list):
    #         return int(len(cwe) > 0)

    #     if isinstance(cwe, str):
    #         cwe = cwe.strip()
            
    #         if cwe == "" or cwe == "[]":
    #             return 0
    #         return 1

    #     return 0

    # tokenized_dataset = tokenized_dataset.map(
    #     lambda x: {
    #         "labels": create_label(x)
    #     }
    # )

    tokenized_dataset = tokenized_dataset.rename_column("target","labels")

    return tokenized_dataset