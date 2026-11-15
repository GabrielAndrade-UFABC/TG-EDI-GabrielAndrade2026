from dataset_map import map_BigVul, map_DiverseVul_PrimeVul

def fix_tokenizer_padding(tokenizer, model):

    if tokenizer.pad_token is None:

        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def tokenize_dataset(tokenize, dataset, dataset_name):

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # hardcoded fix
    match dataset_name:
        case "bstee615/bigvul":
            tokenized_dataset = map_BigVul(tokenized_dataset)
        case "bstee615/diversevul":
            tokenized_dataset = map_DiverseVul_PrimeVul(tokenized_dataset)
        case "colin/PrimeVul":
            tokenized_dataset = map_DiverseVul_PrimeVul(tokenized_dataset)
        case _:
            raise ValueError("Unidentified dataset.")

    tokenized_dataset.set_format("torch")

    return tokenized_dataset