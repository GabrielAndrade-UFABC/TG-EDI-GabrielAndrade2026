# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def compute_metrics(pred):

#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels,
#         preds,
#         average="binary"
#     )

#     acc = accuracy_score(labels, preds)

#     return {
#         "accuracy": acc,
#         "f1": f1,
#         "precision": precision,
#         "recall": recall
#     }

import numpy as np
from evaluate import load

bleu_metric = load("bleu")


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )

    exact_match = np.mean([
        pred.strip() == label.strip()
        for pred, label in zip(decoded_preds, decoded_labels)
    ])

    return {
        "bleu": bleu["bleu"],
        "exact_match": exact_match
    }