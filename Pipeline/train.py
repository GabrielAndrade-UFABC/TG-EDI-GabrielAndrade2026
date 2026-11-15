"""Treino LoRA das três fases, com os mesmos prompts, a mesma inferência e as mesmas métricas da pipeline.

    python train.py --smoke                        # teste rápido de todas as combinações
    python train.py --phase fix --model gpt2       # um treino
    python train.py                                # 3 fases x 4 decoders
    python train.py --phase cwe fix --source cvefixes bigvul --output checkpoints/bigvul
    python train.py --eval-only --phase fix --model gpt2   # reavalia um checkpoint, sem treinar
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pipeline

ROOT = Path(__file__).resolve().parent
# Reduz a fragmentação nos 6 GB da 2060; precisa valer antes do torch iniciar a CUDA.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
PHASES = pipeline.PHASES
DECODERS = ('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'NinedayWang/PolyCoder-160M',
            'google/gemma-3-270m-it', 'openai-community/gpt2')
C_LANGUAGES = {'C', 'C++'}
MAX_CHARS = 2000


def encode_generation(tokenizer, prompt, target, max_length):
    """Loss só na resposta (+EOS). Exemplo grande demais é descartado: truncar cortaria a resposta."""
    # O prompt usa a mesma chamada de LocalModel.encode (com tokens especiais, ex.: BOS do gemma).
    prompt_ids = tokenizer(prompt)['input_ids']
    target_ids = tokenizer(target, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_length:
        return None
    return {'input_ids': prompt_ids + target_ids, 'attention_mask': [1] * (len(prompt_ids) + len(target_ids)),
            'labels': [-100] * len(prompt_ids) + target_ids}


def cve_record(row):
    """Linha do CVEfixes -> campos da pipeline; None para outra linguagem, NVD-CWE-* ou dados vazios."""
    cwe_id, description = row.get('cwe_id'), row.get('cwe_description')
    code, fixed = row.get('vulnerable_code'), row.get('fixed_code')
    if row.get('language') not in C_LANGUAGES or not isinstance(cwe_id, str) or not re.fullmatch(r'CWE-\d+', cwe_id):
        return None
    if not all(isinstance(v, str) and v.strip() for v in (description, code, fixed)):
        return None
    # Corte por caracteres, igual para todos os tokenizers: o mesmo teste cabe no contexto de todos (gpt2 = 1024).
    if len(code) + len(fixed) > MAX_CHARS:
        return None
    # A pipeline lê a CWE só da primeira linha da resposta.
    return {'code': code, 'cwe_output': f"{cwe_id}: {' '.join(description.split())}",
            'cwe_ids': [cwe_id], 'fixed_code': fixed}


def bigvul_record(row):
    """Linha vulnerável do BigVul -> mesmos campos do CVEfixes; descrição pelo ID (cwe_descriptions.json)."""
    cwe_id, code, fixed = row.get('CWE ID'), row.get('func_before'), row.get('func_after')
    description = pipeline.CWE_DESCRIPTIONS.get(cwe_id)
    if row.get('vul') != 1 or not description or not all(isinstance(v, str) and v.strip() for v in (code, fixed)):
        return None
    # Par sem mudança real ensinaria a copiar a entrada; o corte de tamanho é o mesmo do CVEfixes.
    if code.split() == fixed.split() or len(code) + len(fixed) > MAX_CHARS:
        return None
    return {'code': code, 'cwe_output': f'{cwe_id}: {description}', 'cwe_ids': [cwe_id], 'fixed_code': fixed}


def drop_seen(rows, test_codes):
    """Tira do treino o que também está no teste, ignorando espaços."""
    seen = {''.join(code.split()) for code in test_codes}
    return [r for r in rows if ''.join(r['code'].split()) not in seen]


def balanced(rows, size, seed):
    rng = random.Random(seed)
    groups = [[r for r in rows if r['label'] == label] for label in (0, 1)]
    for group in groups:
        rng.shuffle(group)
    take = min(size // 2, *map(len, groups))
    picked = groups[0][:take] + groups[1][:take]
    rng.shuffle(picked)
    return picked


def split(rows, max_train, max_test, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    first, second = int(len(rows) * 0.8), int(len(rows) * 0.9)
    return rows[:first][:max_train], rows[first:second][:max_test], rows[second:][:max_test]


def load_splits(phase, max_train, max_test, seed, sources=('cvefixes',), max_detection_test=0):
    from datasets import load_dataset
    if phase == 'detection':
        data = load_dataset('colin/PrimeVul')
        # PrimeVul preenche `cwe` também nas funções seguras; o rótulo oficial é `target`.
        rows = {name: [{'code': r['func'], 'label': int(r['target'])} for r in data[name]]
                for name in ('train', 'validation', 'test')}
        # Teste na proporção real (~2% vulneráveis): um teste balanceado infla precisão e F1.
        test = rows['test']
        random.Random(seed).shuffle(test)
        return (balanced(rows['train'], max_train, seed), balanced(rows['validation'], max_test, seed),
                test[:max_detection_test or None])
    parts, test_codes = [], []
    if 'cvefixes' in sources:
        records = [r for r in map(cve_record, load_dataset('hitoshura25/cvefixes')['train']) if r]
        parts.append(split(records, len(records), len(records), seed))
    if 'bigvul' in sources:
        data = load_dataset('bstee615/bigvul')
        columns = ['vul', 'CWE ID', 'func_before', 'func_after']
        # Splits oficiais; o teste inteiro (seguras também) fica fora do treino, pois a avaliação de ponta a ponta sai dele.
        parts.append(tuple([r for r in map(bigvul_record, data[name].select_columns(columns)) if r]
                           for name in ('train', 'validation', 'test')))
        test_codes += data['test']['func_before']
    train_rows, validation_rows, test_rows = ([r for part in parts for r in part[i]] for i in range(3))
    if len(parts) > 1:
        rng = random.Random(seed)
        for rows in (train_rows, validation_rows, test_rows):
            rng.shuffle(rows)
    test_codes += [r['code'] for r in test_rows]
    return (drop_seen(train_rows, test_codes)[:max_train], drop_seen(validation_rows, test_codes)[:max_test],
            test_rows[:max_test])


def prompt_target(phase, record):
    if phase == 'cwe':
        return pipeline.cwe_prompt(record['code']), record['cwe_output']
    return pipeline.fix_prompt(record['code'], record['cwe_output']), record['fixed_code']


def tokenize(phase, tokenizer, rows, max_length):
    from datasets import Dataset
    if phase == 'detection':
        # Mesmo corte de LocalModel.encode na inferência.
        encoded = [{**tokenizer(r['code'], truncation=True, max_length=512), 'labels': r['label']} for r in rows]
    else:
        encoded = [e for e in (encode_generation(tokenizer, *prompt_target(phase, r), max_length) for r in rows) if e]
    print(f'{phase}: {len(encoded)} de {len(rows)} exemplos cabem em {max_length} tokens', file=sys.stderr)
    return Dataset.from_list(encoded)


def build(phase, model_name):
    # fp32 explícito: o transformers carregaria Qwen em bf16, que a 2060 (Turing) não treina com fp16 AMP.
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorForSeq2Seq, DataCollatorWithPadding)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if phase == 'detection':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, dtype=torch.float32)
        model.config.pad_token_id = tokenizer.pad_token_id
        collator, task = DataCollatorWithPadding(tokenizer), TaskType.SEQ_CLS
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        collator, task = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100), TaskType.CAUSAL_LM
    model.enable_input_require_grads()  # gradient checkpointing com LoRA
    model = get_peft_model(model, LoraConfig(task_type=task, r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules='all-linear'))
    return tokenizer, model, collator


def evaluate(phase, path, test, device, max_new_tokens):
    """Avalia o checkpoint salvo pelo mesmo caminho da pipeline (LocalModel + predict)."""
    model = pipeline.LocalModel(path, phase, device, max_new_tokens)
    pairs, failures = [], Counter()
    try:
        for index, record in enumerate(test):
            if index % pipeline.BATCH_SIZE == 0:
                model.prefetch([pipeline.model_input(r, phase) for r in test[index:index + pipeline.BATCH_SIZE]])
            try:
                pairs.append((record, pipeline.predict(dict(record), phase, model, 0.5)))
            except Exception as exc:  # entrada longa demais ou resposta vazia
                failures[str(exc)[:80]] += 1
                # Falha conta como resposta errada; tirá-la da média inflaria a métrica.
                pairs.append((record, {'vulnerable': False, 'cwe_ids': [], 'cwe_output': '', 'fixed_code': ''}))
    finally:
        model.close()
    report = {'evaluated': len(pairs), 'failed': sum(failures.values()), 'failure_reasons': dict(failures)}
    if not pairs:
        return report
    if phase == 'detection':
        report.update(pipeline.detection_scores([(r['label'] == 1, out['vulnerable']) for r, out in pairs]))
    elif phase == 'cwe':
        report.update(pipeline.cwe_id_scores([(set(r['cwe_ids']), set(out['cwe_ids'])) for r, out in pairs]))
        report['bleu'] = pipeline.corpus_bleu([pipeline.description_text(r['cwe_output']) for r, _ in pairs],
                                              [pipeline.description_text(out['cwe_output']) for _, out in pairs])['bleu']
    else:
        report.update(pipeline.codebleu_scores([r['fixed_code'] for r, _ in pairs],
                                               [out['fixed_code'] for _, out in pairs], 'c'))
    return report


def baselines(phase, train_rows, test):
    """Resposta trivial no mesmo teste: o modelo só demonstra aprendizado se passar dela."""
    if phase == 'detection':
        return {'baseline_all_vulnerable': pipeline.detection_scores([(r['label'] == 1, True) for r in test])}
    if phase == 'cwe':
        top = Counter(r['cwe_ids'][0] for r in train_rows).most_common(1)[0][0]
        return {'baseline_majority_cwe': {'cwe': top, **pipeline.cwe_id_scores(
            [(set(r['cwe_ids']), {top}) for r in test])}}
    return {'baseline_copy_input': pipeline.codebleu_scores([r['fixed_code'] for r in test],
                                                            [r['code'] for r in test], 'c')}


def run(phase, model_name, data, args):
    import torch
    train_rows, _, test_rows = data
    out = args.output / phase / model_name.replace('/', '__')
    cuda = torch.cuda.is_available()
    result = {} if args.eval_only else train_model(phase, model_name, data, args, out, cuda)
    metrics = evaluate(phase, out, test_rows, 'cuda' if cuda else 'cpu', args.max_new_tokens[phase])
    return out, {**result, **metrics, **baselines(phase, train_rows, test_rows)}


def train_model(phase, model_name, data, args, out, cuda):
    import torch
    from transformers import Trainer, TrainingArguments
    train_rows, validation_rows, _ = data
    tokenizer, model, collator = build(phase, model_name)
    train_set = tokenize(phase, tokenizer, train_rows, args.max_length)
    validation_set = tokenize(phase, tokenizer, validation_rows, args.max_length) if validation_rows else None
    trainer = Trainer(model=model, data_collator=collator, train_dataset=train_set, eval_dataset=validation_set,
                      args=TrainingArguments(
                          output_dir=str(out / 'trainer'), num_train_epochs=args.epochs, max_steps=args.max_steps,
                          per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                          gradient_accumulation_steps=args.grad_accum, learning_rate=2e-4, warmup_ratio=0.03,
                          gradient_checkpointing=True, eval_strategy='epoch' if validation_set else 'no',
                          save_strategy='no', logging_steps=10, report_to='none', seed=args.seed,
                          # shortcut: gemma 3 transborda em fp16 e a 2060 não tem bf16; roda em fp32.
                          fp16=cuda and 'gemma' not in model_name.lower()))
    result = trainer.train()
    validation_loss = trainer.evaluate().get('eval_loss') if validation_set else None
    trainer.model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    del trainer, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return {'train_loss': result.training_loss, 'validation_loss': validation_loss}


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(['timestamp', 'phase', 'model', 'runtime_seconds', 'status', 'checkpoint', 'metrics', 'error'])
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', nargs='+', choices=PHASES, default=list(PHASES))
    parser.add_argument('--model', nargs='+', default=list(DECODERS))
    parser.add_argument('--epochs', type=float, default=2)
    parser.add_argument('--max-train', type=int, default=10000)
    parser.add_argument('--max-test', type=int, default=200)
    parser.add_argument('--max-test-detection', type=int, default=0,
                        help='Teste da Detection, na proporção real do PrimeVul (0 = teste inteiro).')
    parser.add_argument('--eval-only', action='store_true',
                        help='Não treina: só avalia os checkpoints já salvos em --output.')
    parser.add_argument('--max-length', type=int, default=768, help='Tokens de prompt + resposta (CWE/Fix).')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source', nargs='+', choices=('cvefixes', 'bigvul'), default=['cvefixes'],
                        help='Dados de CWE/Fix (a Detection usa sempre o PrimeVul).')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--smoke', action='store_true', help='Poucos exemplos e 10 passos, em checkpoints/smoke.')
    args = parser.parse_args(argv)
    args.max_steps = -1
    args.max_new_tokens = {'detection': 1, 'cwe': 64, 'fix': 384}
    if args.smoke:
        args.max_train, args.max_test, args.max_test_detection, args.max_steps = 64, 8, 64, 10
        args.output = args.output / 'smoke'
    results = args.output / 'results.csv'
    failures = 0
    for phase in args.phase:
        data = load_splits(phase, args.max_train, args.max_test, args.seed, args.source, args.max_test_detection)
        print(f'{phase}: {len(data[0])} treino, {len(data[1])} validação, {len(data[2])} teste', file=sys.stderr)
        for model_name in args.model:
            started = perf_counter()
            try:
                out, metrics = run(phase, model_name, data, args)
                row = ['success', str(out), metrics, '']
                print(f'{phase}/{model_name}: {metrics}', file=sys.stderr)
            except Exception as exc:
                failures += 1
                traceback.print_exc()
                row = ['error', '', '', f'{type(exc).__name__}: {exc}']
            append_result(results, [datetime.now().isoformat(timespec='seconds'), phase, model_name,
                                    round(perf_counter() - started, 1), *row])
    print(f'Resultados: {results}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
