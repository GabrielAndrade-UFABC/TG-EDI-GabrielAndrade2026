"""Inferência local: Detection encaminha código vulnerável para CWE e Fix."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from contextlib import contextmanager
from html import escape
from urllib.parse import urlsplit

PHASES = ('detection', 'cwe', 'fix')
DEFAULT_INPUT_TOKEN_LIMITS = {'detection': 512, 'cwe': 512, 'fix': 512}
# Configurável por --batch-size; a geração usa precisão reduzida somente em CUDA.
BATCH_SIZE = 16
SAFE_CODE_REASON = 'Detection classificou o código como não vulnerável.'
PARQUET_DATASETS = ('hitoshura25/cvefixes', 'hitoshura25/megavul', 'hitoshura25/crossvul')
CWE_LABEL_DATASETS = ('bstee615/diversevul', 'colin/PrimeVul')

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    # Gerados pelo train.py; melhores por fase em checkpoints/results.csv.
    'detection': ROOT / 'checkpoints/detection/PolyCoder-160M-bigvul',
    'cwe': ROOT / 'checkpoints/cwe/Qwen2.5-Coder-0.5B-Instruct-cvefixes',
    'fix': ROOT / 'checkpoints/fix/Qwen2.5-Coder-0.5B-Instruct-bigvul',
}
DEFAULT_INPUT = ROOT / 'examples/pipeline_input.json'


def hub_dataset_id(source):
    if str(source) in (*PARQUET_DATASETS, *CWE_LABEL_DATASETS):
        return str(source)
    url = urlsplit(str(source))
    if url.scheme == 'https' and url.netloc == 'huggingface.co':
        match = re.fullmatch(r'/datasets/([^/]+/[^/]+)(?:/(?:tree|viewer)(?:/.*)?|/)?', url.path)
        if match:
            return match[1]
    return None


def sequential_test_split(train):
    test = train.train_test_split(test_size=0.2, seed=42)['test']
    return test.train_test_split(test_size=0.5, seed=42)['test']


def load_hub_dataset(repo_id):
    from datasets import load_dataset

    if repo_id in PARQUET_DATASETS:
        train = load_dataset('parquet', data_files=f'hf://datasets/{repo_id}/data/train-*.parquet',
                             split='train')
        return sequential_test_split(train)
    splits = load_dataset(repo_id)
    dataset = splits['test'] if 'test' in splits else splits['train']
    if repo_id in CWE_LABEL_DATASETS:
        if 'target' not in dataset.column_names:
            raise ValueError(f'Dataset {repo_id}: coluna oficial target ausente.')
    return dataset


def read_input_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        return data if isinstance(data, list) else [data]
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == '.parquet':
        # Arquivo local: leitura direta, sem consultar o Hub nem criar cache de dataset.
        import pyarrow.parquet as parquet
        return parquet.read_table(path).to_pylist()
    return None


def has_code(value):
    return isinstance(value, str) and bool(value.strip())


def read_records(path, code_column='code'):
    rows = read_input_rows(path)
    if rows is None:
        path = Path(path)
        rows = [{'id': path.name, code_column: path.read_text(encoding='utf-8')}]
    return records_from_rows(rows, code_column)


def records_from_rows(rows, code_column):
    if not rows:
        raise ValueError('A entrada está vazia.')
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not has_code(row.get(code_column)):
            raise ValueError(f'Registro {index}: coluna {code_column!r} ausente ou código vazio.')
        records.append({'id': row.get('id', str(index)), 'code': row[code_column],
                        'vulnerable': None, 'vulnerability_probability': None,
                        'cwe_ids': [], 'cwe_output': None, 'fixed_code': None,
                        'status': 'pending', 'error': None, 'skip_reason': None,
                        'validation_status': 'not_performed', 'phase_results': {}})
    validate_ids(records)
    return records


def valid_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_ids(records):
    if any(not isinstance(r['id'], (str, int)) or isinstance(r['id'], bool) or not str(r['id']).strip() for r in records):
        raise ValueError('Cada registro precisa de um ID textual ou numérico não vazio.')
    identifiers = [str(r['id']) for r in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('IDs duplicados na entrada; cada código precisa de um ID único.')


def read_checkpoint(path):
    """Retoma apenas resultados da pipeline; preserva previsões anteriores."""
    records = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(records, list) or not records:
        raise ValueError('Checkpoint deve conter uma lista não vazia de resultados.')
    required = {'id', 'code', 'status', 'vulnerable', 'vulnerability_probability',
                'cwe_ids', 'cwe_output', 'fixed_code', 'error'}
    statuses = {'pending', 'detected', 'classified', 'not_vulnerable',
                'cwe_unidentified', 'generated_unvalidated', 'error', 'completed'}
    for index, r in enumerate(records):
        if not isinstance(r, dict) or not required.issubset(r):
            raise ValueError(f'Registro {index}: não é um resultado da pipeline.')
        if not isinstance(r['code'], str) or not r['code'].strip() or not isinstance(r['status'], str) or r['status'] not in statuses:
            raise ValueError(f'Registro {index}: código ou status inválido.')
        if not isinstance(r['cwe_ids'], list) or not all(isinstance(x, str) for x in r['cwe_ids']):
            raise ValueError(f'Registro {index}: cwe_ids inválido.')
        if any(not re.fullmatch(r'CWE-\d+', x) for x in r['cwe_ids']):
            raise ValueError(f'Registro {index}: identificador CWE inválido.')
        if r['vulnerable'] is not None and type(r['vulnerable']) is not bool:
            raise ValueError(f'Registro {index}: vulnerable deve ser booleano ou nulo.')
        probability = r['vulnerability_probability']
        if probability is not None and (not valid_probability(probability)):
            raise ValueError(f'Registro {index}: probabilidade inválida.')
        for field in ('cwe_output', 'cwe_raw_output', 'fixed_code', 'skip_reason'):
            if r.get(field) is not None and not isinstance(r[field], str):
                raise ValueError(f'Registro {index}: {field} deve ser texto ou nulo.')
        if r['status'] != 'pending' and r['status'] != 'error' and (r['vulnerable'] is None or probability is None):
            raise ValueError(f'Registro {index}: decisão de Detection ausente.')
        if r['status'] == 'pending' and (r['vulnerable'] is not None or probability is not None or r['cwe_ids'] or r['cwe_output'] is not None or r['fixed_code'] is not None):
            raise ValueError(f'Registro {index}: resultado pendente contém previsões.')
        if r['status'] == 'not_vulnerable' and r['vulnerable'] is not False:
            raise ValueError(f'Registro {index}: decisão de Detection inconsistente.')
        if r['status'] == 'cwe_unidentified' and (r['vulnerable'] is not True or r['cwe_ids'] or r['cwe_output'] is None):
            raise ValueError(f'Registro {index}: CWE não identificada inconsistente.')
        if r['status'] in {'completed', 'generated_unvalidated'} and (r['vulnerable'] is not True or not has_code(r['fixed_code'])):
            raise ValueError(f'Registro {index}: geração sem detecção positiva ou código corrigido.')
        has_cwe_error = isinstance(r['error'], dict) and r['error'].get('stage') == 'cwe'
        if r['fixed_code'] is not None and not (r['status'] in {'completed', 'generated_unvalidated'}
                                                or (r['status'] == 'error' and has_cwe_error and r['vulnerable'] is True)):
            raise ValueError(f'Registro {index}: código gerado incompatível com o status.')
        if r['status'] in {'detected', 'classified'} and r['vulnerable'] is not True:
            raise ValueError(f'Registro {index}: resultado de Detection inconsistente.')
        if r['status'] == 'classified' and (not r['cwe_ids'] or not isinstance(r['cwe_output'], str) or not r['cwe_output'].strip()):
            raise ValueError(f'Registro {index}: resultado de CWE inconsistente.')
        if r['error'] is not None and (not isinstance(r['error'], dict) or r['error'].get('stage') not in PHASES or not isinstance(r['error'].get('message'), str)):
            raise ValueError(f'Registro {index}: erro inválido.')
        if (r['status'] == 'error') != (r['error'] is not None):
            raise ValueError(f'Registro {index}: status e erro inconsistentes.')
        if r['status'] == 'completed':
            r['status'] = 'generated_unvalidated'
        r.setdefault('skip_reason', None)
        r.setdefault('validation_status', 'not_performed')
        r.setdefault('phase_results', {})
        if not isinstance(r['phase_results'], dict):
            raise ValueError(f'Registro {index}: phase_results inválido.')
        for kind, result in list(r['phase_results'].items()):
            allowed = {'pending', 'success', 'error', 'skipped'} | ({'unidentified'} if kind == 'cwe' else set())
            if kind not in PHASES or not isinstance(result, dict) or not isinstance(result.get('status'), str) or result['status'] not in allowed:
                raise ValueError(f'Registro {index}: histórico de fase inválido.')
            if result.get('reason') is not None and not isinstance(result['reason'], str):
                raise ValueError(f'Registro {index}: motivo de fase inválido.')
            # Checkpoints antigos registravam Fix como pulado quando CWE falhava ou não retornava ID.
            if kind == 'fix' and result['status'] == 'skipped' and r['vulnerable'] is True and not r['fixed_code'] and (
                    has_cwe_error or r['status'] == 'cwe_unidentified'):
                del r['phase_results'][kind]
                r['skip_reason'] = None
                continue
            # Se ambas as etapas falharam, o erro de Fix ocupa o campo agregado.
            if (kind == 'cwe' and result['status'] == 'error' and (r['error'] or {}).get('stage') == 'fix'
                    and r['vulnerable'] is True and r['cwe_output'] is None and not r['cwe_ids']
                    and isinstance(result.get('reason'), str)):
                continue
            expected, _ = phase_outcome({**r, 'phase_results': {}}, kind)
            if result['status'] != expected:
                raise ValueError(f'Registro {index}: histórico de {kind} inconsistente.')
    validate_ids(records)
    return records


def reset_errors(records):
    """Reabre as etapas que falharam, preservando as previsões das demais."""
    count = 0
    for record in records:
        if record['status'] != 'error':
            continue
        stage = record['error']['stage']
        history = record['phase_results']
        for kind in PHASES:
            if kind == stage or history.get(kind, {}).get('status') == 'error' or stage == 'detection':
                history.pop(kind, None)
        if record['fixed_code']:
            status = 'generated_unvalidated'
        elif record['cwe_ids']:
            status = 'classified'
        elif record['cwe_output'] is not None:
            status = 'cwe_unidentified'
        else:
            status = 'detected' if record['vulnerable'] is True else 'pending'
        record.update(status=status, error=None, skip_reason=None)
        count += 1
    return count


def phase_outcome(record, kind):
    """Usa o histórico da etapa, mesmo após falha em uma etapa posterior."""
    saved = record.get('phase_results', {}).get(kind)
    if saved:
        return saved['status'], saved.get('reason')
    error = record.get('error') or {}
    if error.get('stage') == kind:
        return 'error', error['message']
    if error.get('stage') == 'detection':
        return 'skipped', f"Falha na etapa {error['stage']}: {error['message']}"
    if kind == 'detection':
        return ('success', None) if record.get('vulnerable') is not None else ('pending', None)
    if record.get('vulnerable') is False:
        return 'skipped', SAFE_CODE_REASON
    if kind == 'cwe':
        if record.get('cwe_ids'):
            return 'success', None
        if record.get('cwe_output') is not None:
            reason = ('Resposta sem identificador CWE-N.' if (record.get('cwe_raw_output') or record['cwe_output']).strip()
                      else 'O modelo retornou uma resposta vazia.')
            return 'unidentified', reason
    if kind == 'fix' and record.get('fixed_code'):
        return 'success', None
    return 'pending', None


@contextmanager
def atomic_text(path, encoding='utf-8', newline=None):
    """Temporário exclusivo evita colisões e escrita por links .tmp existentes."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline=newline,
                                         prefix=f'.{path.name}.', suffix='.tmp',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, data):
    with atomic_text(path) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def spreadsheet_cell(value):
    """Exporta conteúdo como texto quando poderia ser interpretado como fórmula."""
    if isinstance(value, str) and (value.startswith(('\t', '\r', '\n')) or value.lstrip().startswith(('=', '+', '-', '@'))):
        return "'" + value
    return value


CWE_CODE_HEADER = '\n    ### Vulnerable code:\n    '
CWE_ANSWER_MARKER = '### Identify the CWE:'
CWE_ANSWER_HEADER = f'\n\n    {CWE_ANSWER_MARKER}\n    '
LEGACY_CWE_CHECKPOINTS = {'codeberta-small-v1-cvefixes', 'graphcodebert-base-cvefixes'}
FIX_CODE_HEADER = '\n\n    ### Buggy code:\n    '
FIX_ANSWER_HEADER = '\n\n    ### Fixed code:\n    '


def cwe_prompt(code):
    return f'{CWE_CODE_HEADER}{code}{CWE_ANSWER_HEADER}'


def fix_prompt(code, cwe_output=''):
    # Os checkpoints originais foram treinados sem o cabeçalho CWE.
    prefix = f'\n    ### CWE:\n    {cwe_output}' if cwe_output else ''
    header = FIX_CODE_HEADER if prefix else FIX_CODE_HEADER[1:]
    return f'{prefix}{header}{code}{FIX_ANSWER_HEADER}'


def model_token_limits(context, kind, requested_input, requested_output):
    """Divide a janela física entre prompt e geração sem zerar o prompt.

    Quando os limites não cabem juntos, reserva até metade da janela para
    entrada e reduz a saída. Entradas menores permitem uma saída maior.
    """
    if min(context, requested_input, requested_output) < 1:
        raise ValueError('Contexto e limites de entrada/saída devem ser positivos.')
    if kind == 'detection':
        return min(context, requested_input), requested_output
    if context < 2:
        raise ValueError(f'{kind}: janela de {context} token insuficiente para entrada e saída.')
    reserved_input = min(requested_input, context // 2)
    output = min(requested_output, context - reserved_input)
    return min(requested_input, context - output), output


class LocalModel:
    def __init__(self, path, kind, device='auto', max_new_tokens=128, max_input_tokens=None, precision='auto'):
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
        self.torch = torch
        self.kind = kind
        self.max_new_tokens = max_new_tokens
        self.use_cache = True
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA indisponível; use --device cpu.')
        path = str(Path(path).resolve())
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.legacy_cwe = (kind == 'cwe' and config.model_type == 'roberta' and not config.is_decoder
                           and Path(path).name.casefold() in LEGACY_CWE_CHECKPOINTS)
        if self.legacy_cwe:
            self.use_cache = False
        if kind == 'fix' and config.model_type == 'roberta' and not config.is_decoder:
            print('Aviso: Fix foi salvo com is_decoder=False; ativando atenção causal em memória. A qualidade precisa ser validada.', file=sys.stderr)
            config.is_decoder = True
            self.use_cache = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer não possui pad_token nem eos_token.')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # RoBERTa reserva posições para padding; demais modelos usam seu limite explícito.
        context = getattr(config, 'max_position_embeddings', 2048)
        if config.model_type == 'roberta':
            context -= (config.pad_token_id or 0) + 1
        if max_input_tokens is None:
            max_input_tokens = DEFAULT_INPUT_TOKEN_LIMITS[kind]
        self.context_token_limit = context
        self.input_token_limit, self.max_new_tokens = model_token_limits(
            context, kind, max_input_tokens, max_new_tokens)
        if self.input_token_limit <= self.tokenizer.num_special_tokens_to_add():
            raise ValueError(f'{kind}: limite de entrada de {self.input_token_limit} tokens insuficiente para o modelo.')
        if self.max_new_tokens != max_new_tokens:
            print(f'{kind}: janela total de {context} tokens; ajustando os limites para '
                  f'{self.input_token_limit} de entrada e {self.max_new_tokens} de saída '
                  f'(saída solicitada: {max_new_tokens}).', file=sys.stderr)
        cls = AutoModelForSequenceClassification if kind == 'detection' else AutoModelForCausalLM
        if precision == 'auto':
            precision = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if (
                self.device == 'cuda' and kind != 'detection') else 'float32'
        if self.device == 'cpu' and precision == 'float16':
            raise ValueError('float16 requer CUDA; em CPU use auto ou float32.')
        self.precision = precision
        self.model = cls.from_pretrained(path, config=config, local_files_only=True,
                                         trust_remote_code=False, use_safetensors=True,
                                         torch_dtype=getattr(torch, precision)).to(self.device).eval()
        self.generation_options = {}
        self.generation_eos_token_id = self.tokenizer.eos_token_id
        if kind != 'detection':
            from transformers import StopStringCriteria, StoppingCriteriaList
            stops = ['\n'] if kind == 'cwe' else ['\n### ', '\n    ### ']
            if not self.legacy_cwe:
                self.generation_options['stopping_criteria'] = StoppingCriteriaList([
                    StopStringCriteria(self.tokenizer, stop_strings=stops)])
            # Os checkpoints antigos aprenderam PAD como fim de resposta. Parar
            # nele preserva o texto e evita continuar gerando padding até o limite.
            pad, eos = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
            if pad is not None and pad != eos:
                self.generation_eos_token_id = [eos, pad] if eos is not None else pad
            if kind == 'fix':
                # A regra de tamanho mínimo bloqueia todos os tokens de fim no primeiro passo.
                self.generation_options['min_new_tokens'] = 1
            if self.legacy_cwe:
                # Estes tokenizadores byte-level representam espaços, tabs e quebras com estes símbolos.
                whitespace = {idx for token, idx in self.tokenizer.get_vocab().items()
                              if token and not token.strip('ĠĊĉ \t\r\n')}
                end_ids = {pad, eos}
                self.cwe_retry_suppressed_tokens = sorted((whitespace | set(self.tokenizer.all_special_ids)) - end_ids)
        self.cache = {}

    def generate_answers(self, tokens, *, retry=False):
        options = dict(self.generation_options)
        if retry:
            options['min_new_tokens'] = min(16, self.max_new_tokens)
            # Não encerra a segunda tentativa numa linha vazia inicial.
            options.pop('stopping_criteria', None)
            if self.kind == 'cwe':
                options.update(min_new_tokens=min(8, self.max_new_tokens), repetition_penalty=1.2,
                               no_repeat_ngram_size=2, suppress_tokens=self.cwe_retry_suppressed_tokens)
        output = self.model.generate(**tokens, max_new_tokens=self.max_new_tokens,
                                     do_sample=False, use_cache=self.use_cache,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.generation_eos_token_id, **options)
        return [self.answer(row) for row in output[:, tokens['input_ids'].shape[1]:]]

    def needs_generation_retry(self, answer):
        return (self.kind == 'fix' or getattr(self, 'legacy_cwe', False)) and not answer.strip()

    def prefetch(self, texts):
        """Roda um lote de uma vez. Detect e generate leem o resultado daqui."""
        self.cache = {}
        texts = list(dict.fromkeys(texts))
        if not texts:
            return
        tokens = self.encode(texts)
        with self.torch.inference_mode():
            if self.kind == 'detection':
                scores = self.model(**tokens).logits.softmax(dim=-1)
                if scores.shape[-1] != 2:
                    return
                results = [float(row[1].item()) for row in scores]
            else:
                results = self.generate_answers(tokens)
                for i, answer in enumerate(results):
                    if self.needs_generation_retry(answer):
                        results[i] = self.generate_answers(self.encode([texts[i]]), retry=True)[0]
        self.cache = dict(zip(texts, results))

    def encode(self, texts):
        if self.kind == 'detection':
            self.tokenizer.padding_side = 'right'
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                                    max_length=self.input_token_limit)
        else:
            self.tokenizer.padding_side = 'left'
            batch = self.tokenizer(texts, verbose=False)
            for index, text in enumerate(texts):
                if len(batch['input_ids'][index]) > self.input_token_limit:
                    shortened = self.truncate_prompt(text)
                    for key in batch:
                        batch[key][index] = shortened[key]
            tokens = self.tokenizer.pad(batch, padding=True, return_tensors='pt')
        return tokens.to(self.device)

    def truncate_prompt(self, text):
        code_header, answer_header = ((CWE_CODE_HEADER, CWE_ANSWER_HEADER) if self.kind == 'cwe'
                                      else (FIX_CODE_HEADER, FIX_ANSWER_HEADER))
        intro, marker, body = text.partition(code_header)
        limit = self.input_token_limit
        if not marker or not body.endswith(answer_header):
            return self.tokenizer(text, truncation=True, max_length=limit)

        prefix = intro + marker
        code = body[:-len(answer_header)]
        framing = self.tokenizer(prefix + answer_header)
        budget = limit - len(framing['input_ids'])
        if budget < 1:
            raise ValueError(f'{self.kind}: limite de entrada de {limit} tokens insuficiente para o prompt e o código.')
        ids = self.tokenizer(code, add_special_tokens=False, truncation=True, max_length=budget)['input_ids']
        while ids:
            kept_code = self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            encoded = self.tokenizer(prefix + kept_code + answer_header)
            excess = len(encoded['input_ids']) - limit
            if excess <= 0:
                return encoded
            ids = ids[:-excess]
        raise ValueError(f'{self.kind}: limite de entrada de {limit} tokens insuficiente para o prompt e o código.')

    def detect(self, code):
        if code in self.cache:
            return self.cache[code]
        with self.torch.inference_mode():
            scores = self.model(**self.encode([code])).logits.softmax(dim=-1)[0]
        if scores.numel() != 2:
            raise ValueError('Detection precisa ter duas classes: 0=seguro, 1=vulnerável.')
        return float(scores[1].item())

    def generate(self, prompt):
        if prompt in self.cache:
            return self.cache[prompt]
        tokens = self.encode([prompt])
        with self.torch.inference_mode():
            answer = self.generate_answers(tokens)[0]
            if self.needs_generation_retry(answer):
                answer = self.generate_answers(tokens, retry=True)[0]
        return answer

    def answer(self, ids):
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if self.kind == 'cwe':
            return text
        if self.kind == 'fix':
            text = re.sub(r'^\s*### Fixed code:\s*', '', text)
        # Um modelo que não emite EOS continua o prompt com novos blocos '### ...'; isso não é resposta.
        return re.split(r'\n[ \t]*### ', text, maxsplit=1)[0].strip()

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def write_html_table(path, number, kind, columns, rows, counts):
    """Exibe o conteúdo original como texto, sem executar HTML das entradas."""
    headers = ''.join(f'<th>{escape(column)}</th>' for column in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(row[column])) if row[column] is not None else ""}</td>'
                                  for column in columns) + '</tr>' for row in rows)
    with atomic_text(path) as stream:
        stream.write('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                     '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
                     f'<title>Fase {number}: {kind}</title>'
                     '<style>body{font:14px system-ui;margin:24px;color:#17212b;background:#f5f7fa}'
                     'table{border-collapse:collapse;background:white}th,td{border:1px solid #ccd3dc;padding:10px;text-align:left;vertical-align:top}'
                     'th{position:sticky;top:0;background:#e8edf4}td{white-space:pre-wrap;min-width:110px;max-width:480px;overflow-wrap:anywhere}'
                     '.table{overflow:auto}tbody tr:nth-child(even){background:#f8fafc}</style></head><body>'
                     f'<h1>Fase {number}: {kind}</h1><p>{len(rows)} registros. '
                     f'Sucesso: {counts["success"]}; erros: {counts["error"]}; '
                     f'CWE não identificada: {counts["unidentified"]}; pulados: {counts["skipped"]}.</p>'
                     '<p>Sucesso indica execução da etapa. As previsões e correções não foram validadas.</p>'
                     f'<div class="table"><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div></body></html>')


def write_phase_results(directory, kind, records):
    """Snapshot independente, incluindo registros que não chegaram à fase."""
    number = {'detection': 1, 'cwe': 2, 'fix': 3}[kind]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{number:02d}_{kind}.csv'
    columns = ['id', 'phase_status', 'pipeline_status', 'code', 'vulnerable',
               'vulnerability_probability', 'cwe_ids', 'cwe_output', 'fixed_code',
               'skip_reason', 'validation_status', 'error_stage', 'error_message',
               'runtime_seconds', 'model_path', 'device']
    if kind == 'cwe' and any('cwe_raw_output' in record for record in records):
        columns.insert(columns.index('cwe_output') + 1, 'cwe_raw_output')
    rows = []
    for record in records:
        status, reason = phase_outcome(record, kind)
        phase = record.get('phase_results', {}).get(kind, {})
        row = {column: record.get(column) for column in columns}
        row.update(phase_status=status, pipeline_status=record['status'],
                   cwe_ids=', '.join(record['cwe_ids']), skip_reason=reason,
                   validation_status=record.get('validation_status', 'not_performed'),
                   error_stage=kind if status == 'error' else None,
                   error_message=reason if status == 'error' else None)
        row.update({key: phase.get(key) for key in ('runtime_seconds', 'model_path', 'device')})
        rows.append(row)
    # BOM e quoting preservam acentos e códigos com vírgulas/quebras de linha.
    with atomic_text(path, encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: spreadsheet_cell(value) for key, value in row.items()} for row in rows)
    json_path = path.with_suffix('.json')
    write_json(json_path, records)
    counts = Counter(row['phase_status'] for row in rows)
    html_path = path.with_suffix('.html')
    write_html_table(html_path, number, kind, columns, rows, counts)
    print(f"Fase {number} ({kind}) concluída: {counts['success']} sucesso(s), "
          f"{counts['error']} erro(s), {counts['unidentified']} sem CWE identificada, "
          f"{counts['skipped']} não executado(s).\nPlanilha: {path}\nTabela: {html_path}",
          file=sys.stderr)
    return path


# ----------------------------------------------------------------------------
# Métricas. Comparam as previsões da pipeline com rótulos esperados da entrada:
#   Detection -> F1 (classe positiva = vulnerável)
#   CWE (ID)  -> F1 multirrótulo, macro e micro
#   CWE (descrição) -> BLEU corpus (BLEU-4, tokenização 13a, sem suavização)
#   Fix       -> média por exemplo de CodeBLEU; toda saída ausente vale zero,
#                sem passar pelo fallback de dataflow do pacote.
# As métricas de CWE e Fix incluem o efeito da decisão de Detection.
# ----------------------------------------------------------------------------
REFERENCE_ALIASES = {
    'vulnerable': ('expected_vulnerable', 'target', 'vul', 'label', 'is_vulnerable'),
    'cwe_ids': ('expected_cwe_ids', 'cwe_id', 'cwe_ids', 'cwe', 'CWE ID'),
    'cwe_description': ('expected_cwe_description', 'cwe_description', 'CWE Description'),
    'fixed_code': ('expected_fixed_code', 'fixed_code', 'func_after'),
}
# ID -> descrição oficial, tirado do CVEfixes (o mesmo texto que o modelo de CWE aprendeu).
# Preenche a descrição esperada em conjuntos que só trazem o ID, como o BigVul.
_CWE_DESCRIPTIONS_FILE = ROOT / 'cwe_descriptions.json'
CWE_DESCRIPTIONS = json.loads(_CWE_DESCRIPTIONS_FILE.read_text(encoding='utf-8')) if _CWE_DESCRIPTIONS_FILE.is_file() else {}
METRICS_FOR_PHASE = {'detection': ('detection',), 'cwe': ('cwe_id', 'cwe_description'), 'fix': ('fix',)}
_TRUE_LABELS = {'1', 'true', 'yes', 'sim', 'vulnerable', 'vulneravel', 'vulnerável'}
_FALSE_LABELS = {'0', 'false', 'no', 'nao', 'não', 'safe', 'not_vulnerable', 'seguro'}


def parse_expected_vulnerable(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_LABELS:
        return True
    if text in _FALSE_LABELS:
        return False
    raise ValueError(f'rótulo de vulnerabilidade desconhecido: {value!r}')


def parse_cwe_ids(value):
    """'CWE-79', 79, 'CWE-79, CWE-20' ou ['CWE-79'] viram ['CWE-79', ...] (sem zeros à esquerda)."""
    if value is None:
        return []
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = ' '.join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
    found = re.findall(r'(?i)cwe[-_ ]?0*(\d+)', text)
    if not found and re.fullmatch(r'[\s\d,;\[\]\'"]*', text):
        found = re.findall(r'\d+', text)
    return list(dict.fromkeys(f'CWE-{int(n)}' for n in found))


def clean_cwe_answer(text, *, after_code=False):
    # O trainer descarta o prompt ecoado e recupera o prefixo WE- do GraphCodeBERT.
    completion = text.split(CWE_ANSWER_MARKER, 1)[-1].strip()
    pattern = r'(?<!\w)(?:CWE|WE)[ \t]*-[ \t]*(\d+)\b'
    if after_code:
        match = (re.search(r'^[ \t]*' + pattern, completion, re.IGNORECASE | re.MULTILINE)
                 or re.search(pattern + r'[ \t]*:', completion, re.IGNORECASE))
        if match is None:
            return '', []
        completion = completion[match.start():].lstrip()
    answer = completion.partition('\n')[0].strip()
    match = re.search(pattern, answer, re.IGNORECASE)
    if match is None:
        return answer, []
    answer = re.sub(pattern, lambda m: f'CWE-{int(m[1])}', answer[match.start():], flags=re.IGNORECASE)
    return answer, parse_cwe_ids(answer)


def description_text(value):
    """Só a descrição: tira 'CWE-N:' do início, igual dos dois lados da comparação."""
    if isinstance(value, (list, tuple)):
        value = ', '.join(str(v) for v in value)
    if not isinstance(value, str):
        return ''
    return ' '.join(re.sub(r'(?i)\bcwe[-_ ]?\d+\b\s*[:\-–]?\s*', ' ', value).split())


def read_references(path, code_column='code', *, columns=None):
    """Rótulos esperados por ID, lidos da entrada. Arquivo de código solto não tem rótulos."""
    rows = read_input_rows(path)
    return references_from_rows(rows, columns) if rows is not None else {}


def references_from_rows(rows, columns=None):
    aliases = {**REFERENCE_ALIASES, **{field: (name,) for field, name in (columns or {}).items()}}
    references = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if 'phase_results' in row or 'vulnerability_probability' in row:
            raise ValueError('O arquivo de referências parece um resultado da pipeline. '
                             'Use a entrada original, com os rótulos esperados.')
        identifier = str(row.get('id', index))
        if identifier in references:
            raise ValueError(f'ID duplicado nas referências: {identifier}.')
        expected = {}
        for field, names in aliases.items():
            name = next((n for n in names if row.get(n) not in (None, '', [])), None)
            if name is None:
                continue
            try:
                if field == 'vulnerable':
                    expected[field] = parse_expected_vulnerable(row[name])
                elif field == 'cwe_ids':
                    ids = parse_cwe_ids(row[name])
                    if ids:
                        expected[field] = ids
                elif field == 'cwe_description':
                    text = description_text(row[name])
                    if text:
                        expected[field] = text
                elif isinstance(row[name], str) and row[name].strip():
                    expected[field] = row[name]
            except ValueError as exc:
                raise ValueError(f'Referência {identifier}, coluna {name!r}: {exc}') from exc
        if expected.get('vulnerable') is False:
            # BigVul repete a CWE e o func_after do commit nas funções seguras; não são rótulos delas.
            for field in ('cwe_ids', 'cwe_description', 'fixed_code'):
                expected.pop(field, None)
        if 'cwe_ids' in expected and 'cwe_description' not in expected:
            known = [CWE_DESCRIPTIONS[i] for i in expected['cwe_ids'] if i in CWE_DESCRIPTIONS]
            if known:
                expected['cwe_description'] = description_text(known)
        if expected:
            references[identifier] = expected
    return references


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def detection_scores(pairs):
    """pairs: (esperado, previsto), ambos booleanos. Positivo = vulnerável."""
    tp = sum(1 for e, p in pairs if e and p)
    fp = sum(1 for e, p in pairs if not e and p)
    fn = sum(1 for e, p in pairs if e and not p)
    tn = len(pairs) - tp - fp - fn
    precision, recall, f1 = _prf(tp, fp, fn)
    return {'f1': f1, 'precision': precision, 'recall': recall,
            'accuracy': (tp + tn) / len(pairs), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def cwe_id_scores(pairs):
    """pairs: (conjunto esperado, conjunto previsto). Macro = média por CWE; micro = pares globais."""
    labels = sorted(set().union(*(e | p for e, p in pairs)))
    per_label = []
    totals = [0, 0, 0]
    for label in labels:
        tp = sum(1 for e, p in pairs if label in e and label in p)
        fp = sum(1 for e, p in pairs if label not in e and label in p)
        fn = sum(1 for e, p in pairs if label in e and label not in p)
        per_label.append(_prf(tp, fp, fn)[2])
        totals = [totals[0] + tp, totals[1] + fp, totals[2] + fn]
    precision, recall, micro = _prf(*totals)
    return {'f1_macro': sum(per_label) / len(per_label), 'f1_micro': micro,
            'precision_micro': precision, 'recall_micro': recall, 'classes': len(labels)}


_BLEU_RULES = [(re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
               (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
               (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
               (re.compile(r'([0-9])(-)'), r'\1 \2 ')]


def bleu_tokens(text):
    """Tokenização 13a (a mesma do `evaluate`/sacreBLEU)."""
    text = text.replace('<skipped>', '').replace('-\n', '').replace('\n', ' ')
    if '&' in text:
        text = (text.replace('&quot;', '"').replace('&amp;', '&')
                .replace('&lt;', '<').replace('&gt;', '>'))
    for rule, replacement in _BLEU_RULES:
        text = rule.sub(replacement, text)
    return text.split()


def corpus_bleu(references, hypotheses, max_order=4):
    """BLEU de corpus com uma referência por hipótese, sem suavização (como `evaluate.load('bleu')`)."""
    matches = [0] * max_order
    possible = [0] * max_order
    ref_len = hyp_len = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref, hyp = bleu_tokens(reference), bleu_tokens(hypothesis)
        ref_len += len(ref)
        hyp_len += len(hyp)
        for order in range(1, max_order + 1):
            ref_ngrams = Counter(tuple(ref[i:i + order]) for i in range(len(ref) - order + 1))
            hyp_ngrams = Counter(tuple(hyp[i:i + order]) for i in range(len(hyp) - order + 1))
            matches[order - 1] += sum((hyp_ngrams & ref_ngrams).values())
            possible[order - 1] += max(len(hyp) - order + 1, 0)
    precisions = [m / p if p else 0.0 for m, p in zip(matches, possible)]
    geometric = math.exp(sum(math.log(p) for p in precisions) / max_order) if min(precisions) > 0 else 0.0
    ratio = hyp_len / ref_len if ref_len else 0.0
    penalty = 1.0 if ratio > 1 else (math.exp(1 - 1 / ratio) if ratio > 0 else 0.0)
    return {'bleu': geometric * penalty, 'precisions': precisions, 'brevity_penalty': penalty,
            'length_ratio': ratio, 'translation_length': hyp_len, 'reference_length': ref_len}


def codebleu_scores(references, predictions, language):
    references, predictions = list(references), list(predictions)
    if len(references) != len(predictions):
        raise ValueError('CodeBLEU: referências e previsões devem ter o mesmo tamanho.')
    keys = ('codebleu', 'ngram_match_score', 'weighted_ngram_match_score',
            'syntax_match_score', 'dataflow_match_score')
    totals = dict.fromkeys(keys, 0.0)
    pairs = [(r, p) for r, p in zip(references, predictions) if has_code(p)]
    if not pairs:
        return {**totals, 'codebleu_generated': 0.0, 'aggregation': 'mean_per_example_missing_zero'}
    try:
        from codebleu import calc_codebleu
    except ImportError as exc:
        raise RuntimeError('pacote codebleu ausente; instale o commit fixado do requirements.txt '
                           'junto com tree-sitter e tree-sitter-c') from exc
    cache = {}
    for reference, prediction in pairs:
        pair = (reference, prediction)
        if pair not in cache:
            cache[pair] = calc_codebleu(references=[[reference]], predictions=[prediction], lang=language)
        for key in keys:
            totals[key] += cache[pair][key]
    # Uma saída ausente vale zero. Não passa pelo fallback de dataflow do pacote,
    # que poderia atribuir 0.25 a uma correção vazia.
    return {**{key: value / len(references) for key, value in totals.items()},
            'codebleu_generated': totals['codebleu'] / len(pairs),
            'aggregation': 'mean_per_example_missing_zero'}


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


def compute_metrics(records, references, phases=PHASES, language='c'):
    """Geração: toda referência entra no denominador, com zero para saída ausente.
    `evaluated` inclui esses casos; `predicted`, `missing` e `coverage` mostram a cobertura real.
    As métricas durante uma fase incluem pendências como ausência de saída."""
    by_id = {str(r['id']): r for r in records}
    matched = [(by_id[i], ref) for i, ref in references.items() if i in by_id]
    unmatched = sorted(set(references) - set(by_id))
    report = {'language': language, 'reference_records': len(references),
              'matched_records': len(matched), 'unmatched_reference_ids': unmatched[:10]}

    def section(expected, evaluated, scores, predicted=None):
        predicted = evaluated if predicted is None else predicted
        return {'available': True, 'reason': None, 'expected': expected, 'evaluated': evaluated,
                'predicted': predicted, 'missing': expected - predicted,
                'coverage': round(predicted / expected, 4) if expected else None, **scores}

    def unavailable(reason, expected=0):
        return {'available': False, 'reason': reason, 'expected': expected, 'evaluated': 0,
                'missing': expected, 'coverage': None}

    def guarded(build):
        try:
            return build()
        except Exception as exc:  # métrica indisponível não derruba a execução
            return unavailable(f'{type(exc).__name__}: {exc}')

    wanted = {name for phase in phases for name in METRICS_FOR_PHASE[phase]}

    def detection():
        expected = [(r, ref['vulnerable']) for r, ref in matched if 'vulnerable' in ref]
        if not expected:
            return unavailable('a entrada não tem rótulo de vulnerabilidade (ex.: coluna vul ou label)')
        pairs = [(e, r['vulnerable']) for r, e in expected if type(r['vulnerable']) is bool]
        if not pairs:
            return unavailable('nenhuma previsão de Detection disponível', len(expected))
        return section(len(expected), len(pairs), detection_scores(pairs))

    def cwe_id():
        expected = [(r, ref['cwe_ids']) for r, ref in matched if ref.get('cwe_ids')]
        if not expected:
            return unavailable('a entrada não tem CWE esperada (ex.: coluna cwe_id)')
        pairs = [(set(ids), set(parse_cwe_ids(r.get('cwe_ids')))) for r, ids in expected]
        return section(len(expected), len(pairs), cwe_id_scores(pairs),
                       predicted=sum(bool(p) for _, p in pairs))

    def cwe_description():
        expected = [(r, ref['cwe_description']) for r, ref in matched if ref.get('cwe_description')]
        if not expected:
            return unavailable('a entrada não tem descrição esperada (ex.: coluna cwe_description)')
        pairs = [(text, description_text(r.get('cwe_output'))) for r, text in expected]
        bleu = corpus_bleu([p[0] for p in pairs], [p[1] for p in pairs])
        return section(len(expected), len(pairs), bleu, predicted=sum(bool(p) for _, p in pairs))

    def fix():
        expected = [(r, ref['fixed_code']) for r, ref in matched if ref.get('fixed_code')]
        if not expected:
            return unavailable('a entrada não tem correção esperada (ex.: coluna fixed_code)')
        pairs = [(text, r['fixed_code'] if has_code(r.get('fixed_code')) else '') for r, text in expected]
        try:
            scores = codebleu_scores([p[0] for p in pairs], [p[1] for p in pairs], language)
        except Exception as exc:  # pacote ausente ou linguagem sem parser
            return unavailable(f'{type(exc).__name__}: {exc}', len(expected))
        predicted = sum(bool(p) for _, p in pairs)
        copied = sum(has_code(r.get('fixed_code')) and r['fixed_code'].split() == r.get('code', '').split()
                     for r, _ in expected)
        scores.update(copied_input=copied, copied_input_rate=copied / predicted if predicted else 0.0)
        return section(len(expected), len(pairs), scores, predicted=predicted)

    builders = {'detection': detection, 'cwe_id': cwe_id, 'cwe_description': cwe_description, 'fix': fix}
    for name, build in builders.items():
        if name in wanted:
            report[name] = guarded(build)
    return _finite(report)


def format_metrics(report):
    """Linhas curtas para o terminal."""
    labels = [('detection', 'Detection  F1', lambda s: f"{s['f1']:.4f}"),
              ('cwe_id', 'CWE (ID)   F1', lambda s: f"macro {s['f1_macro']:.4f} | micro {s['f1_micro']:.4f}"),
              ('cwe_description', 'CWE (desc.) BLEU', lambda s: f"{s['bleu']:.4f}"),
              ('fix', 'Fix        CodeBLEU', lambda s: f"{s['codebleu']:.4f}")]
    lines = []
    for key, title, show in labels:
        section = report.get(key)
        if section is None:
            continue
        if section['available']:
            lines.append(f"  {title}: {show(section)}  (avaliados {section['evaluated']}/{section['expected']})")
        else:
            lines.append(f"  {title}: indisponível ({section['reason']})")
    return lines


def model_info(model):
    return {'model_path': getattr(model, 'path', None),
            'device': getattr(model, 'device', None), 'precision': getattr(model, 'precision', None),
            'context_token_limit': getattr(model, 'context_token_limit', None),
            'input_token_limit': getattr(model, 'input_token_limit', None),
            'max_new_tokens': getattr(model, 'max_new_tokens', None)}


def model_input(record, kind):
    if kind == 'detection':
        return record['code']
    # Checkpoints originais de Fix não foram condicionados à resposta da CWE.
    return cwe_prompt(record['code']) if kind == 'cwe' else fix_prompt(record['code'])


def predict(record, kind, model, threshold):
    """Uma etapa recebe o registro anterior e devolve apenas seus novos campos."""
    code = record['code']
    if kind == 'detection':
        probability = model.detect(code)
        if not valid_probability(probability):
            raise ValueError('Detection retornou uma probabilidade inválida.')
        vulnerable = probability >= threshold
        return {'vulnerability_probability': probability, 'vulnerable': vulnerable,
                'status': 'detected' if vulnerable else 'not_vulnerable',
                'skip_reason': None if vulnerable else SAFE_CODE_REASON}

    answer = model.generate(model_input(record, kind))
    if not isinstance(answer, str):
        raise ValueError(f'{kind.upper()} retornou uma resposta que não é texto.')
    if kind == 'cwe':
        legacy_cwe = getattr(model, 'legacy_cwe', False)
        cleaned, ids = clean_cwe_answer(answer, after_code=legacy_cwe)
        result = {'cwe_output': cleaned, 'cwe_ids': ids,
                'status': 'classified' if ids else 'cwe_unidentified', 'skip_reason': None}
        if legacy_cwe:
            result['cwe_raw_output'] = answer.strip()
        return result
    if not answer.strip():
        raise ValueError('Fix retornou uma resposta vazia.')
    return {'fixed_code': answer, 'status': 'generated_unvalidated', 'skip_reason': None}


def process_record(record, kind, model, threshold, load_error=None):
    """Registra falhas de inferência; erros de gravação ficam fora desta função."""
    started = perf_counter()
    history = record.setdefault('phase_results', {})
    history.pop(kind, None)
    try:
        if load_error is not None:
            raise RuntimeError(load_error)
        record.update(predict(record, kind, model, threshold))
        if record.get('error'):
            record['status'] = 'error'
        elif record.get('fixed_code'):
            record['status'] = 'generated_unvalidated'
    except Exception as exc:
        record.update(status='error', error={'stage': kind, 'message': str(exc)}, skip_reason=None)
    status, reason = phase_outcome(record, kind)
    history[kind] = {'status': status, 'reason': reason,
                     'runtime_seconds': round(perf_counter() - started, 4), **model_info(model)}


def run_pipeline(records, factory, threshold=0.5, checkpoint=None, phase_finished=None,
                  stop_after='fix', phase_metrics=None, record_finished=None, checkpoint_throttled=False,
                  batch_size=BATCH_SIZE):
    """Carrega um modelo por vez e encaminha somente registros elegíveis."""
    if stop_after not in PHASES:
        raise ValueError('stop_after inválido.')
    if not valid_probability(threshold):
        raise ValueError('threshold deve estar entre 0 e 1.')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch-size deve ser inteiro positivo.')
    for kind in PHASES[:PHASES.index(stop_after) + 1]:
        started = perf_counter()
        eligible = []
        for record in records:
            status, reason = phase_outcome(record, kind)
            if status == 'pending' and (kind == 'detection' or record['vulnerable'] is True):
                eligible.append(record)
            else:
                record.setdefault('phase_results', {}).setdefault(kind, {'status': status, 'reason': reason})
        print(f'{kind}: {len(eligible)} registro(s) para executar', file=sys.stderr)
        model, load_error = None, None
        try:
            if eligible:
                try:
                    model = factory(kind)
                    if kind != 'detection' and hasattr(model, 'prefetch'):
                        # Aproxima comprimentos sem tokenizar duas vezes; reduz padding nos lotes.
                        eligible.sort(key=lambda r: len(r['code']))
                except Exception as exc:
                    load_error = str(exc)
            for index, record in enumerate(eligible, 1):
                if (index - 1) % batch_size == 0 and hasattr(model, 'prefetch'):
                    chunk = eligible[index - 1:index - 1 + batch_size]
                    try:
                        model.prefetch([model_input(r, kind) for r in chunk])
                    except Exception as exc:  # o caminho individual refaz e registra o erro de cada registro
                        print(f'{kind}: lote falhou ({exc}); seguindo um por vez.', file=sys.stderr)
                process_record(record, kind, model, threshold, load_error)
                print(f"{kind} [{index}/{len(eligible)}] {record['id']}: {record['status']}", file=sys.stderr)
                if checkpoint:
                    checkpoint(records, force=False) if checkpoint_throttled else checkpoint(records)
                if record_finished:
                    record_finished(kind, record, index, len(eligible), records)
        finally:
            if model is not None:
                model.close()
        if phase_metrics is not None:
            phase_metrics[kind] = {'runtime_seconds': round(perf_counter() - started, 4),
                                   'executed_this_run': len(eligible), **model_info(model),
                                   'counts': dict(Counter(phase_outcome(r, kind)[0] for r in records))}
        if checkpoint:
            checkpoint(records)
        if phase_finished:
            phase_finished(kind, records)
    return records


def short(text, limit=160):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def debug_record_line(kind, record, threshold):
    """Resumo de um registro recém-processado, para o modo --debug."""
    phase = record.get('phase_results', {}).get(kind, {})
    head = f"[debug] {kind} {record['id']}: {phase.get('status')} em {phase.get('runtime_seconds')}s"
    if phase.get('status') == 'error':
        return f"{head} | erro: {short(phase.get('reason'))}"
    if kind == 'detection':
        verdict = 'vulnerável' if record['vulnerable'] else 'seguro'
        return f"{head} | p(vulnerável)={record['vulnerability_probability']:.4f} (limiar {threshold}) -> {verdict}"
    if kind == 'cwe':
        return f"{head} | ids={record['cwe_ids'] or '-'} | resposta: {short(record['cwe_output'])}"
    return f"{head} | correção: {len(record['fixed_code'] or '')} caractere(s)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', help='URL de dataset do Hugging Face, ID de um dataset conhecido '
                                       'ou arquivo local (Parquet, JSON, CSV ou código).')
    source.add_argument('--resume-from', type=Path, help='JSON da pipeline para continuar etapas pendentes.')
    parser.add_argument('--retry-errors', action='store_true',
                        help='Com --resume-from, executa de novo os registros que terminaram em erro.')
    parser.add_argument('--stop-after', choices=PHASES, default='fix', help='Última etapa a executar.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/pipeline_results.json')
    parser.add_argument('--code-column', help='Coluna do código de entrada.')
    parser.add_argument('--fixed-code-column', help='Coluna com o código corrigido esperado.')
    parser.add_argument('--cwe-id-column', help='Coluna com os IDs CWE esperados.')
    parser.add_argument('--cwe-description-column', help='Coluna com a descrição CWE esperada.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto',
                        help='auto: geração em bf16/fp16 na GPU e fp32 em CPU; Detection em fp32.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Registros por lote (padrão: %(default)s).')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--references', type=Path,
                        help='Parquet, JSON ou CSV com os rótulos esperados (padrão: o próprio --input).')
    parser.add_argument('--metrics-language', default='c', help='Linguagem do CodeBLEU (padrão: c).')
    parser.add_argument('--no-metrics', action='store_true', help='Não calcula métricas.')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Recalcula métricas de --resume-from com --references, sem carregar modelos.')
    parser.add_argument('--debug', '--yolo', dest='debug', action='store_true',
                        help='Mostra, durante a execução, o detalhe de cada registro e as métricas parciais.')
    parser.add_argument('--metrics-every', type=int, default=10, metavar='N',
                        help='Com --debug, imprime as métricas parciais a cada N registros (padrão: 10).')
    for kind, path in DEFAULT_MODELS.items():
        parser.add_argument(f'--{kind}-model', type=Path, default=path)
        parser.add_argument(f'--{kind}-max-input-tokens', type=int, default=DEFAULT_INPUT_TOKEN_LIMITS[kind],
                            help=f'Limite de tokens de entrada de {kind} (padrão: %(default)s).')
    args = parser.parse_args(argv)
    if args.metrics_only:
        if not args.resume_from or not args.references or args.no_metrics:
            parser.error('--metrics-only exige --resume-from e --references e não aceita --no-metrics.')
        if args.output == ROOT / 'results/pipeline_results.json':
            args.output = args.resume_from.with_name(f'{args.resume_from.stem}_metrics.json')
    if args.input is None and args.resume_from is None:
        args.input = DEFAULT_INPUT
    if not 0 <= args.threshold <= 1 or min(args.max_new_tokens, args.metrics_every, args.batch_size) < 1:
        parser.error('threshold deve estar entre 0 e 1; tokens, metrics-every e batch-size devem ser positivos.')
    input_token_limits = {kind: getattr(args, f'{kind}_max_input_tokens') for kind in PHASES}
    if min(input_token_limits.values()) < 1:
        parser.error('Os limites de tokens de entrada devem ser inteiros positivos.')
    repo_id = hub_dataset_id(args.input) if args.input is not None else None
    input_path = None if repo_id else Path(args.input or args.resume_from)
    if input_path is not None and (input_path.resolve() == args.output.resolve()
            or (input_path.exists() and args.output.exists() and os.path.samefile(input_path, args.output))):
        parser.error('Entrada e saída precisam ser arquivos diferentes.')
    reference_columns = {field: name for field, name in (
        ('fixed_code', args.fixed_code_column), ('cwe_ids', args.cwe_id_column),
        ('cwe_description', args.cwe_description_column),
    ) if name is not None}
    try:
        if repo_id:
            dataset = load_hub_dataset(repo_id)
            default_code_column = ('func' if repo_id in CWE_LABEL_DATASETS else
                                   'vulnerable_code' if 'vulnerable_code' in dataset.column_names else 'code')
            code_column = args.code_column or default_code_column
            if code_column not in dataset.column_names:
                raise ValueError(f'Coluna {code_column!r} ausente no dataset. Colunas: {dataset.column_names}')
            columns = {'id', code_column}
            if not args.no_metrics and args.references is None:
                for field, aliases in REFERENCE_ALIASES.items():
                    columns.update((reference_columns[field],) if field in reference_columns else aliases)
            dataset = dataset.select_columns([name for name in dataset.column_names if name in columns])
            rows = [row for row in dataset if has_code(row[code_column])]
            records = records_from_rows(rows, code_column)
        elif args.resume_from:
            records = read_checkpoint(args.resume_from)
        else:
            records = read_records(input_path, args.code_column or 'code')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.retry_errors:
        if not args.resume_from:
            parser.error('--retry-errors exige --resume-from.')
        print(f'{reset_errors(records)} registro(s) com erro serão executados de novo.', file=sys.stderr)
    references = {}
    if not args.no_metrics:
        reference_path = args.references or args.input
        if reference_path is None:
            print('Métricas: retomada sem --references; informe o arquivo com os rótulos esperados '
                  'para calculá-las.', file=sys.stderr)
        else:
            try:
                if repo_id and args.references is None:
                    references = references_from_rows(rows, reference_columns)
                else:
                    references = read_references(reference_path, columns=reference_columns)
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
            if not references:
                print('Métricas: nenhum rótulo esperado encontrado (colunas aceitas: '
                      + ', '.join(n for names in REFERENCE_ALIASES.values() for n in names)
                      + '). Nada será medido.', file=sys.stderr)
    if args.metrics_only:
        report = compute_metrics(records, references, PHASES[:PHASES.index(args.stop_after) + 1], args.metrics_language)
        write_json(args.output, report)
        print(f'Métricas recalculadas sem inferência: {args.output}')
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phases_dir = Path(tempfile.mkdtemp(prefix=f'{args.output.stem}_phases_', dir=args.output.parent))
    # Cada execução preserva também sua própria cópia consolidada.
    run_output = phases_dir / 'pipeline_results.json'
    print(f'Resultados por fase: {phases_dir}', file=sys.stderr)
    started = perf_counter()
    metrics = {}
    summary = {
        'execution_state': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'input': str(args.input) if repo_id else str(input_path.resolve()), 'resumed': bool(args.resume_from),
        'output': str(args.output.resolve()), 'stop_after': args.stop_after,
        'run_output': str(run_output.resolve()),
        'device_requested': args.device, 'threshold': args.threshold,
        'max_new_tokens': args.max_new_tokens,
        'batch_size': args.batch_size, 'precision_requested': args.precision,
        'input_token_limits': input_token_limits,
        'models_requested': {kind: str(getattr(args, f'{kind}_model').resolve()) for kind in PHASES},
        'phases': metrics,
    }
    metrics_report = {}
    if references:
        summary['metrics'] = metrics_report
        summary['metrics_language'] = args.metrics_language

    last_save = [0.0]

    def save(rows, force=True):
        # Por registro, no máximo a cada 5s: regravar tudo a cada item é O(n²) em entradas grandes.
        if not force and perf_counter() - last_save[0] < 5:
            return
        write_json(args.output, rows)
        write_json(run_output, rows)
        last_save[0] = perf_counter()

    def save_summary():
        summary.update(total_records=len(records),
                       status_counts=dict(Counter(r['status'] for r in records)),
                       runtime_seconds=round(perf_counter() - started, 4))
        write_json(phases_dir / 'summary.json', summary)

    def phase_finished(kind, rows):
        write_phase_results(phases_dir, kind, rows)
        if references:
            done = PHASES[:PHASES.index(kind) + 1]
            metrics_report.clear()
            metrics_report.update(compute_metrics(rows, references, done, args.metrics_language))
            write_json(phases_dir / 'metrics.json', metrics_report)
            print('Métricas até esta fase:', file=sys.stderr)
            for line in format_metrics(metrics_report):
                print(line, file=sys.stderr)
        save_summary()

    def load_model(kind):
        loading = perf_counter()
        model = LocalModel(getattr(args, f'{kind}_model'), kind, args.device, args.max_new_tokens,
                            input_token_limits[kind], **({'precision': args.precision} if args.precision != 'auto' else {}))
        if args.debug:
            print(f'[debug] {kind}: modelo {getattr(model, "path", "?")} em {getattr(model, "device", "?")}, '
                  f'carregado em {perf_counter() - loading:.1f}s', file=sys.stderr)
        return model

    def record_finished(kind, record, index, total, rows):
        print(debug_record_line(kind, record, args.threshold), file=sys.stderr)
        # A última do lote sai em phase_finished, com o resumo da fase.
        if references and index % args.metrics_every == 0 and index < total:
            partial = compute_metrics(rows, references, (kind,), args.metrics_language)
            print(f'[debug] métricas parciais de {kind} ({index}/{total} registros):', file=sys.stderr)
            for line in format_metrics(partial):
                print(line, file=sys.stderr)

    save(records)
    save_summary()
    try:
        run_pipeline(records, load_model, args.threshold, save, phase_finished,
                     stop_after=args.stop_after, phase_metrics=metrics, checkpoint_throttled=True,
                      record_finished=record_finished if args.debug else None, batch_size=args.batch_size)
        summary['execution_state'] = 'completed_with_errors' if any(r['status'] == 'error' for r in records) else 'completed'
    except KeyboardInterrupt:
        summary['execution_state'] = 'interrupted'
        print(f'Execução interrompida. Retome com --resume-from {run_output}', file=sys.stderr)
        return 130
    except Exception as exc:
        summary['execution_state'] = 'failed'
        summary['execution_error'] = f'{type(exc).__name__}: {exc}'
        print(f'Falha na execução: {summary["execution_error"]}', file=sys.stderr)
        return 1
    finally:
        try:
            save(records)
        except OSError as exc:
            print(f'Aviso: não foi possível gravar o checkpoint final: {exc}', file=sys.stderr)
        summary['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_summary()
    print(f'Resultado: {args.output}')
    if references:
        print(f"Métricas: {phases_dir / 'metrics.json'}")
    errors = [r for r in records if r['status'] == 'error']
    for record in errors:
        print(f"{record['id']} [{record['error']['stage']}]: {record['error']['message']}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
