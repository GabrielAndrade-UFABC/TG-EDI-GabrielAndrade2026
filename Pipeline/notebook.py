import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TCC: Detection, CWE e Fix")


@app.cell
def _():
    import json
    import sys
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))
    import pipeline

    return ROOT, datetime, json, mo, pd, pipeline


@app.cell
def _(mo):
    mo.md("""
    # Detection, CWE e Fix

    Versão local do notebook do Colab. Roda a pipeline sobre um conjunto **com rótulos** e mede, de ponta a ponta:

    | Etapa | Métrica |
    | --- | --- |
    | Detection | F1 (classe positiva = vulnerável) |
    | CWE, identificador | F1 multirrótulo (macro e micro) |
    | CWE, descrição | BLEU-4 de corpus |
    | Fix | CodeBLEU (`lang="c"`) |

    Quando Detection sinaliza uma vulnerabilidade, CWE e Fix são executados de forma independente.
    Ajuste a configuração, confira a amostra e clique em **Rodar pipeline**.
    """)
    return


@app.cell
def _(mo):
    import torch

    _gpu = (f"**GPU:** {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)"
            if torch.cuda.is_available() else "**Sem GPU**: a pipeline vai rodar na CPU (lento).")
    mo.md(f"## 1. Ambiente\n\nPyTorch {torch.__version__} · {_gpu}")
    return


@app.cell
def _(ROOT, mo, pipeline):
    def _checkpoints(kind):
        # checkpoints/<fase>/<modelo> e variações como checkpoints/bigvul/<fase>/<modelo>; smoke fica de fora.
        found = sorted(p.parent for p in (ROOT / "checkpoints").glob(f"**/{kind}/*/model.safetensors")
                       if "smoke" not in p.parts)
        found += [p for p in [pipeline.DEFAULT_MODELS[kind]] if p not in found]
        return {str(p.relative_to(ROOT / "checkpoints")) if p.is_relative_to(ROOT / "checkpoints") else p.name: str(p)
                for p in found}

    _local = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "datasets").glob("*")
                    if p.suffix.lower() in (".csv", ".json", ".parquet"))
    _example = "examples/pipeline_input.json"
    _sources = [*_local, _example, *pipeline.PARQUET_DATASETS, *pipeline.CWE_LABEL_DATASETS]
    dataset = mo.ui.dropdown(
        options=_sources,
        value="datasets/dataset_completo.csv" if "datasets/dataset_completo.csv" in _local else next(iter(_local), _example),
        label="Conjunto",
    )
    dataset_url = mo.ui.text(label="Link do dataset no Hugging Face",
                             placeholder="https://huggingface.co/datasets/autor/dataset")
    sample_size = mo.ui.number(start=2, stop=5000, step=2, value=100, label="Tamanho da amostra")
    balanced = mo.ui.checkbox(value=True, label="Metade vulnerável, metade segura")
    full_file = mo.ui.checkbox(value=False, label="Conjunto inteiro (ignora amostra e tamanho máx.)")
    max_chars = mo.ui.number(start=200, stop=20000, step=100, value=1500, label="Tamanho máx. do código (caracteres)")
    seed = mo.ui.number(start=0, stop=10**6, value=42, label="Seed")
    input_token_limits = {
        kind: mo.ui.number(start=1, step=1, value=pipeline.DEFAULT_INPUT_TOKEN_LIMITS[kind],
                           label=f"{kind.title()} · tokens de entrada")
        for kind in pipeline.PHASES
    }
    max_new_tokens = mo.ui.number(start=16, stop=1024, step=16, value=512, label="CWE e Fix · tokens gerados")
    batch_size = mo.ui.number(start=1, stop=128, step=1, value=pipeline.BATCH_SIZE, label="Registros por lote")
    threshold = mo.ui.slider(0, 1, step=0.05, value=0.5, label="Limiar da Detection", show_value=True)
    models = {kind: mo.ui.dropdown(_checkpoints(kind), value=next(k for k, v in _checkpoints(kind).items() if v == str(pipeline.DEFAULT_MODELS[kind])), label=f"Modelo {kind}")
              for kind in pipeline.PHASES}

    mo.vstack([
        mo.md("## 2. Configuração"),
        mo.hstack([dataset, dataset_url], justify="start"),
        mo.md("Se preenchido, o link tem prioridade sobre o conjunto selecionado."),
        mo.hstack([sample_size, balanced, max_chars, seed], justify="start"),
        full_file,
        mo.hstack(list(input_token_limits.values()), justify="start"),
        mo.hstack([max_new_tokens, batch_size, threshold], justify="start"),
        mo.hstack(list(models.values()), justify="start"),
    ])
    return (
        balanced,
        batch_size,
        dataset,
        dataset_url,
        full_file,
        input_token_limits,
        max_chars,
        max_new_tokens,
        models,
        sample_size,
        seed,
        threshold,
    )


@app.cell
def _(ROOT, dataset, dataset_url, mo, pd, pipeline):
    _url = dataset_url.value.strip()
    _repo_id = pipeline.hub_dataset_id(_url or dataset.value)
    mo.stop(bool(_url and not _repo_id), mo.callout("Informe um link de dataset do Hugging Face.", kind="danger"))
    if _repo_id:
        source_table = pipeline.load_hub_dataset(_repo_id)
        if _repo_id in pipeline.PARQUET_DATASETS:
            mo.output.append(mo.md(f"Teste de `{_repo_id}` com {len(source_table)} registros (cerca de 10% do conjunto original)."))
    else:
        _path = ROOT / dataset.value
        if _path.suffix.lower() == ".csv":
            source_table = pd.read_csv(_path, keep_default_na=False)
        else:
            _rows = pipeline.read_input_rows(_path)
            source_table = _rows if _path.suffix.lower() == ".parquet" else pd.DataFrame(_rows)
    source_columns = list(source_table.column_names if hasattr(source_table, "column_names") else source_table.columns)
    return source_columns, source_table


@app.cell
def _(mo, pipeline, source_columns):
    def _column(label, candidates, required=False):
        options = {name: name for name in source_columns}
        if not required:
            options = {"Sem coluna": "", **options}
        default = next((name for name in candidates if name in source_columns),
                       (source_columns[0] if source_columns else None) if required else "Sem coluna")
        return mo.ui.dropdown(options, value=default, label=label)

    code_column = _column("Função vulnerável", ("vulnerable_code", "func_before", "code", "func"), required=True)
    fixed_column = _column("Função corrigida", pipeline.REFERENCE_ALIASES["fixed_code"])
    cwe_id_column = _column("CWE ID", pipeline.REFERENCE_ALIASES["cwe_ids"])
    cwe_desc_column = _column("CWE description", pipeline.REFERENCE_ALIASES["cwe_description"])
    mo.hstack([code_column, fixed_column, cwe_id_column, cwe_desc_column], justify="start")
    return code_column, cwe_desc_column, cwe_id_column, fixed_column


@app.cell
def _(
    balanced,
    code_column,
    cwe_desc_column,
    cwe_id_column,
    fixed_column,
    full_file,
    max_chars,
    mo,
    pd,
    pipeline,
    sample_size,
    seed,
    source_columns,
    source_table,
):
    _code = code_column.value
    mo.stop(_code not in source_columns, mo.callout("Selecione a coluna com o código de entrada.", kind="danger"))
    _label_columns = (*pipeline.REFERENCE_ALIASES["vulnerable"], "target")
    _columns = {"id", _code, fixed_column.value, cwe_id_column.value, cwe_desc_column.value, *_label_columns}
    _columns = [name for name in source_columns if name in _columns]
    table = (source_table.select_columns(_columns).to_pandas()
             if hasattr(source_table, "select_columns") else source_table[_columns])
    _rows = table[table[_code].map(pipeline.has_code)]
    # Este filtro usa caracteres. A execução aplica o limite de tokens de cada etapa.
    if not full_file.value:
        _rows = _rows[_rows[_code].str.len() <= max_chars.value]
    mo.stop(_rows.empty, mo.callout("Nenhum código disponível com os filtros atuais.", kind="warn"))
    if "id" not in _rows.columns:
        _rows = _rows.assign(id=[f"r{i}" for i in _rows.index])
    _label = next((c for c in _label_columns if c in _rows.columns), None)
    if full_file.value:
        sample = _rows
    elif balanced.value and _label:
        _half = sample_size.value // 2
        _groups = [g.sample(n=min(_half, len(g)), random_state=seed.value) for _, g in _rows.groupby(_rows[_label].astype(int))]
        sample = pd.concat(_groups).sample(frac=1, random_state=seed.value)
    else:
        sample = _rows.sample(n=min(sample_size.value, len(_rows)), random_state=seed.value)
    _counts = sample[_label].value_counts().to_dict() if _label else "sem coluna de rótulo"
    mo.vstack([
        mo.md(f"## 3. Amostra\n\n{len(sample)} de {len(table)} registros "
              f"({'conjunto inteiro' if full_file.value else f'código ≤ {max_chars.value} caracteres'}). "
              f"Rótulos (`{_label}`): {_counts}"),
        mo.ui.table(sample, page_size=5, selection=None),
    ])
    return (sample,)


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Rodar pipeline", kind="success")
    mo.vstack([mo.md("## 4. Execução\n\nO andamento aparece no console desta célula."), run])
    return (run,)


@app.cell
def _(
    ROOT,
    batch_size,
    code_column,
    cwe_desc_column,
    cwe_id_column,
    datetime,
    fixed_column,
    input_token_limits,
    max_new_tokens,
    mo,
    models,
    pipeline,
    run,
    sample,
    threshold,
):
    mo.stop(not run.value, mo.md("_Clique em **Rodar pipeline**._"))
    _limits = {kind: field.value for kind, field in input_token_limits.items()}
    mo.stop(batch_size.value is None or batch_size.value < 1 or int(batch_size.value) != batch_size.value,
            mo.callout("Preencha o tamanho do lote com um número inteiro positivo.", kind="danger"))
    mo.stop(any(value is None or value < 1 or int(value) != value for value in _limits.values()),
            mo.callout("Preencha os limites de entrada com números inteiros positivos.", kind="danger"))
    run_dir = ROOT / "results" / "notebook" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    _sample_path = run_dir / "amostra.parquet"
    sample.to_parquet(_sample_path, index=False)
    _argv = ["--input", str(_sample_path), "--output", str(run_dir / "pipeline_results.json"),
             "--code-column", code_column.value, "--fixed-code-column", fixed_column.value,
             "--cwe-id-column", cwe_id_column.value, "--cwe-description-column", cwe_desc_column.value,
             "--max-new-tokens", str(max_new_tokens.value),
             "--batch-size", str(int(batch_size.value)),
             "--threshold", str(threshold.value)]
    for _kind, _model in models.items():
        _argv += [f"--{_kind}-model", _model.value, f"--{_kind}-max-input-tokens", str(int(_limits[_kind]))]
    try:
        _code = pipeline.main(_argv)
    except SystemExit as _exc:
        _code = _exc.code
    _phase_dirs = list(run_dir.glob("pipeline_results_phases_*"))
    mo.stop(not _phase_dirs, mo.callout(f"A execução terminou antes de gerar resultados (código {_code}).", kind="danger"))
    phases_dir = max(_phase_dirs, key=lambda p: p.stat().st_mtime)
    mo.md(f"Concluído (código {_code}). Resultados em `{run_dir}`.")
    return (phases_dir,)


@app.cell
def _(json, mo, pd, phases_dir):
    _path = phases_dir / "metrics.json"
    mo.stop(not _path.exists(), mo.callout("Sem métricas: a entrada não tem rótulos.", kind="warn"))
    report = json.loads(_path.read_text())

    def _row(name, key, field):
        section = report.get(key) or {}
        if not section.get("available"):
            return {"métrica": name, "valor": None, "avaliados": "-", "observação": section.get("reason")}
        note = "" if section.get("coverage") == 1 else f"{section.get('missing')} sem previsão"
        return {"métrica": name, "valor": section.get(field),
                "avaliados": f"{section.get('evaluated')}/{section.get('expected')}", "observação": note}

    _summary = json.loads((phases_dir / "summary.json").read_text())
    _times = {k: v["runtime_seconds"] for k, v in _summary["phases"].items()}
    mo.vstack([
        mo.md(f"## 5. Métricas\n\nTempo total: {_summary['runtime_seconds']:.1f}s · por fase: {_times}"),
        mo.ui.table(pd.DataFrame([
            _row("Detection - F1", "detection", "f1"),
            _row("Detection - precisão", "detection", "precision"),
            _row("Detection - recall", "detection", "recall"),
            _row("CWE (ID) - F1 macro", "cwe_id", "f1_macro"),
            _row("CWE (ID) - F1 micro", "cwe_id", "f1_micro"),
            _row("CWE (descrição) - BLEU", "cwe_description", "bleu"),
            _row("Fix - CodeBLEU", "fix", "codebleu"),
        ]), selection=None),
    ])
    return


@app.cell
def _(mo, pd, phases_dir):
    _tabs = {name: mo.ui.table(pd.read_csv(phases_dir / f"{file}.csv", encoding="utf-8-sig", keep_default_na=False),
                               page_size=10, selection=None)
             for name, file in (("Detection", "01_detection"), ("CWE", "02_cwe"), ("Fix", "03_fix"))}
    mo.vstack([mo.md("## 6. Resultados por fase"), mo.ui.tabs(_tabs)])
    return


@app.cell
def _(json, mo, phases_dir):
    records = {str(r["id"]): r for r in json.loads((phases_dir / "pipeline_results.json").read_text(encoding="utf-8"))}
    record_id = mo.ui.dropdown(list(records), value=next(iter(records)), label="Registro")
    mo.vstack([mo.md("## 7. Inspecionar um registro"), record_id])
    return record_id, records


@app.cell
def _(code_column, cwe_desc_column, cwe_id_column, fixed_column, mo, pipeline, record_id, records, sample):
    _r = records[record_id.value]
    _expected = sample[sample["id"].astype(str) == record_id.value].iloc[0]
    _fixed_ref = _expected.get(fixed_column.value)
    _cwe_ref = pipeline.parse_cwe_ids(_expected.get(cwe_id_column.value))
    _cwe_desc = pipeline.description_text(_expected.get(cwe_desc_column.value))
    _label = next((_expected[c] for c in (*pipeline.REFERENCE_ALIASES["vulnerable"], "target") if c in _expected), "?")
    _prob = _r.get("vulnerability_probability")
    mo.vstack([
        mo.md(f"**Status:** `{_r['status']}` · **p(vulnerável):** {_prob if _prob is None else round(_prob, 4)} "
              f"· **rótulo:** {_label}\n\n**CWE prevista:** {_r.get('cwe_output') or '-'} "
              f"· **CWE esperada:** {', '.join(_cwe_ref) or '-'}\n\n**Descrição esperada:** {_cwe_desc or '-'}"),
        mo.hstack([
            mo.vstack([mo.md("**Código**"), mo.ui.code_editor(str(_expected[code_column.value]), language="c", disabled=True)]),
            mo.vstack([mo.md("**Fix gerado**"), mo.ui.code_editor(_r.get("fixed_code") or "", language="c", disabled=True)]),
            mo.vstack([mo.md("**Fix esperado**"), mo.ui.code_editor(_fixed_ref if pipeline.has_code(_fixed_ref) else "", language="c", disabled=True)]),
        ], widths="equal"),
    ])
    return


@app.cell
def _(ROOT, mo, pd):
    import ast

    _path = ROOT / "checkpoints" / "results.csv"
    mo.stop(not _path.exists(), mo.md("_Sem `checkpoints/results.csv`: rode o `train.py`._"))
    _rows = []
    for _r in pd.read_csv(_path).fillna("").to_dict("records"):
        _m = ast.literal_eval(_r["metrics"]) if _r["metrics"] else {}
        _rows.append({"fase": _r["phase"], "modelo": _r["model"], "status": _r["status"],
                      "minutos": round(_r["runtime_seconds"] / 60, 1),
                      **{k: _m.get(k) for k in ("f1", "f1_macro", "f1_micro", "bleu", "codebleu", "failed")}})
    mo.vstack([mo.md("## 8. Treinos (`checkpoints/results.csv`)\n\nMétricas no teste de cada fase (200 exemplos)."),
               mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return


if __name__ == "__main__":
    app.run()
