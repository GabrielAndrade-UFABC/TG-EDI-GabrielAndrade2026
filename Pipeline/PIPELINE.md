# Detection, CWE e Fix

Execute diretamente com Python, com as dependências já disponíveis no seu ambiente:

```console
python main.py
```

Os padrões de inferência são 512 tokens de entrada por etapa, até 512 tokens gerados e 16 registros por lote. O notebook permite ajustar esses valores. Na GPU, a geração usa bf16 quando suportado, ou fp16; em CPU usa fp32. `--precision float32` permite uma comparação controlada. Os lotes generativos agrupam códigos por tamanho e reutilizam previsões de entradas repetidas dentro do lote.

Esses valores são limites solicitados, sujeitos à janela total do modelo. CodeBERTa e GraphCodeBERT têm 512 posições úteis para entrada e saída juntas. Com os padrões 512/512, a pipeline ajusta automaticamente para 256 de entrada e até 256 de saída, imprime o ajuste e grava os valores efetivos por fase em `summary.json`. Se a entrada for limitada a 128, a saída pode usar até 384. Modelos com janela suficiente mantêm os limites solicitados.

Os checkpoints antigos aprenderam o token de padding como encerramento. A inferência reconhece esse encerramento depois de produzir conteúdo e para também no próximo bloco do prompt. Respostas vazias de Fix recebem uma segunda tentativa. Se ela também falhar, o registro mantém erro explícito: não é preenchido com o código original. Cópias efetivamente produzidas pelo modelo são preservadas.

Em DiverseVul e PrimeVul, o rótulo oficial é `target`. A presença de CWE não é usada para criar um rótulo de vulnerabilidade. Nas referências locais, `target` tem prioridade sobre o antigo campo `vul`.

Para recalcular métricas de uma execução preservada, sem carregar modelos nem treinar novamente:

```console
python pipeline.py --metrics-only --resume-from results/notebook/EXECUCAO/pipeline_results.json --references results/notebook/EXECUCAO/amostra.parquet --output results/metricas_recalculadas.json
```

Isso corrige a avaliação das previsões existentes. Para substituir os escores antigos calculados por `Trainer.predict`, é preciso rodar a geração real dos mesmos checkpoints. A inferência não recupera aprendizado que tenha sido perdido no treinamento.

Todas as referências elegíveis entram nas métricas generativas, inclusive registros pulados, com erro ou sem resposta. `evaluated` é o total pontuado, `predicted` conta as saídas disponíveis e `missing`/`coverage` informam as ausências. CWE ID usa conjuntos vazios para ausências; BLEU inclui hipóteses vazias. CodeBLEU é a média por exemplo, com zero explícito para saída ausente, evitando a pontuação de dataflow que o pacote poderia atribuir a texto vazio. `codebleu_generated` mostra separadamente a média apenas das saídas não vazias. A agregação é registrada como `mean_per_example_missing_zero` e não é diretamente comparável ao antigo CodeBLEU de corpus.

Modo debug (`--debug`, ou o alias `--yolo`): além do resumo final de cada etapa, mostra durante a execução uma linha por registro (probabilidade, CWE, tamanho da correção, tempo) e as métricas parciais a cada `--metrics-every N` registros (padrão 10). As métricas precisam de rótulos na entrada ou em `--references`.

```console
python main.py --debug --input dataset.csv --code-column func
```

Para usar um dataset do Hugging Face:

```console
python pipeline.py --input https://huggingface.co/datasets/hitoshura25/crossvul
```

CVEfixes, MegaVul e CrossVul são lidos em Parquet. A pipeline separa 20% do `train`, depois usa metade desse subconjunto como teste. As duas divisões usam seed 42.

Para um arquivo de código:

```console
python TCC/pipeline.py --input caminho/funcao.c --device cpu
```

A pipeline usa os checkpoints locais em `checkpoints/<fase>/<modelo>`, gerados por `python train.py` (não acompanham o código, pois ocupam cerca de 14 GB). `--device auto` seleciona CUDA quando disponível e CPU em caso contrário. Não treina novamente.

A saída padrão é `TCC/results/pipeline_results.json`. Ao terminar cada fase, gera CSV, JSON e uma tabela HTML numa pasta `pipeline_results_phases_*` ao lado do resultado final. `summary.json` contém contagens, tempos e parâmetros. `phase_results` mantém o histórico por registro. Dentro da pasta da execução também há uma cópia própria de `pipeline_results.json`, preservada quando você roda novamente.

| Etapa | Planilha | Tabela para abrir no navegador |
| --- | --- | --- |
| Detection | `01_detection.csv` | `01_detection.html` |
| CWE | `02_cwe.csv` | `02_cwe.html` |
| Fix | `03_fix.csv` | `03_fix.html` |

Cada tabela contém todos os registros, incluindo os pulados, com código, previsão, motivo, tempo, dispositivo e modelo quando disponíveis. O terminal mostra o caminho assim que a fase termina. `phase_status` distingue `success`, `unidentified`, `error`, `skipped` e `pending`. Sucesso indica conclusão da etapa, não validação da previsão. Os CSVs usam UTF-8 com BOM e separador vírgula; respostas com aparência de fórmula são exportadas como texto. JSON e HTML preservam o conteúdo original.

JSON aceita um objeto ou lista com `code` e `id` opcional. CSV e Parquet aceitam as mesmas colunas. Use `--code-column` para escolher a coluna de entrada e `--fixed-code-column`, `--cwe-id-column` e `--cwe-description-column` para mapear as referências. Quando Detection sinaliza uma vulnerabilidade, CWE e Fix recebem o código original e executam de forma independente.

Para parar ou retomar:

```console
python TCC/pipeline.py --input TCC/examples/pipeline_input.json --stop-after detection --output TCC/results/detection.json
python TCC/pipeline.py --resume-from TCC/results/detection.json --output TCC/results/final.json
```

A retomada executa etapas pendentes. Com `--retry-errors`, também tenta novamente as etapas que falharam e preserva as previsões das demais. `phase_results` registra separadamente os resultados de CWE e Fix. `generated_unvalidated` indica geração sem validação de compilação ou testes. As saídas dos modelos ficam nos arquivos de resultados.

Os checkpoints atuais ainda precisam de revisão de qualidade: o fluxo pode concluir mesmo quando a classificação ou o texto gerado estão incorretos.

O wrapper carrega somente modelos locais em formato safetensors, sem executar código remoto de modelos. Não executa o código de entrada nem o código gerado. A retomada valida tipos, probabilidades, IDs e consistência do histórico. Arquivos temporários exclusivos evitam colisões durante a gravação. HTML escapa o conteúdo inserido; o CSV neutraliza fórmulas. Isso reduz riscos na exportação, mas não representa auditoria das dependências ou garantia sobre o código gerado.

`summary.json` informa `execution_state`: `running`, `completed`, `completed_with_errors`, `interrupted` ou `failed`. Uma interrupção pelo teclado retorna 130 e permite retomar do JSON salvo. Falhas ao gravar arquivos não são confundidas com erros de inferência.

Para verificar o wrapper sem carregar os modelos:

```console
python -m unittest discover -s TCC/tests -v
```

Os testes usam modelos simulados; não medem a qualidade nem substituem a execução dos checkpoints reais.
