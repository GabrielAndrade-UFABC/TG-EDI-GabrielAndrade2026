# Identificação e correção de vulnerabilidades em código-fonte com Transformers

Projeto do Trabalho de Graduação em Engenharia da Informação da UFABC, de Gabriel Andrade de Vasconcelos, sob orientação do professor Luneque Silva Jr. O código avalia modelos Transformer em funções C para detectar indícios de vulnerabilidade, identificar a classe de fraqueza segundo a CWE (*Common Weakness Enumeration*) e propor uma correção.

## Estrutura do projeto

| Diretório | Finalidade | Ponto de entrada |
| --- | --- | --- |
| [`Detection/`](Detection/) | Experimentos de classificação binária de vulnerabilidade | `Detection/main.py` |
| [`CWE/`](CWE/) | Experimentos de geração do identificador e da descrição da CWE | `CWE/main.py` |
| [`Fix/`](Fix/) | Experimentos de geração de código corrigido | `Fix/main.py` |
| [`Pipeline/`](Pipeline/) | Inferência integrada com checkpoints locais e treinamento com LoRA | `Pipeline/main.py` e `Pipeline/train.py` |

A pipeline executa **Detection → CWE → Fix**, com um modelo carregado por vez. **Detection** compara a probabilidade prevista com o limiar configurado (padrão `0.5`). Para cada registro classificado como vulnerável, **CWE** extrai os identificadores da resposta gerada e **Fix** gera uma proposta de correção. As duas etapas recebem o código original de forma independente.

## Instalação

A partir da raiz do projeto, crie um ambiente virtual com Python 3.12 e instale as dependências básicas para inferência em arquivos locais JSON, CSV ou de código-fonte. Os comandos usam PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch transformers safetensors
```

Para treinamento, leitura de Parquet, integração com o Hugging Face e avaliação com CodeBLEU, use a lista de dependências em [`Pipeline/requirements.txt`](Pipeline/requirements.txt). Tenha o Git no `PATH`, pois o CodeBLEU é instalado a partir de um repositório Git.

```powershell
.\.venv\Scripts\python.exe -m pip install -r Pipeline/requirements.txt
```

Nos comandos seguintes, `python` deve apontar para o ambiente em que as dependências foram instaladas. No PowerShell, você pode substituir `python` por `.\.venv\Scripts\python.exe` nos comandos executados da raiz ou ativar o ambiente com `.\.venv\Scripts\Activate.ps1`.

## Executar a pipeline

A cópia do projeto contém os checkpoints abaixo, em formato `safetensors`. Mantenha os pesos, a configuração e os arquivos do tokenizador na pasta de cada modelo.

| Etapa | Checkpoint padrão em `Pipeline/checkpoints/` |
| --- | --- |
| Detection | `detection/PolyCoder-160M-bigvul` |
| CWE | `cwe/Qwen2.5-Coder-0.5B-Instruct-cvefixes` |
| Fix | `fix/Qwen2.5-Coder-0.5B-Instruct-bigvul` |

Execute o exemplo incluído no projeto a partir da raiz.

```powershell
python Pipeline/main.py --input Pipeline/examples/pipeline_input.json
```

Em JSON, informe um objeto ou uma lista de objetos com o campo `code` e um `id` opcional. Em CSV ou Parquet, use `--code-column` quando a coluna que contém o código tiver outro nome. A entrada também aceita um arquivo de código-fonte ou uma URL de dataset do Hugging Face.

```powershell
python Pipeline/main.py --input Pipeline/datasets/teste.csv --code-column vulnerable_code
python Pipeline/main.py --input caminho/funcao.c --device cpu
```

O dispositivo padrão (`--device auto`) usa CUDA quando disponível e CPU nos demais casos. Use `--detection-model`, `--cwe-model` e `--fix-model` para indicar outros checkpoints locais. `--help` lista as opções de execução, incluindo o tamanho dos lotes e os limites de tokens.

### Resultados e métricas

O resultado consolidado é salvo em `Pipeline/results/pipeline_results.json` por padrão. Cada execução cria uma pasta `pipeline_results_phases_*` ao lado do arquivo de saída, com sua própria cópia do JSON consolidado, relatórios por etapa em CSV, JSON e HTML e um `summary.json` com parâmetros, tempos e estado da execução.

| Saída avaliada | Métrica principal da pipeline |
| --- | --- |
| Detecção binária | F1 da classe vulnerável |
| Identificadores CWE | F1 multirrótulo macro e micro |
| Descrição da CWE | BLEU-4 de corpus |
| Código corrigido | CodeBLEU médio por exemplo, com zero para correções ausentes |

As métricas exigem referências, que podem estar na própria entrada ou em um arquivo informado com `--references`. Para usar arquivos separados, mantenha os mesmos IDs. A amostra `Pipeline/datasets/teste.csv` inclui `label`, `cwe_id`, `cwe_description` e `fixed_code`. Verifique também a cobertura das saídas generativas ao interpretar os resultados.

O estado `generated_unvalidated` indica que a pipeline gerou uma proposta de correção. A avaliação compara a proposta com uma referência textual. A validação por compilação e testes fica a cargo do usuário.

Use `--stop-after detection` para encerrar a primeira execução após Detection. Use `--resume-from` para continuar a partir do JSON salvo.

```powershell
python Pipeline/main.py --input Pipeline/examples/pipeline_input.json --stop-after detection --output Pipeline/results/detection.json
python Pipeline/main.py --resume-from Pipeline/results/detection.json --output Pipeline/results/final.json
```

### Notebooks

O projeto inclui uma versão para [Google Colab](Pipeline/colab_tcc.ipynb) e uma interface local em [Marimo](Pipeline/notebook.py). Para abrir a interface local, instale `marimo` no ambiente com as dependências da pipeline e execute `python -m marimo run Pipeline/notebook.py` a partir da raiz.

## Treinamento e experimentos

`Pipeline/train.py` faz ajuste fino com LoRA e salva os pesos mesclados com o modelo base e os arquivos do tokenizador. Nessa rotina, Detection usa PrimeVul. CWE e Fix usam CVEfixes por padrão, com BigVul como opção em `--source`.

Use `--smoke` para treinar e avaliar Fix com uma amostra de BigVul.

```powershell
python Pipeline/train.py --phase fix --model Qwen/Qwen2.5-Coder-0.5B-Instruct --source bigvul --smoke
```

Com `--smoke`, os checkpoints e o `results.csv` ficam em `Pipeline/checkpoints/smoke/`. Os diretórios gerados pelo treinamento usam o ID do modelo com `/` substituído por `__`. Para usar um checkpoint recém-treinado na pipeline, passe esse caminho na opção da etapa correspondente, como `--fix-model`.

Para rodar os experimentos independentes, entre no diretório da tarefa (`Detection`, `CWE` ou `Fix`) e execute `python main.py`. Os scripts leem `models.txt` e `datasets.txt` no diretório atual. Em CWE e Fix, selecione primeiro `0` para `trainer_encoder.py` ou `1` para `trainer_decoder.py`. Em seguida, selecione o dataset. O script percorre os modelos listados em `models.txt` e registra métricas, tempos e erros em `results/experiment_results.csv` dentro da pasta da tarefa. Cada linha de `datasets.txt` segue o formato `id_do_dataset: coluna_de_codigo`.

## Testes e documentação

Execute os testes automatizados a partir da raiz. Os testes usam modelos simulados para verificar o fluxo e as métricas.

```powershell
python -m unittest discover -s Pipeline/tests -v
```

- [Guia da pipeline](Pipeline/PIPELINE.md)
- [Arquitetura e implementação](Pipeline/docs/arquitetura_e_implementacao.md)

