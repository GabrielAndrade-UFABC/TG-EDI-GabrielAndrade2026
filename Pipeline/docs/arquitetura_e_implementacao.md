# Arquitetura e implementação da pipeline Detection → CWE → Fix

## 1. Visão geral

O sistema transforma um conjunto de funções em C em três respostas por função:

(i) se a função é vulnerável
(ii) qual a fraqueza (identificador e descrição da CWE)
(iii) uma proposta de correção. 


Cada resposta é produzida por um modelo Transformer especializado e avaliada por uma métrica própria:

| Etapa | Tarefa | Arquitetura | Métrica |
| --- | --- | --- | --- |
| Detection | Classificação binária | `AutoModelForSequenceClassification` | F1 (classe positiva = vulnerável), precisão, recall, acurácia |
| CWE | Geração do texto `CWE-N: descrição` | `AutoModelForCausalLM` | F1 multirrótulo do identificador (macro e micro); BLEU-4 da descrição |
| Fix | Geração do código corrigido | `AutoModelForCausalLM` | CodeBLEU (`lang="c"`) e seus quatro componentes |

As etapas são encadeadas de forma condicional: apenas as funções que a Detection classifica como vulneráveis seguem para a CWE, e apenas as que recebem um identificador CWE válido seguem para o Fix. A medição é, portanto, de ponta a ponta: um falso negativo da Detection não recebe CWE nem correção e é contabilizado como erro nas etapas seguintes, como ocorreria em uso real.

O projeto é composto por quatro artefatos:

| Arquivo | Papel |
| --- | --- |
| `main.py` | Ponto de entrada: valida o ambiente e delega a execução |
| `pipeline.py` | Wrapper de inferência: entrada, encadeamento das etapas, persistência e métricas |
| `train.py` | Ajuste fino (LoRA) dos modelos de cada etapa, com os mesmos prompts e métricas da inferência |
| `notebook.py` | Interface interativa (marimo) equivalente ao notebook do Google Colab, executada localmente |

## 2. Evolução em relação à versão original

A primeira versão do trabalho era formada por três projetos independentes (`MultiTransformer - Detection`, `- CWE` e `- Fix`), cada um com seu `main.py`, seu treinamento e sua avaliação. A análise dessa versão revelou limitações que motivaram a reestruturação:

1. **Ausência de encadeamento.** Cada etapa era avaliada isoladamente sobre seu próprio conjunto de teste; nunca se mediu o desempenho do sistema completo.
2. **Avaliação por *teacher forcing*.** As métricas de CWE e Fix eram calculadas por `Trainer.predict`, que não gera texto: em cada posição, o modelo prevê o próximo token tendo acesso aos tokens corretos anteriores. Isso superestima a qualidade em relação à geração autorregressiva real.
3. **Vazamento da resposta em modelos encoder.** Modelos bidirecionais (RoBERTa/CodeBERTa, GraphCodeBERT) treinados como modelos causais sem atenção causal enxergam a resposta durante o treino, produzindo perdas próximas de zero e métricas artificialmente altas.
4. **Rótulo incorreto na Detection para PrimeVul/DiverseVul.** O rótulo era derivado da coluna `cwe`, preenchida também em funções seguras; o rótulo oficial é `target`. Na prática, quase todas as amostras eram rotuladas como vulneráveis.
5. **Truncamento que removia o alvo.** O corte em 256 tokens descartava a resposta na maior parte dos exemplos de CWE.
6. **Credencial no código-fonte.** O token do Hugging Face estava escrito diretamente nos três `main.py`.

A versão atual corrige esses pontos: as três etapas rodam encadeadas, as métricas são calculadas sobre texto efetivamente gerado, apenas modelos decoder são usados nas etapas generativas, a Detection usa o rótulo `target`, exemplos que não cabem no contexto são descartados em vez de truncados e o token foi removido.

## 3. Ponto de entrada (`main.py`)

O `main.py` é deliberadamente pequeno: não contém lógica de inferência, apenas garante que o ambiente está pronto antes de delegar a execução ao `pipeline.py`. Essa separação permite que a ajuda (`--help`) funcione mesmo em uma máquina sem PyTorch instalado e que falhas de ambiente sejam relatadas uma única vez, antes de qualquer modelo ser carregado, em vez de se repetirem registro a registro.

### 3.1 Componentes

- **`DependencyCheck`** verifica, com `importlib.util.find_spec`, se os pacotes necessários à inferência (`torch`, `transformers`, `safetensors`) estão disponíveis, sem importá-los. Se faltar algum e a sessão for interativa, oferece instalá-lo com o `pip` do próprio interpretador (`sys.executable -m pip`), o que evita instalar no ambiente errado. Em sessões não interativas, encerra com a linha de comando exata a ser executada.
- **`ModelCheck`** confere se cada etapa possui um checkpoint (`model.safetensors`) no caminho padrão. Etapas cujo modelo foi substituído por linha de comando (`--detection-model`, `--cwe-model`, `--fix-model`) não são verificadas. A importação de `pipeline.DEFAULT_MODELS` é tardia, ocorrendo só depois da verificação de dependências.
- **`Launcher`** orquestra a ordem: dependências → checkpoints → pipeline. Quando o usuário pede ajuda, as verificações são puladas.

### 3.2 Decisões de projeto

**Metaclasse `SlottedMeta` e `__slots__`.** Todas as classes do ponto de entrada herdam de `Slotted`, cuja metaclasse recusa, no momento da definição da classe, qualquer subclasse que não declare `__slots__`. Com `__slots__`, as instâncias não possuem `__dict__`: os atributos são fixos e declarados, um erro de digitação em uma atribuição (`self.dependecies = ...`) gera `AttributeError` em vez de criar silenciosamente um novo atributo, e cada instância ocupa menos memória. A metaclasse transforma essa convenção em uma regra verificada pelo interpretador, e não por revisão manual.

**Injeção de dependências.** As operações com efeito colateral — procurar pacotes, executar o `pip`, ler do teclado, detectar se o terminal é interativo — são recebidas como parâmetros com valores padrão reais. Em produção nada muda; nos testes, essas funções são substituídas por versões controladas, permitindo exercitar todos os caminhos (pacote ausente, instalação recusada, instalação com falha, sessão não interativa) sem instalar nada nem depender do ambiente.

**Desenvolvimento orientado a testes.** A reestruturação do `main.py` foi feita com TDD: cada comportamento foi primeiro descrito por um teste que falhava e só então implementado. Os 16 testes de `tests/test_main.py` cobrem a metaclasse, os sete caminhos do `DependencyCheck`, o `ModelCheck`, a ordem de execução do `Launcher` e uma execução real de `main.py --help` em subprocesso.

## 4. Wrapper de inferência (`pipeline.py`)

### 4.1 Entrada e identificação dos registros

A entrada pode ser um CSV, um JSON (lista de objetos) ou um arquivo de código isolado. A coluna do código é configurável (`--code-column`), o que permite usar diretamente conjuntos como BigVul (`func_before`) ou CVEfixes (`vulnerable_code`). Cada registro recebe um identificador estável: o campo `id`, quando existe, ou a posição original da linha. Identificadores vazios ou duplicados são rejeitados na leitura, pois todos os arquivos de saída e o cálculo das métricas dependem da correspondência por identificador.

### 4.2 Máquina de estados

Cada registro carrega um `status` que determina a próxima etapa elegível:

```
pending ──Detection──▶ detected ──CWE──▶ classified ──Fix──▶ generated_unvalidated
   │                       │
   └──▶ not_vulnerable     └──▶ cwe_unidentified
qualquer etapa ──falha──▶ error
```

O estado final do Fix chama-se `generated_unvalidated` para deixar explícito que a correção não foi compilada nem testada: o CodeBLEU mede similaridade com a correção de referência, não a efetividade do patch.

O histórico de cada etapa (status, motivo, tempo, modelo e dispositivo usados) fica em `phase_results`, o que permite auditar a decisão tomada para cada registro.

### 4.3 Gerenciamento de memória da GPU

Os três modelos nunca ficam carregados simultaneamente. Para cada etapa, o wrapper carrega o modelo, processa todos os registros elegíveis, libera as referências, executa a coleta de lixo e chama `torch.cuda.empty_cache()` antes de carregar o modelo seguinte. Assim, o pico de memória corresponde ao maior dos três modelos, e não à soma deles, o que viabiliza a execução em uma GPU de 6 GB.

### 4.4 Prompts e pós-processamento

Os prompts de inferência são os mesmos usados no treino, definidos uma única vez (`cwe_prompt` e `fix_prompt`) e importados pelo `train.py`, de modo que treino e inferência não podem divergir:

- CWE: `### Vulnerable code:` + código + `### Identify the CWE:`
- Fix: `### CWE:` + resposta da etapa anterior + `### Buggy code:` + código + `### Fixed code:`

A resposta gerada passa por três regras de limpeza:

1. A geração é cortada no primeiro marcador `### `, o que descarta a continuação do prompt quando o modelo não emite o token de fim de sequência.
2. Na CWE, considera-se apenas a primeira linha, que é o formato do alvo de treino (`CWE-N: descrição`).
3. Os identificadores são extraídos pela expressão regular genérica `\bCWE-\d+\b`, sem lista fixa de classes. Se nenhum for encontrado, o registro é marcado como `cwe_unidentified` e não segue para o Fix.

Na Detection, a entrada é truncada em 512 tokens, o mesmo limite do treino. Nas etapas generativas, uma entrada que não cabe no contexto do modelo, já descontado o espaço reservado à resposta, gera um erro explícito naquele registro, em vez de um truncamento silencioso que cortaria o código a ser analisado.

### 4.5 Processamento em lote

A inferência é feita em lotes de oito registros. Antes de processar cada grupo, o wrapper chama `LocalModel.prefetch`, que tokeniza o lote com preenchimento, executa uma única chamada ao modelo e guarda os resultados. Em seguida, cada registro passa normalmente pelo fluxo individual, que lê o resultado já calculado. Esse desenho preserva todo o tratamento por registro (erro, histórico, checkpoint e modo de depuração) e acrescenta duas salvaguardas:

- **Fallback:** se o lote falhar, por exemplo por falta de memória, o wrapper avisa e processa aquele grupo um registro por vez.
- **Entradas longas:** uma entrada longa demais fica fora do lote e recebe, no caminho individual, o mesmo erro que receberia sem o lote.

O preenchimento é feito à esquerda nas etapas generativas, exigência da geração autorregressiva, e à direita na classificação, como no treino.

### 4.6 Persistência, retomada e robustez

- **Gravação atômica.** Todo arquivo JSON é escrito primeiro em um temporário exclusivo no mesmo diretório e depois renomeado (`Path.replace`). Uma interrupção durante a escrita nunca deixa um arquivo de resultados truncado.
- **Checkpoint periódico.** O resultado consolidado é regravado após cada registro, com intervalo mínimo de cinco segundos, para evitar custo quadrático em entradas grandes, e ao fim de cada etapa.
- **Retomada.** `--resume-from` continua uma execução interrompida a partir do JSON de resultados, executando apenas as etapas pendentes. O arquivo é validado campo a campo (tipos, identificadores CWE, coerência entre status e previsões) antes de ser aceito. `--retry-errors` reexecuta apenas os registros que terminaram em erro, e `--stop-after` permite rodar só as primeiras etapas.
- **Saídas por etapa.** Ao fim de cada etapa são gerados CSV, JSON e uma tabela HTML. Nos CSV, células que começam com `=`, `+`, `-` ou `@` são exportadas como texto, prevenindo a injeção de fórmulas ao abrir a planilha. No HTML, todo conteúdo é escapado, já que o código analisado pode conter qualquer sequência de caracteres.
- **Separação entre entrada e saída.** O wrapper recusa usar o mesmo arquivo como entrada e saída, inclusive por links, o que evitaria sobrescrever o conjunto original.

### 4.7 Métricas

As métricas são calculadas pelo próprio wrapper ao fim de cada etapa, a partir de um arquivo de referências que pode ser a própria entrada. Os nomes de coluna mais comuns são reconhecidos automaticamente (por exemplo, `vul`, `label` ou `target` para o rótulo e `CWE ID` ou `cwe_id` para a CWE).

- **Detection:** F1 binário, precisão, recall, acurácia e matriz de confusão, sobre todos os registros com rótulo.
- **CWE (identificador):** F1 multirrótulo macro e micro sobre os registros vulneráveis com CWE conhecida. Um registro que a Detection julgou seguro conta como previsão vazia, ou seja, como erro.
- **CWE (descrição):** BLEU-4 de corpus, com tokenização 13a e sem suavização, implementado no próprio wrapper. O prefixo `CWE-N:` é removido dos dois lados, para que o BLEU meça apenas o texto da descrição. Quando o conjunto traz só o identificador, como o BigVul, a descrição de referência é obtida de uma tabela de 267 CWEs (`cwe_descriptions.json`) extraída do CVEfixes, o mesmo texto visto pelo modelo no treino.
- **Fix:** CodeBLEU (`calc_codebleu`, `lang="c"`), com os componentes n-grama, n-grama ponderado, sintaxe e fluxo de dados. Assim como no identificador da CWE, um registro que não chegou ao Fix por decisão de uma etapa anterior conta como correção vazia.

Cada métrica informa quantos registros foram efetivamente avaliados em relação aos esperados. Assim, uma cobertura baixa fica visível em vez de inflar o resultado.

No BigVul, as funções seguras repetem a CWE e o `func_after` do commit que corrigiu outra função. O wrapper descarta esses campos nas funções seguras, para que não sejam tratados como rótulos.

## 5. Treinamento (`train.py`)

- **Modelos:** apenas decoders (Qwen2.5-Coder-0.5B-Instruct, PolyCoder-160M e GPT-2), ajustados com LoRA (r = 16, α = 32, todas as camadas lineares).
- **Perda apenas na resposta:** os tokens do prompt recebem rótulo −100 e não entram na perda. A resposta termina com o token de fim de sequência, o que ensina o modelo a parar.
- **Sem truncamento do alvo:** exemplos que não cabem em 768 tokens são descartados.
- **Precisão numérica:** a RTX 2060 (Turing) não suporta bf16. Os pesos são carregados em fp32 e o treino usa precisão mista fp16, exceto no Gemma 3, que transborda em fp16.
- **Dados:**
  - Detection: PrimeVul com o rótulo `target`. Treino e validação balanceados; teste na proporção real do conjunto (cerca de 2,2% de funções vulneráveis), já que um teste balanceado infla precisão e F1.
  - CWE e Fix: CVEfixes (C/C++), opcionalmente somado aos splits de treino e validação do BigVul (`--source cvefixes bigvul`).
- **Controle de vazamento:** toda função do treino ou da validação que também aparece no teste, comparada sem espaços em branco, é removida. Esse filtro encontrou e eliminou 48 duplicatas internas do CVEfixes e cerca de 380 do BigVul.
- **Avaliação:** feita pelo mesmo caminho da inferência (`LocalModel` + `predict`), ou seja, com geração real e as mesmas métricas. Uma falha (por exemplo, resposta vazia) conta como resposta errada e permanece na média. `--eval-only` reavalia checkpoints já treinados.
- **Baselines:** cada avaliação traz, no mesmo teste, a métrica de uma resposta trivial: sempre "vulnerável" (Detection), sempre a CWE mais frequente no treino (CWE) e o código vulnerável devolvido sem alteração (Fix). O modelo só demonstra aprendizado quando supera o seu baseline.

## 6. Desempenho

### 6.1 Onde o tempo é gasto

Antes de otimizar, mediu-se a distribuição do tempo de uma execução com 100 funções do BigVul:

| Componente | Tempo |
| --- | --- |
| Inicialização do `main.py` (imports e verificações) | ≈ 0,1 s (0,1%) |
| Detection | 11,4 s |
| CWE | 35,9 s |
| Fix | 151,8 s |
| **Total** | **199,5 s** |

Praticamente todo o tempo é inferência na GPU, isto é, a geração autorregressiva token a token. O código Python de orquestração responde por uma fração desprezível.

### 6.2 Técnicas avaliadas e descartadas

Foram consideradas técnicas de aceleração do código Python, como compilação com Cython e compilação *just-in-time* com Numba. Ambas aceleram laços numéricos executados no interpretador Python. Neste sistema, esses laços não existem no caminho crítico: o trabalho pesado já é executado em kernels CUDA pelo PyTorch, e o código Python apenas orquestra chamadas. Pela lei de Amdahl, acelerar uma fração de 0,1% do tempo produz ganho total inferior a 0,1%, mesmo com aceleração infinita dessa parte. As duas técnicas foram, portanto, descartadas, por acrescentarem dependências e complexidade de compilação sem benefício mensurável.

### 6.3 Técnicas adotadas

O esforço foi direcionado ao que domina o tempo, a chamada ao modelo.

**Processamento em lote.** Medição no Fix, com 47 funções e 384 tokens de geração:

| Configuração | Tempo | Saídas idênticas à execução sem lote | Pico de VRAM |
| --- | --- | --- | --- |
| Lote 1, fp32 | 130,4 s | 47/47 | 0,7 GB |
| Lote 8, fp32 (adotado) | 41,9 s (3,1×) | 47/47 | 1,2 GB |
| Lote 8, fp16 | 22,3 s (5,8×) | 47/47 | 0,7 GB |
| Lote 16, fp16 | 16,4 s (8,0×) | 46/47 | 0,9 GB |

Adotou-se o lote de 8 em fp32, a configuração mais rápida entre as que comprovadamente não alteram nenhuma saída. Na pipeline completa, com 100 funções, o tempo caiu de 199,5 s para 70,3 s (2,8×), com resultados idênticos registro a registro.

**Limite de geração.** O limite padrão de 128 tokens novos cortava as correções no meio da função. Elevá-lo para 384 tokens aumentou o CodeBLEU de 0,378 para 0,722 na mesma amostra. Combinado com o processamento em lote, o resultado foi mais rápido (70 s) que a configuração original (93 s).

## 7. Validade dos resultados

- **Vazamento entre treino e teste.** 347 das 1.000 funções da amostra de avaliação do BigVul aparecem no treino do PrimeVul, usado na Detection. Separando os dois grupos, o F1 foi de 0,887 nas funções vistas e de 0,889 nas inéditas, sem indício de memorização. Nenhuma das funções vulneráveis avaliadas consta do treino de CWE e Fix.
- **Baseline do Fix.** No BigVul, a correção de referência costuma alterar poucas linhas. Por isso, devolver o código vulnerável sem nenhuma alteração já obtém CodeBLEU de 0,810 nas mesmas 405 funções em que o modelo obteve 0,658. O CodeBLEU do Fix deve ser sempre reportado junto desse baseline.
- **Piso do CodeBLEU.** O pacote `codebleu` substitui um fluxo de dados igual a 0 por 1 (`dataflow_match_score or 1`). Uma única correção vazia ainda recebe 0,25. Em corpus de centenas de funções o efeito é desprezível, mas valores calculados sobre poucas funções devem ser lidos com cautela.
- **Variação do CodeBLEU.** A biblioteca `codebleu` produz variações da ordem de ±0,001 entre execuções com saídas idênticas, dependentes da semente de hash do Python. Para valores exatamente reprodutíveis, executa-se com `PYTHONHASHSEED=0`.
- **Determinismo.** A geração é gulosa (`do_sample=False`). Para uma mesma entrada e um mesmo checkpoint, as saídas são idênticas entre execuções.

## 8. Qualidade de software

O projeto conta com 72 testes automatizados (`python -m unittest discover -s tests`):

| Arquivo | Testes | Escopo |
| --- | --- | --- |
| `test_main.py` | 16 | Ponto de entrada, metaclasse, dependências, checkpoints |
| `test_pipeline.py` | 17 | Encadeamento, retomada, falhas de carga e de gravação, lotes e fallback |
| `test_metrics.py` | 24 | F1, BLEU, CodeBLEU, leitura de referências, descrições de CWE |
| `test_train.py` | 15 | Mascaramento da perda, filtros de dados, balanceamento, divisão, vazamento, falhas na avaliação e baselines |

Os testes usam modelos falsos, de comportamento determinístico, e rodam em menos de um segundo, sem GPU e sem baixar modelos. Assim, podem ser executados a cada alteração.

## 9. Reprodução

```fish
cd TCC
python -m venv .venv && source .venv/bin/activate.fish
pip install -r requirements.txt
python -m unittest discover -s tests                     # testes
python train.py                                          # treina e avalia as três fases (gera checkpoints/)
python train.py --phase cwe fix --source cvefixes bigvul --output checkpoints/bigvul
python main.py --input amostra.csv --code-column func_before
marimo edit notebook.py                                  # interface interativa
```
