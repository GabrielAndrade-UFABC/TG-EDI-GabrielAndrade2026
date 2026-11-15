import os
from datasets import load_dataset

# os.environ["HF_TOKEN"] = TOKEN

# 1. Defina o caminho da pasta e o nome do arquivo
# (Exemplo usando caminho relativo. Para caminhos absolutos no Windows use: "C:/usuarios/nome/pasta")
pasta_destino = "inputs"
nome_arquivo = os.path.join(pasta_destino, "bigvul.csv")

# 2. Garanta que a pasta exista (cria se não existir)
os.makedirs(pasta_destino, exist_ok=True)

# 3. Carregue e salve o dataset
dataset = load_dataset("bstee615/bigvul")
dataset["test"].to_csv(nome_arquivo)