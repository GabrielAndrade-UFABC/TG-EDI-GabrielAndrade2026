MODELS_FILE = "models.txt"
DATASETS_FILE = "datasets.txt"

MODELS = []
MODELS_TYPE = {}
DATASETS = []
TEXT_COLUMNS = {}

# with open(MODELS_FILE, 'r') as f:
#     MODELS = f.read().splitlines()

with open(MODELS_FILE, 'r') as f:
    MODELS_TYPE = {line.split(': ', 1)[0].strip(): line.split(': ', 1)[1].strip() 
                 for line in f if ': ' in line}
    MODELS = list(MODELS_TYPE.keys())

with open(DATASETS_FILE, 'r') as f:
    TEXT_COLUMNS = {line.split(': ', 1)[0].strip(): line.split(': ', 1)[1].strip() 
                 for line in f if ': ' in line}
    DATASETS = list(TEXT_COLUMNS.keys())

for model_name in MODELS:
    model_type = MODELS_TYPE[model_name]
    print(f"{model_name}: {model_type}")