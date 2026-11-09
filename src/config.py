from pathlib import Path

# Modelo local
MODEL_PATH = (
    Path.home()
    / ".lmstudio/models/lmstudio-community/"
    "Qwen3-4B-Instruct-2507-MLX-4bit"
)

# Dados
DATA_SOURCE = "./src/data/processed"
DATA_OUTPUT = "./src/data/mlx_ready"

# Onde serão salvos os adapters
ADAPTER_PATH = "./adapters"

# Configurações de treinamento
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048

LEARNING_RATE = 1e-5
ITERATIONS = 200

NUM_LAYERS = 8
LORA_RANK = 8

VALIDATION_SIZE = 0.10
RANDOM_SEED = 42