from mlx_lm import load

from config import MODEL_PATH

print("Carregando modelo...")

model, tokenizer = load(str(MODEL_PATH)) # type: ignore

print("Modelo carregado com sucesso!")
print("Tokenizer:", type(tokenizer).__name__)