import json

from config import MODEL_PATH

config_file = MODEL_PATH / "config.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modelo não encontrado: {MODEL_PATH}"
    )

if not config_file.exists():
    raise FileNotFoundError("config.json não encontrado")

with open(config_file, "r", encoding="utf-8") as file:
    config = json.load(file)

print("Arquitetura:", config.get("model_type"))
print("Arquiteturas:", config.get("architectures"))
print("Quantização:", config.get("quantization_config"))

print("\nConfiguração completa:")
print(json.dumps(config, indent=4))