import copy
import types
from functools import partial
from pathlib import Path

import mlx.core as mx
from mlx_lm.lora import CONFIG_DEFAULTS, run

from config import (
    MODEL_PATH,
    DATA_OUTPUT,
    ADAPTER_PATH,
    BATCH_SIZE,
    MAX_SEQ_LENGTH,
    LEARNING_RATE,
    ITERATIONS,
    NUM_LAYERS,
    LORA_RANK,
    RANDOM_SEED,
)


def main():
    data_output = Path(DATA_OUTPUT)
    adapter_path = Path(ADAPTER_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    if not (data_output / "train.jsonl").exists():
        raise FileNotFoundError(
            "Dataset não preparado. Execute prepare_dataset.py."
        )

    # Carregar configurações padrão da versão instalada
    config = copy.deepcopy(CONFIG_DEFAULTS)

    # Sobrescrever com nossas configurações
    config.update({
        "model": str(MODEL_PATH),
        "train": True,
        "test": True,

        "data": str(data_output),
        "adapter_path": str(adapter_path),

        "fine_tune_type": "lora",

        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,

        "learning_rate": LEARNING_RATE,
        "iters": ITERATIONS,

        "num_layers": NUM_LAYERS,

        "lora_parameters": {
            "rank": LORA_RANK,
            "dropout": 0.05,
            "scale": 20.0,
            "keys": [
                "self_attn.q_proj",
                "self_attn.v_proj",
            ],
        },

        "mask_prompt": True,

        "steps_per_report": 10,
        "steps_per_eval": 50,
        "save_every": 50,

        "val_batches": -1,
        "test_batches": -1,

        "seed": RANDOM_SEED,

        "grad_checkpoint": True,
    })

    args = types.SimpleNamespace(**config)

    adapter_path.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Starting LoRA fine-tuning...")
    print("Model:", MODEL_PATH)
    print("Dataset:", data_output)
    print("Adapters:", adapter_path)

    # mlx-lm calls init() repeatedly; force ring on every call because `any`
    # auto-detects Conda's incompatible MPICH installation.
    mx.distributed.init = partial(mx.distributed.init, backend="ring")
    run(args)

    print("\nTraining finished!")


if __name__ == "__main__":
    main()