import argparse
import sys
from contextlib import nullcontext
from functools import cache
from pathlib import Path

from config import ADAPTER_PATH, DATA_OUTPUT, MODEL_PATH, TORCH_ADAPTER_PATH
from prepare_dataset import SYSTEM_PROMPT, load_hospital_examples, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_local_model():
    if sys.platform == "darwin":
        return load_mlx_model()
    return load_torch_model()


def load_mlx_model():
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Modelo MLX local não encontrado em {MODEL_PATH}.")
    adapter_path = PROJECT_ROOT / ADAPTER_PATH
    if not all((adapter_path / name).is_file() for name in ("adapter_config.json", "adapters.safetensors")):
        raise RuntimeError(f"Adapter MLX não encontrado em {adapter_path}. Execute src/train.py no Mac.")
    try:
        from mlx_lm import generate, load
    except ImportError as error:
        raise RuntimeError("Use a .venv do macOS com as dependências de requirements-macos.txt.") from error

    @cache
    def load_variant(use_adapter):
        return load(str(MODEL_PATH), adapter_path=str(adapter_path) if use_adapter else None)

    print("Carregando modelo local e adapter treinado (MLX)...")
    load_variant(True)

    def answer(messages, *, use_adapter=True):
        model, tokenizer = load_variant(use_adapter)
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        return generate(model, tokenizer, prompt=tokens, max_tokens=512, verbose=False)

    return answer


def load_torch_model():
    adapter_path = PROJECT_ROOT / TORCH_ADAPTER_PATH
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Adapter treinado nao encontrado em {adapter_path}.")
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Use a .venv com as dependencias de requirements-windows.txt.") from error

    if not torch.cuda.is_available():
        raise RuntimeError("Este comando requer uma GPU NVIDIA e PyTorch com CUDA.")

    print("Carregando modelo local e adapter treinado...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        local_files_only=True,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    ).eval()

    def answer(messages, *, use_adapter=True):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        adapter_context = nullcontext() if use_adapter else model.disable_adapter()
        with adapter_context, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    return answer


def main():
    parser = argparse.ArgumentParser(description="Testar o modelo treinado no Windows ou no macOS.")
    parser.add_argument("--sample", type=int, help="Indice de um exemplo do conjunto de teste (inicia em 0).")
    parser.add_argument("--compare", action="store_true", help="Mostrar tambem a resposta do modelo original.")
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--data", type=Path, help="Arquivo JSONL alternativo para usar com --sample.")
    dataset.add_argument("--hospital", action="store_true", help="Avaliar os exemplos hospitalares de teste.")
    args = parser.parse_args()
    if args.data is not None and args.sample is None:
        parser.error("Use --data junto com --sample para escolher um exemplo.")

    examples = []
    if args.hospital:
        examples = load_hospital_examples()["test"]
    elif args.sample is not None:
        data_path = args.data if args.data is not None else Path(DATA_OUTPUT) / "test.jsonl"
        examples = read_jsonl(PROJECT_ROOT / data_path)
    if args.sample is not None:
        if not 0 <= args.sample < len(examples):
            parser.error(f"--sample deve estar entre 0 e {len(examples) - 1}.")
        examples = [examples[args.sample]]

    try:
        answer = load_local_model()
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    def show_answers(messages):
        if args.compare:
            print("\nMODELO ORIGINAL:\n", answer(messages, use_adapter=False))
        print("\nMODELO TREINADO:\n", answer(messages))

    if examples:
        for example in examples:
            if "id" in example:
                print("\nCASO:", example["id"], "| Categoria:", example.get("category", ""))
            print("\nPERGUNTA E CONTEXTO:\n", example["messages"][-2]["content"])
            show_answers(example["messages"][:-1])
            print("\nRESPOSTA DE REFERENCIA:\n", example["messages"][-1]["content"])
            for title, key in (("CRITERIOS ESPERADOS", "must_include"), ("NAO DEVE AFIRMAR", "must_not_claim")):
                if example.get(key):
                    print(f"\n{title}:")
                    for criterion in example[key]:
                        print("-", criterion)
        return

    print("Digite uma pergunta e seu contexto cientifico. Digite sair para encerrar.")
    while True:
        try:
            question = input("\nPergunta: ").strip()
            if question.casefold() in {"sair", "exit", "quit"}:
                return
            if not question:
                continue
            context = input("Contexto cientifico (em uma linha): ").strip()
            if not context:
                print("Informe o contexto: o modelo foi treinado para responder com evidencias.")
                continue
            show_answers([
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nScientific contexts:\n{context}"
                        "\n\nAnswer based on the provided evidence."
                    ),
                },
            ])
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrado.")
            return


if __name__ == "__main__":
    main()
