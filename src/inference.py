import argparse
from pathlib import Path

from config import DATA_OUTPUT, TORCH_ADAPTER_PATH
from prepare_dataset import SYSTEM_PROMPT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Testar o modelo treinado no Windows.")
    parser.add_argument("--sample", type=int, help="Indice de um exemplo do conjunto de teste (inicia em 0).")
    parser.add_argument("--compare", action="store_true", help="Mostrar tambem a resposta do modelo original.")
    parser.add_argument("--data", type=Path, help="Arquivo JSONL alternativo para usar com --sample.")
    args = parser.parse_args()
    if args.data is not None and args.sample is None:
        parser.error("Use --data junto com --sample para escolher um exemplo.")

    project_root = Path(__file__).resolve().parent.parent
    adapter_path = project_root / TORCH_ADAPTER_PATH
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise SystemExit(f"Adapter treinado nao encontrado em {adapter_path}.")

    example = None
    if args.sample is not None:
        data_path = args.data if args.data is not None else Path(DATA_OUTPUT) / "test.jsonl"
        examples = read_jsonl(project_root / data_path)
        if not 0 <= args.sample < len(examples):
            parser.error(f"--sample deve estar entre 0 e {len(examples) - 1}.")
        example = examples[args.sample]

    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise SystemExit("Use a .venv com as dependencias de requirements-windows.txt.") from error

    if not torch.cuda.is_available():
        raise SystemExit("Este comando requer uma GPU NVIDIA e PyTorch com CUDA.")

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

    def answer(messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
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

    def show_answers(messages):
        if args.compare:
            with model.disable_adapter():
                print("\nMODELO ORIGINAL:\n", answer(messages))
        print("\nMODELO TREINADO:\n", answer(messages))

    if example is not None:
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
