from pathlib import Path

from config import (
    BATCH_SIZE,
    DATA_OUTPUT,
    ITERATIONS,
    LEARNING_RATE,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    NUM_LAYERS,
    RANDOM_SEED,
    TORCH_ADAPTER_PATH,
    TORCH_MODEL_ID,
)


def tokenize_example(record, tokenizer):
    messages = record["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Cada exemplo deve terminar com uma resposta do assistant.")

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("O chat template do modelo nao permite mascarar o prompt.")

    input_ids = input_ids[:MAX_SEQ_LENGTH]
    prompt_length = min(len(prompt_ids), len(input_ids))
    # Somente a resposta do assistant participa do calculo da loss.
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main():
    try:
        import torch
        from datasets import load_dataset
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit(
            "Dependencias de treino ausentes ou incompativeis. "
            "Siga a instalacao para Windows no README.md."
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit(
            "O treino no Windows requer uma GPU NVIDIA com CUDA. "
            "Confira o driver NVIDIA e a instalacao do PyTorch com CUDA no README.md."
        )

    project_root = Path(__file__).resolve().parent.parent
    data_output = project_root / DATA_OUTPUT
    adapter_path = project_root / TORCH_ADAPTER_PATH
    data_files = {
        "train": data_output / "train.jsonl",
        "validation": data_output / "valid.jsonl",
        "test": data_output / "test.jsonl",
    }
    for path in data_files.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset nao encontrado: {path}. Execute python src/prepare_dataset.py."
            )

    set_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(TORCH_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    datasets = load_dataset(
        "json", data_files={split: str(path) for split, path in data_files.items()}
    )
    datasets = datasets.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=datasets["train"].column_names,
    )
    for split, dataset in datasets.items():
        # Um prompt que ocupa toda a sequencia nao deixa tokens para aprender.
        usable = dataset.filter(lambda record: any(label != -100 for label in record["labels"]))
        print(f"{split}: {len(usable)}/{len(dataset)} exemplos com tokens de resposta.")
        if not len(usable):
            raise ValueError(
                f"Dataset {split} sem respostas utilizaveis. Confira os dados e MAX_SEQ_LENGTH."
            )
        datasets[split] = usable

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print("Starting QLoRA fine-tuning (PyTorch/CUDA)...")
    print("GPU:", torch.cuda.get_device_name())
    print("Model:", TORCH_MODEL_ID)
    print("Dataset:", data_output)
    print("Adapters:", adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        TORCH_MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    layer_count = model.config.num_hidden_layers
    if not 0 < NUM_LAYERS <= layer_count:
        raise ValueError(f"NUM_LAYERS deve estar entre 1 e {layer_count}.")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            # PEFT usa alpha/rank; MLX multiplica diretamente pelo scale 20.
            lora_alpha=20 * LORA_RANK,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=list(range(layer_count - NUM_LAYERS, layer_count)),
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_path),
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            max_steps=ITERATIONS,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            weight_decay=0.0,
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            prediction_loss_only=True,
            label_names=["labels"],
            dataloader_num_workers=0,
            report_to="none",
            seed=RANDOM_SEED,
            data_seed=RANDOM_SEED,
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
    )
    result = trainer.train()
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    print("\nTraining finished! Adapters:", adapter_path)


if __name__ == "__main__":
    main()
