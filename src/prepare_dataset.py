import json
import random
from pathlib import Path
from typing import Any

from config import (
    DATA_SOURCE,
    DATA_OUTPUT,
    VALIDATION_SIZE,
    RANDOM_SEED,
)


SYSTEM_PROMPT = """
You are a biomedical research assistant.

Your task is to answer questions based on the scientific
contexts provided by the user.

Use the supplied evidence to formulate your answer.

Do not invent findings or information that are not
supported by the provided contexts.

Clearly communicate uncertainty when the evidence
is insufficient.

This is a research assistance task, not a substitute
for professional medical judgment.
""".strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {error}"
                )

    return records


def format_example(record: dict[str, Any]) -> dict[str, Any] | None:
    question = record.get("QUESTION")
    contexts = record.get("CONTEXTS")
    answer = record.get("LONG_ANSWER")

    if not question or not answer or not contexts:
        return None

    if isinstance(contexts, list):
        contexts_text = "\n\n".join(
            str(context) for context in contexts
        )
    else:
        contexts_text = str(contexts)

    user_content = f"""
Question:
{question}

Scientific contexts:
{contexts_text}

Answer based on the provided evidence.
""".strip()

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": str(answer),
            },
        ]
    }


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    random.seed(RANDOM_SEED)

    data_source = Path(DATA_SOURCE)
    data_output = Path(DATA_OUTPUT)

    original_train = read_jsonl(
        data_source / "train.jsonl"
    )

    original_test = read_jsonl(
        data_source / "test.jsonl"
    )

    print("Original training examples:", len(original_train))
    print("Original test examples:", len(original_test))

    # Converter os exemplos
    train_data = [
        item
        for record in original_train
        if (item := format_example(record)) is not None
    ]

    test_data = [
        item
        for record in original_test
        if (item := format_example(record)) is not None
    ]

    # Embaralhar apenas o conjunto de treinamento
    random.shuffle(train_data)

    validation_count = int(
        len(train_data) * VALIDATION_SIZE
    )

    valid_data = train_data[:validation_count]
    train_data = train_data[validation_count:]

    # Salvar os arquivos
    save_jsonl(
        train_data,
        data_output / "train.jsonl"
    )

    save_jsonl(
        valid_data,
        data_output / "valid.jsonl"
    )

    save_jsonl(
        test_data,
        data_output / "test.jsonl"
    )

    print("\nDataset preparado!")
    print("Training:", len(train_data))
    print("Validation:", len(valid_data))
    print("Test:", len(test_data))


if __name__ == "__main__":
    main()