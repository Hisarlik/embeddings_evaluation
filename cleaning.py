import json
import os
from typing import Dict

def load_jsonl(file_path: str) -> Dict:
    data = {}
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_jsonl(data: Dict, file_path: str) -> None:
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Create data directory paths
    data_dir = "data"
    train_path = os.path.join(data_dir, "training_dataset_eu_regulation.jsonl")
    val_path = os.path.join(data_dir, "val_dataset_eu_regulation.jsonl")
    test_path = os.path.join(data_dir, "test_dataset_eu_regulation.jsonl")

    # Load datasets from JSONL files
    train_dataset = load_jsonl(train_path)
    val_dataset = load_jsonl(val_path)
    test_dataset = load_jsonl(test_path)

    train_dataset["questions"] = {clave: valor for clave, valor in train_dataset["questions"].items() if len(valor) > 20}
    val_dataset["questions"] = {clave: valor for clave, valor in val_dataset["questions"].items() if len(valor) > 20}
    test_dataset["questions"] = {clave: valor for clave, valor in test_dataset["questions"].items() if len(valor) > 20}

    train_dataset["questions"] = {clave: valor.replace("QUESTION #1: ","").replace("QUESTION #2: ","") for clave, valor in train_dataset["questions"].items()}
    val_dataset["questions"] = {clave: valor.replace("QUESTION #1: ","").replace("QUESTION #2: ","") for clave, valor in val_dataset["questions"].items() }
    test_dataset["questions"] = {clave: valor.replace("QUESTION #1: ","").replace("QUESTION #2: ","") for clave, valor in test_dataset["questions"].items() }

    # Save cleaned datasets
    data_clean_dir = "data_clean"
    save_jsonl(train_dataset, os.path.join(data_clean_dir, "train_dataset_clean.json"))
    save_jsonl(val_dataset, os.path.join(data_clean_dir, "validation_dataset_clean.json"))
    save_jsonl(test_dataset, os.path.join(data_clean_dir, "test_dataset_clean.json"))

if __name__ == "__main__":
    main()