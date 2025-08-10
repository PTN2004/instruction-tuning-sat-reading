# src/dataset_loader/load_dataset.py
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from src.data_preprocessing.generate_prompt import generate_prompt
from src.utils.config_loader import load_config

def generate_and_tokenize_samples(dataset_split, tokenizer, max_length):
    samples = []
    for item in tqdm(dataset_split, desc="Processing dataset"):
        text = item.get("text", "")
        answer_letter = (item.get("answer") or "").strip()
        if not text:
            continue
        try:
            messages = generate_prompt(text, answer_letter)
            system_msg = next((m["content"] for m in messages if m["role"]=="system"), "")
            user_msg = next((m["content"] for m in messages if m["role"]=="user"), "")
            prompt_str = f"{system_msg}\n\n{user_msg}"

            tokenized = tokenizer(
                prompt_str,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]
            labels = input_ids.clone()
            samples.append({
                "input_ids": input_ids.numpy().tolist(),
                "attention_mask": attention_mask.numpy().tolist(),
                "labels": labels.numpy().tolist()
            })
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
    return Dataset.from_list(samples)

def prepare_train_eval():
    dataset_config = load_config("dataset_config")
    model_config = load_config("model_config")

    data = load_dataset(dataset_config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds_raw = data["train"]
    train_dataset = generate_and_tokenize_samples(train_ds_raw, tokenizer, max_length=dataset_config["max_length"])
    split_ratio = dataset_config["train_test_split_ratio"]
    split_data = train_dataset.train_test_split(test_size=split_ratio, seed=42)

    return split_data["train"], split_data["test"], tokenizer
