import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from ..utils.config_loader import load_config

def load_base_model():
    model_config = load_config("model_config")
    quant_cfg = model_config["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"]
    )

    print(f"[base_model] Loading {model_config['base_model']}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def apply_lora(model):
    model_config = load_config("model_config")
    lora_cfg = model_config["lora"]

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )
    return get_peft_model(model, peft_config)
