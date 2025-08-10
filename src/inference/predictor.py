import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, PeftConfig
from ..utils.config_loader import load_config
from ..data_preprocessing.generate_prompt import generate_prompt


def load_peft_model():
    infer_cfg = load_config("inference_config")
    config = PeftConfig.from_pretrained(infer_cfg["peft_model_id"])
    base_model_name = config.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, infer_cfg["peft_model_id"])
    return model, tokenizer


def predict_text(model, tokenizer, text: str):
    infer_cfg = load_config("inference_config")
    messages = generate_prompt(text, "A")
    system_msg = next((m["content"]
                      for m in messages if m["role"] == "system"), "")
    user_msg = next((m["content"]
                    for m in messages if m["role"] == "user"), "")
    prompt = f"{system_msg}\n\n{user_msg}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=infer_cfg["max_new_tokens"],
            temperature=infer_cfg["temperature"],
            top_p=infer_cfg["top_p"],
            do_sample=infer_cfg["do_sample"],
            repetition_penalty=infer_cfg["repetition_penalty"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
