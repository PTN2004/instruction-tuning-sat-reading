from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from ..model.base_model import load_base_model, apply_lora
from ..utils.config_loader import load_config

def train(train_dataset, eval_dataset):
    model, tokenizer = load_base_model()
    model = apply_lora(model)

    training_config = load_config("training_config")

    training_args = TrainingArguments(
        **{k: v for k, v in training_config.items() if k not in ["output_dir"]},
        output_dir=training_config["output_dir"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(f"{training_config['output_dir']}/peft_adapter")
    tokenizer.save_pretrained(training_config["output_dir"])
