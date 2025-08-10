import os
from dotenv import load_dotenv
from huggingface_hub import login
from src.utils.config_loader import load_config
from src.dataset_loader.load_dataset import prepare_train_eval
from src.training.trainer import train

load_dotenv()
print(os.getenv("HUGGINGFACE_TOKEN"))
login(token=os.getenv("HUGGINGFACE_TOKEN"))

if __name__ == "__main__":
    print("===== TRAINING START =====")
    model_config = load_config("model_config")
    training_config = load_config("training_config")
    dataset_config = load_config("dataset_config")

    print("[INFO] Preparing dataset...")
    train_dataset, eval_dataset, tokenizer = prepare_train_eval()

    print(f"[INFO] Training model: {model_config['base_model']}")
    train(train_dataset, eval_dataset)

    print("===== TRAINING COMPLETE =====")
