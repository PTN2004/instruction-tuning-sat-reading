# 📚 Instruction Tuning SAT Reading

This project performs **Instruction Tuning** on LLaMA/Mistral models to improve English reading comprehension, specifically for SAT Reading passages.  
The pipeline uses **LoRA/QLoRA + PEFT** to fine-tune large language models (LLMs) with instruction-style datasets.  
The system can serve inference via **FastAPI** and be deployed using Docker.

---

## 🚀 Features
- Fine-tune LLaMA/Mistral with **LoRA/QLoRA**
- Prepare SAT Reading datasets for instruction tuning
- Deploy inference API using **FastAPI**
- Support running on:
  - **GPU CUDA** (NVIDIA)
  - **Apple Silicon** (MPS)
  - CPU-only
- Load fine-tuned LoRA checkpoints
- Docker integration for quick deployment

---

## 📂 Project Structure
```bash
instruction-tuning-sat-reading/
├── configs/                  # Config files for model, training, and dataset
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── inference_config.yaml
│   └── dataset_config.yaml
├── scripts/
│   ├── train.py               # Training script
│            
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI app
│   ├── dataset_loader/
│   │   └── load_dataset.py
│   ├── training/
│   │   └── trainer.py
│   ├── utils/
│   │   └── config_loader.py
│   ├── model_loader.py
│   └── inference.py
├── requirements.txt
├── run_app.py
└── README.md
```

---

## ⚙️ Installation


### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/instruction-tuning-sat-reading.git
cd instruction-tuning-sat-reading
```

### 2️⃣ Create virtual environment & install dependencies
**On Mac (Apple Silicon, CPU/MPS)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**On GPU CUDA**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔑 Hugging Face Token Setup
1. Go to https://huggingface.co/settings/tokens

2. Create a Fine-grained token with:

    * ✅ Read access to all repos

    * ✅ Access to gated repos

3. Add it to .env file:
```ini
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## ⚠️ Notes for Running on Mac
* Do not use bitsandbytes with load_in_8bit/load_in_4bit

* Use "mps" device to leverage Apple Silicon GPU

* If the model is too large → consider using GGUF format + llama.cpp
