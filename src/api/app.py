from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predictor import predict_text, load_peft_model

model, tokenizer = load_peft_model()

app = FastAPI(title="Instruction Tuning API")


class PromptRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate_endpoint(req: PromptRequest):
    result = predict_text(model, tokenizer, req.prompt)
    return {"response": result}


