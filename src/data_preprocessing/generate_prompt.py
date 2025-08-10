from typing import List, Dict
from .extract_sections import extract_sections
from ..utils.config_loader import load_config


def map_answer(text: str, letter: str) -> str:
    sections = extract_sections(text)
    for ch in sections.get("choices", []):
        if ch.startswith(f"{letter}") or ch.startswith(f"{letter})") \
           or ch.startswith(f"{letter}.") or ch.startswith(f"{letter}:"):
            return ch
    return letter  # fallback


def generate_prompt(text: str, answer_letter: str) -> List[Dict[str, str]]:

    model_config = load_config("model_config")
    system_prompt = model_config.get("system_prompt",
                                     "You are a helpful AI assistant developed by Meta. Respond safely and accurately.")

    sections = extract_sections(text)
    choices_text = "\n".join(sections.get("choices", []))

    user_content = f"""Read the passage and answer the question.

### Passage:

{sections.get('passage', '')}

### Question:

{sections.get('question', '')}

### Choices:

{choices_text}

Respond with ONLY the letter and full text of the correct answer."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": map_answer(text, answer_letter)}
    ]
    return messages
