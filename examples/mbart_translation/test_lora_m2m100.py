from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# --- Model Setup ---
MODEL_NAME = "facebook/m2m100_418M"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_PATH = os.path.join(SCRIPT_DIR, "lora_m2m100_dialect")  # Replace with your saved path

# Load tokenizer, base model, and apply LoRA adapter
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="ar", tgt_lang="ar_mor")  # Use "ar_mor" for Moroccan Arabic
BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
MODEL = PeftModel.from_pretrained(BASE_MODEL, LORA_PATH)

# Setup device and set model to eval mode
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = MODEL.to(DEVICE)
MODEL.eval()

def translate(text, max_length=128):
    """Tokenize, generate, and decode translation"""
    inputs = TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    test_cases = [
        "كيف حالك؟",  # Arabic: How are you?
        "أنا ذاهب إلى السوق",  # Arabic: I am going to the market
        "ماذا تفعل؟",  # Arabic: What are you doing?
        "أين أنت؟",  # Arabic: Where are you?
        "هل تحب البرمجة؟"  # Arabic: Do you like programming?
    ]
    
    print(f"--- Running Translator on {DEVICE} ---")
    for case in test_cases:
        result = translate(case)
        print(f"'{case}' → '{result}'")
    print("--- Done ---")
