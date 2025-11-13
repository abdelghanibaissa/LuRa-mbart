from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base = "facebook/mbart-large-50-many-to-many-mmt"
peft_path = "lora_mbart_dialect"   # local folder

# Load base model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForSeq2SeqLM.from_pretrained(base)

# ✅ Load local LoRA weights
model = PeftModel.from_pretrained(model, peft_path, is_local=True)

# Test input
text = "شنو كتدير؟"
inputs = tokenizer(text, return_tensors="pt")

# Generate
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
