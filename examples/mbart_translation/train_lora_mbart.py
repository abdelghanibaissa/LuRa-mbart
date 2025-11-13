from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "./lora_mbart_results"
SAVE_PATH = "lora_mbart_dialect"

# --- 1. Load Model, Tokenizer, and Apply LoRA ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="ar_AR", tgt_lang="ar_AR")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# --- 2. Load and Preprocess Data ---
dataset = load_dataset("csv", data_files={"train": "dataset/train.csv", "test": "dataset/test.csv"})

def preprocess(examples):
    # Tokenize source
    model_inputs = tokenizer(examples["source"], truncation=True, max_length=128, padding=False)
    # Tokenize target (labels)
    labels = tokenizer(text_target=examples["target"], truncation=True, max_length=128, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = dataset.map(preprocess, batched=True)

# --- 3. Setup Trainer ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-3,
    num_train_epochs=3,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# --- 4. Train and Save ---
trainer.train()
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)