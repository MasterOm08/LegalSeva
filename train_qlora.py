import torch
from dotenv import load_dotenv
load_dotenv()  # Reads .env and sets HF_TOKEN in the environment
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "mistralai/Mistral-7B-v0.1"  # Or "mistralai/Mistral-7B-Instruct-v0.2"

def train():
    # 1. Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Base Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Dataset
    dataset = load_dataset("json", data_files={"train": "data/cuad_train.json"})["train"]

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results_cuad_qlora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        bf16=True,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none"
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config
    )

    print("Starting QLoRA training on CUAD dataset...")
    trainer.train()

    # Save fine-tuned adapter weights
    model.save_pretrained("./cuad_qlora_adapter")
    tokenizer.save_pretrained("./cuad_qlora_adapter")
    print("Training complete! Adapter saved to ./cuad_qlora_adapter")

if __name__ == "__main__":
    train()