import json
import re
from dotenv import load_dotenv
load_dotenv()  # Reads .env and sets HF_TOKEN in the environment
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = re.split(r'\W+', prediction.strip().lower())
    gt_tokens = re.split(r'\W+', ground_truth.strip().lower())
    
    pred_tokens = [t for t in pred_tokens if t]
    gt_tokens = [t for t in gt_tokens if t]

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate():
    base_model_id = "mistralai/Mistral-7B-v0.1"
    adapter_path = "./cuad_qlora_adapter"

    print("Loading fine-tuned model for evaluation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_samples = load_dataset("json", data_files={"test": "data/cuad_test.json"})["test"]

    total_em = 0.0
    total_f1 = 0.0
    count = 0

    print("Evaluating held-out CUAD test partition...")
    for sample in tqdm(test_samples.select(range(min(500, len(test_samples))))):  # Sample set for rapid eval
        prompt_text = sample['text'].split("<|assistant|>\n")[0] + "<|assistant|>\n"
        ground_truth = sample['text'].split("<|assistant|>\n")[1]

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
        
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        em = compute_exact_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)

        total_em += em
        total_f1 += f1
        count += 1

    avg_em = total_em / count
    avg_f1 = total_f1 / count

    print("\n--- OBJ-1 Evaluation Results ---")
    print(f"Exact Match (EM): {avg_em:.4f}  (Target: >= 0.45)")[cite: 10]
    print(f"Clause Span F1:   {avg_f1:.4f}  (Target: >= 0.65)")[cite: 10]

    if avg_f1 >= 0.65 and avg_em >= 0.45:
        print("✅ OBJ-1 PASSED: Model satisfies CUAD benchmark targets.")[cite: 10]
    else:
        print("⚠️ OBJ-1 IN PROGRESS: Adjust epochs/learning rate to reach F1 >= 0.65.")[cite: 10]

if __name__ == "__main__":
    evaluate()