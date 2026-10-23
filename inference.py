import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "./models/cuad_qlora_adapter"  # Ensure folder name matches local setup

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Attaching QLoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def extract_clause(question: str, context: str) -> str:
    prompt = (
        f"<|system|>\nYou are an expert legal assistant specializing in contract clause extraction.\n"
        f"<|user|>\nExtract the clause corresponding to the following category/question from the context:\n"
        f"Question: {question}\n\nContext:\n{context}\n"
        f"<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    sample_context = """
    This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, 
    without giving effect to any principles of conflicts of law.
    """
    sample_question = "Governing Law"
    
    print("\nRunning inference sample...")
    result = extract_clause(sample_question, sample_context)
    print("\n--- Extracted Clause ---")
    print(result)