import json
from datasets import load_dataset

def format_cuad_prompt(sample):
    context = sample['context']
    question = sample['question']
    answers = sample['answers']['text']
    
    target_text = "; ".join(answers) if answers else "No clause found."

    prompt = (
        f"<|system|>\nYou are an expert legal assistant specializing in contract clause extraction.\n"
        f"<|user|>\nExtract the clause corresponding to the following category/question from the context:\n"
        f"Question: {question}\n\nContext:\n{context}\n"
        f"<|assistant|>\n{target_text}"
    )
    return {"text": prompt}

def prepare_data():
    print("Loading CUAD dataset...")
    # Add trust_remote_code=True to allow legacy dataset script execution
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    print("Formatting prompts...")
    train_dataset = dataset["train"].map(format_cuad_prompt)
    test_dataset = dataset["test"].map(format_cuad_prompt)

    train_dataset.to_json("cuad_train.json")
    test_dataset.to_json("cuad_test.json")
    print(f"Saved {len(train_dataset)} train samples and {len(test_dataset)} test samples.")

if __name__ == "__main__":
    prepare_data()