import os
import json
import pandas as pd

def flatten_cuad_json(json_path=os.path.join("CUAD_v1", "CUAD_v1.json")):
    print("Loading raw CUAD_v1.json...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    rows = []
    for contract in data:
        title = contract.get("title", "")
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                q_id = qa["id"]
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                
                answers = qa.get("answers", [])
                if answers:
                    answer_text = answers[0]["text"]
                    answer_start = answers[0]["answer_start"]
                else:
                    answer_text = ""
                    answer_start = -1

                rows.append({
                    "id": q_id,
                    "title": title,
                    "context": context,
                    "question": question,
                    "is_impossible": is_impossible,
                    "answer_text": answer_text,
                    "answer_start": answer_start
                })

    df = pd.DataFrame(rows)
    print(f"Flattened dataset shape: {df.shape} (Rows x Columns)")
    
    # Save processed dataframe as CSV
    os.makedirs("data", exist_ok=True)
    output_csv = os.path.join("data", "cuad_flattened.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved processed dataset to: {output_csv}")
    
    # Print core metrics
    print("\n--- Data Preprocessing Summary ---")
    print(f"Total QA Pairs:          {len(df)}")
    print(f"Positive Answers (Found): {(df['answer_text'] != '').sum()}")
    print(f"Negative Answers (Empty): {(df['answer_text'] == '').sum()}")
    
    return df

if __name__ == "__main__":
    flatten_cuad_json()