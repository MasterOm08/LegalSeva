import os
import json
import zipfile
import requests

def download_and_inspect_cuad():
    json_path = os.path.join("CUAD_v1", "CUAD_v1.json")
    zip_path = "CUAD_v1.zip"
    
    # Download zip if CUAD_v1.json isn't present
    if not os.path.exists(json_path):
        if not os.path.exists(zip_path):
            print("Downloading CUAD_v1.zip from Zenodo (~18MB)...")
            url = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
        print("Extracting CUAD_v1.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    print("Loading JSON dataset into memory...")
    with open(json_path, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)
    
    contracts = cuad_data["data"]
    print("\n================ DATASET SUCCESS ================")
    print(f"Total Unique Contracts Parsed: {len(contracts)}")
    
    sample_contract = contracts[0]
    sample_paragraph = sample_contract["paragraphs"][0]
    sample_qa = sample_paragraph["qas"][0]
    
    print("\n--- First Contract Sample ---")
    print(f"Contract Title:     {sample_contract['title']}")
    print(f"Context Text Len:   {len(sample_paragraph['context'])} chars")
    print(f"Question Category:  {sample_qa['question']}")
    print(f"Sample Answer Text: {sample_qa['answers']}")

if __name__ == "__main__":
    download_and_inspect_cuad()