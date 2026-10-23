import os
import requests
from concurrent.futures import ThreadPoolExecutor

PROCESS_URL = "http://127.0.0.1:8000/api/v1/process-contract"
TXT_FOLDER = "CUAD_v1/full_contract_txt"

def upload_single_file(file_name):
    file_path = os.path.join(TXT_FOLDER, file_name)
    with open(file_path, "rb") as f:
        files = {"file": (file_name, f, "text/plain")}
        res = requests.post(PROCESS_URL, files=files)
        return file_name, res.status_code

# Run parallel uploads for faster bulk processing
all_files = [f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")][:20]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(upload_single_file, all_files))

for fname, code in results:
    print(f"[{code}] {fname}")