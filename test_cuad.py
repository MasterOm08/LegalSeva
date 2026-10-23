import pandas as pd
import requests
import time

ANALYZE_URL = "http://127.0.0.1:8000/api/v1/analyze"
CSV_PATH = "CUAD_v1/master_clauses.csv"

# Load CUAD master clauses CSV
df = pd.read_csv(CSV_PATH)

# Inspect columns and pick active clauses
# CUAD columns typically include 'Filename' and individual clause category columns
print("CSV Columns:", df.columns.tolist()[:10])

# Sample test cases across standard CUAD categories
sample_tests = [
    {
        "query": "Review termination and renewal conditions.",
        "playbook_rule": "Flag any automatic renewal clauses or agreements requiring more than 30 days termination notice."
    },
    {
        "query": "Review limitation of liability and indemnification.",
        "playbook_rule": "The company must not agree to uncapped liability or broad indemnification clauses."
    },
    {
        "query": "Review governing law and jurisdiction.",
        "playbook_rule": "Governing law must be Delaware or New York."
    }
]

print("\nRunning RAG analysis test suite...\n")

for i, test in enumerate(sample_tests, 1):
    payload = {
        "query": test["query"],
        "playbook_rule": test["playbook_rule"]
    }
    
    start_time = time.time()
    response = requests.post(ANALYZE_URL, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        res = response.json()
        print(f"[{i}] Query: {test['query']}")
        print(f"    Execution Time: {elapsed:.2f}s")
        print(f"    Retrieved Sources: {len(res.get('sources', []))}")
        print(f"    Analysis Output snippet:\n    {res.get('analysis', '')[:200]}...")
        print("-" * 70)
    else:
        print(f"[{i}] Failed ({response.status_code}): {response.text}")