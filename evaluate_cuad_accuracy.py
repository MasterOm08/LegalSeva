import pandas as pd
import requests

ANALYZE_URL = "http://127.0.0.1:8000/api/v1/analyze"
CSV_PATH = "CUAD_v1/master_clauses.csv"

df = pd.read_csv(CSV_PATH)

# Filter for ingested contracts only
ingested_docs = [
    "2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding Agreement_ Agency Agreement.txt",
    "ABILITYINC_06_15_2020-EX-4.25-SERVICES AGREEMENT.txt"
]

test_rows = df[df["Filename"].isin(ingested_docs)]

# Map CUAD columns to playbook checks
categories_to_test = {
    "Termination For Convenience": "Flag if either party can terminate for convenience without cause.",
    "Uncapped Liability": "Flag if liability limits are uncapped or exceed $1,000,000.",
    "Governing Law": "Governing law must strictly be Delaware or New York."
}

results = []

for idx, row in test_rows.iterrows():
    filename = row["Filename"]
    for cat, rule in categories_to_test.items():
        if cat in df.columns:
            has_clause = str(row[cat]).strip() != ""
            
            payload = {
                "query": f"Analyze contract for {cat}",
                "playbook_rule": rule
            }
            
            res = requests.post(ANALYZE_URL, json=payload).json()
            analysis_text = res.get("analysis", "")
            
            results.append({
                "Filename": filename,
                "Category": cat,
                "CUAD_Ground_Truth_Present": has_clause,
                "LLM_Output_Snippet": analysis_text[:120]
            })

results_df = pd.DataFrame(results)
print(results_df.head(10))
results_df.to_csv("cuad_benchmark_results.csv", index=False)
print("\nEvaluation results saved to cuad_benchmark_results.csv")