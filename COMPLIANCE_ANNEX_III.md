# EU AI Act Annex III & Model Card Package — LegalSeva

## 1. System Overview
- **System Name:** LegalSeva (Local Legal Contract Auditor)
- **Primary Domain:** LLM-Powered Legal Document Summarisation and Risk Clause Extraction and Assessment for Contract Review
- **Base Architecture:** LLaMA-3-8B-Instruct (4-bit QLoRA adapter)
- **Safety Layer:** DeBERTa-large-mnli NLI Hallucination Verification Engine

## 2. Risk Mitigation & Guardrails
- **Automated Lawyer Routing:** Mandatory legal review triggered when hallucination rate exceeds 20.0%.
- **Source Attribution:** TF-IDF Cosine Similarity highlights top-3 supporting sentence spans for every risk claim.
- **Data Privacy:** Local-first pipeline using ChromaDB vector database; no contract data transmitted to external APIs.

## 3. Performance Metrics
- **Extraction Baseline:** Fine-tuned on Atticus CUAD (41 clause categories).
- **Summarization Quality:** ROUGE-L = 0.4119 (Exceeds benchmark threshold of 0.3000).
- **Entailment Accuracy:** Sentence-by-sentence NLI verification against source contract context.