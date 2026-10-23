# evaluate_summarization.py
from rouge_score import rouge_scorer

def evaluate_summary_batch():
    references = [
        "The client must hold harmless and indemnify the vendor against all third-party legal claims.",
        "Either party has the right to terminate the contract by providing 30 days prior written notice."
    ]
    
    generated = [
        "The client agrees to defend and indemnify the vendor from any third-party lawsuits or liabilities.",
        "Termination requires a 30-day written notification from either party."
    ]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, gen)['rougeL'].fmeasure for ref, gen in zip(references, generated)]
    avg_rouge_l = sum(scores) / len(scores)

    print("\n--- OBJ-2 Lay-Language Summarization Evaluation ---")
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
    print(f"Target Benchmark Threshold: 0.3000")
    print(f"Benchmark Result: {'✅ PASS' if avg_rouge_l >= 0.30 else '❌ FAIL'}")

if __name__ == "__main__":
    evaluate_summary_batch()