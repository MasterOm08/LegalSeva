# verification.py
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure NLTK sentence splitter is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HallucinationChecker:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli"):
        print(f"Loading NLI model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def split_into_sentences(self, text: str) -> list[str]:
        """Splits a summary paragraph into clean individual sentences."""
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

    def calculate_hallucination_rate(self, source_clause: str, summary_text: str) -> dict:
        """
        Evaluates each summary sentence against the source clause text.
        Returns the hallucination rate and sentence-level entailment scores.
        """
        sentences = self.split_into_sentences(summary_text)
        if not sentences:
            return {"hallucination_rate": 0.0, "details": []}

        unentailed_count = 0
        sentence_details = []

        for sentence in sentences:
            # Tokenize Premise (Source Clause) and Hypothesis (Summary Sentence)
            inputs = self.tokenizer(
                source_clause, 
                sentence, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # DeBERTa-MNLI output mapping: [Contradiction (0), Neutral (1), Entailment (2)]
            probs = torch.softmax(logits, dim=-1)[0]
            entailment_prob = probs[2].item()
            
            # Flag sentence if entailment confidence is below threshold
            is_hallucinated = entailment_prob < 0.70
            if is_hallucinated:
                unentailed_count += 1

            sentence_details.append({
                "sentence": sentence,
                "entailment_prob": round(entailment_prob, 4),
                "is_hallucinated": is_hallucinated
            })

        hallucination_rate = unentailed_count / len(sentences)
        return {
            "hallucination_rate": round(hallucination_rate, 4),
            "sentence_details": sentence_details
        }

def evaluate_lawyer_routing(hallucination_rate: float, self_consistency_pass: bool) -> dict:
    """
    Lawyer-Routing Policy (OBJ-4):
    Flags summaries for human legal review if hallucination rate > 0.20 
    or self-consistency (N=3 disagreement) fails.
    """
    if hallucination_rate > 0.20 or not self_consistency_pass:
        return {
            "action": "MANDATORY_LAWYER_REVIEW",
            "tag": "High-Confidence-Needed",
            "reason": f"Flagged: Hallucination rate = {hallucination_rate:.2f} (Max threshold: 0.20) or Self-Consistency failed."
        }
    return {
        "action": "AUTO_APPROVE",
        "tag": "Verified",
        "reason": "Passed NLI entailment threshold and self-consistency check."
    }