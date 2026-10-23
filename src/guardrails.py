import re
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIHallucinationChecker:
    """
    NLI-based Hallucination Guardrail using DeBERTa-v3-large.
    Evaluates whether generated claims are strictly entailed by retrieved contract context.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large"):
        # Optimize PyTorch CPU thread management for local execution
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            torch.set_num_threads(4)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _extract_factual_statements(self, text: str) -> List[str]:
        """
        Strips out redline suggestions, code blocks, and policy rules so NLI
        only evaluates direct claims about the contract text.
        """
        lines = text.split("\n")
        factual_lines = []
        in_redline_block = False

        for line in lines:
            cleaned = line.strip()

            # Skip empty lines, headers, and markdown rules
            if not cleaned or cleaned.startswith("#") or cleaned.startswith("---"):
                continue

            # Detect boundaries for non-factual commentary / redlines
            if any(h in cleaned.lower() for h in ["recommended redline", "redline clause", "key points of the redline"]):
                in_redline_block = True
                continue
            if any(h in cleaned.lower() for h in ["factual clause analysis", "exact quote from context", "clause name"]):
                in_redline_block = False
                continue

            # Exclude policy definition descriptions or bullet points explaining risk tiers
            if not in_redline_block:
                if not any(k in cleaned for k in ["PLAYBOOK POLICY", "LOW -", "MEDIUM -", "HIGH -"]):
                    factual_lines.append(cleaned)

        raw_factual_text = " ".join(factual_lines)
        
        # Split into individual sentence units
        sentences = [
            s.strip() for s in re.split(r'(?<=[.!?]) +', raw_factual_text)
            if len(s.strip()) > 15
        ]
        return sentences

    def evaluate_hallucination_rate(self, source_context: str, generated_text: str) -> Dict[str, Any]:
        """
        Runs sentence-by-sentence NLI evaluation against retrieved contract context.
        Returns total sentences, hallucinated count, and overall hallucination rate percentage.
        """
        sentences = self._extract_factual_statements(generated_text)

        if not sentences:
            return {
                "hallucination_rate": 0.0,
                "total_sentences": 0,
                "hallucinated_sentences": 0,
                "status": "NO_FACTUAL_CLAIMS_FOUND"
            }

        hallucinated_count = 0
        details = []

        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    source_context,
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

                # Index mapping for DeBERTa MNLI: 0 -> Contradiction, 1 -> Neutral, 2 -> Entailment
                contradiction_prob = probs[0].item()
                neutral_prob = probs[1].item()
                entailment_prob = probs[2].item()

                # Flag sentence as ungrounded/hallucinated if entailment is below threshold
                is_hallucinated = entailment_prob < 0.35 or contradiction_prob > 0.50

                if is_hallucinated:
                    hallucinated_count += 1

                details.append({
                    "sentence": sentence,
                    "entailment_score": round(entailment_prob, 4),
                    "is_hallucinated": is_hallucinated
                })

        hallucination_rate = round((hallucinated_count / len(sentences)) * 100, 2)

        return {
            "hallucination_rate": hallucination_rate,
            "total_sentences": len(sentences),
            "hallucinated_sentences": hallucinated_count,
            "details": details
        }


def evaluate_lawyer_routing(hallucination_rate: float, threshold: float = 20.0) -> Dict[str, Any]:
    """
    Lawyer-Routing Policy logic (OBJ-4).
    Determines whether an AI audit response can be delivered directly or requires human review.
    """
    requires_review = hallucination_rate > threshold

    if requires_review:
        return {
            "action": "MANDATORY_LAWYER_REVIEW",
            "requires_human": True,
            "reason": f"Hallucination rate ({hallucination_rate}%) exceeds allowed threshold ({threshold}%).",
            "status_color": "red"
        }

    return {
        "action": "AUTOMATED_PASS",
        "requires_human": False,
        "reason": f"Hallucination rate ({hallucination_rate}%) is within acceptable compliance bounds (≤ {threshold}%).",
        "status_color": "green"
    }