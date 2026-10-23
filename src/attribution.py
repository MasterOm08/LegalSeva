# src/attribution.py
import re

try:
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Download modern and legacy NLTK tokenizers automatically
    for resource in ['punkt_tab', 'punkt']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
            
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False


class SentenceAttributionModule:
    def __init__(self):
        pass

    def split_sentences(self, text: str) -> list[str]:
        """Splits text block into individual clean sentences."""
        if HAS_LIBS:
            try:
                return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 10]
            except Exception:
                pass
        # Regex fallback if NLTK is unavailable or errors
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def get_top_k_attributions(self, source_contract_text: str, summary_sentence: str, k: int = 3) -> list[dict]:
        """
        Calculates TF-IDF cosine similarity scores between summary sentences 
        and source contract clauses (OBJ-5).
        """
        source_sentences = self.split_sentences(source_contract_text)
        if not source_sentences:
            return []

        attributions = []

        if HAS_LIBS:
            corpus = source_sentences + [summary_sentence]
            vectorizer = TfidfVectorizer(stop_words='english').fit(corpus)
            
            source_vectors = vectorizer.transform(source_sentences)
            summary_vector = vectorizer.transform([summary_sentence])

            similarities = cosine_similarity(summary_vector, source_vectors)[0]
            ranked_indices = similarities.argsort()[::-1][:k]

            for idx in ranked_indices:
                attributions.append({
                    "source_sentence_index": int(idx),
                    "source_sentence": source_sentences[idx],
                    "attribution_score": round(float(similarities[idx]), 4)
                })
        else:
            summary_words = set(summary_sentence.lower().split())
            scores = []
            for idx, sent in enumerate(source_sentences):
                sent_words = set(sent.lower().split())
                overlap = len(summary_words.intersection(sent_words))
                score = round(overlap / max(len(summary_words), 1), 4)
                scores.append((idx, sent, score))
            
            scores.sort(key=lambda x: x[2], reverse=True)
            for idx, sent, score in scores[:k]:
                attributions.append({
                    "source_sentence_index": idx,
                    "source_sentence": sent,
                    "attribution_score": score
                })

        return attributions


if __name__ == "__main__":
    contract_sample = (
        "The Vendor shall not be liable for any indirect, special, or consequential damages arising out of this Agreement. "
        "Either party may terminate this agreement upon 30 days written notice. "
        "The Client agrees to indemnify, defend, and hold harmless the Vendor against all third-party legal claims, losses, and liabilities. "
        "Governing law shall be the courts of Mumbai, Maharashtra."
    )
    test_summary = "The company must hold the vendor harmless against third-party lawsuits and legal liabilities."

    attr_engine = SentenceAttributionModule()
    results = attr_engine.get_top_k_attributions(contract_sample, test_summary, k=3)
    
    print("\n--- OBJ-5 Attribution Analysis ---")
    for r in results:
        print(f"Score: {r['attribution_score']} | Source: {r['source_sentence']}")