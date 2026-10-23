from typing import Dict, Any
from langchain_groq import ChatGroq
from src.config import settings
from src.vector_store import VectorStoreManager

class LegalRAGEngine:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY
        )

    def analyze_clause(self, query: str, playbook_rule: str) -> Dict[str, Any]:
        retrieved_chunks = self.vector_store.similarity_search(query=query, top_k=3)
        context_text = "\n\n".join([f"[Source: {c['metadata'].get('filename', 'Doc')}]\n{c['text']}" for c in retrieved_chunks])

        prompt = f"""You are an expert AI Legal Counsel reviewing contract risk against corporate playbook rules.

CORPORATE PLAYBOOK POLICY:
"{playbook_rule}"

CONTRACT CONTEXT:
{context_text}

INSTRUCTIONS:
1. Identify if the contract context complies with or violates the playbook policy.
2. Quote the exact verbatim clause text from the contract.
3. Assign a Risk Severity Level: LOW, MEDIUM, or HIGH.
4. Provide a recommended redline or standard fallback clause if high risk.

OUTPUT FORMAT:
- Assessment: [Compliant / Non-Compliant / Missing]
- Risk Level: [LOW / MEDIUM / HIGH]
- Exact Quote: "[verbatim text from context]"
- Analysis: [Detailed explanation]
- Recommended Redline: [Fallback wording]
"""

        response = self.llm.invoke(prompt)
        
        return {
            "query": query,
            "playbook_rule": playbook_rule,
            "analysis": response.content,
            "sources": [c["metadata"] for c in retrieved_chunks]
        }