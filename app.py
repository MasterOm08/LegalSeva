import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="LegalSeva — Local Legal Contract Auditor",
    page_icon="⚖️",
    layout="wide"
)

# Mandatory Legal Disclaimer Banner
LEGAL_DISCLAIMER = "⚠️ NOT FOR LEGAL ADVICE — FOR INFORMATIONAL & AUDITING PURPOSES ONLY"
st.caption(LEGAL_DISCLAIMER)

st.title("📄 LegalSeva — Local Legal Contract Auditor")
st.markdown("---")

# Session State Initialization
if "analysis_res" not in st.session_state:
    st.session_state.analysis_res = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

col1, col2 = st.columns([1, 1], gap="large")

# --- Column 1: Document Processing ---
with col1:
    st.subheader("1. Contract Document Upload")
    uploaded_file = st.file_uploader("Upload Contract (.txt or .pdf)", type=["txt", "pdf"])

    if uploaded_file and st.button("Process & Index Contract", use_container_width=True):
        with st.spinner("Chunking document and indexing into local ChromaDB..."):
            # Dynamically pass uploaded file MIME type
            mime_type = uploaded_file.type if uploaded_file.type else "application/octet-stream"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
            
            try:
                res = requests.post("http://127.0.0.1:8000/api/v1/process-contract", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.indexed_file = uploaded_file.name
                    st.success(f"Indexed **{data.get('total_chunks_processed', 0)}** chunks from `{uploaded_file.name}` into ChromaDB!")
                else:
                    detail = res.json().get('detail', 'Indexing failed.') if res.headers.get("content-type") == "application/json" else res.text
                    st.error(f"Error {res.status_code}: {detail}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend at http://127.0.0.1:8000. Error: {e}")

# --- Column 2: Playbook Audit Config ---
with col2:
    st.subheader("2. Playbook Policy Audit")
    query = st.text_input(
        "Query Category / Clause Type",
        value="Review limitation of liability and indemnification."
    )
    rule = st.text_area(
        "Playbook Policy Rule",
        value="The company must not agree to uncapped liability or broad indemnification.",
        height=100
    )

    run_audit = st.button("Run Audit", type="primary", use_container_width=True)

    if run_audit:
        if not st.session_state.indexed_file:
            st.warning("Please upload and index a contract document in Step 1 first!")
        else:
            with st.spinner("Executing ChromaDB vector retrieval & LLM risk audit..."):
                payload = {
                    "query": query,
                    "playbook_rule": rule,
                    "document_filter": st.session_state.indexed_file
                }
                try:
                    response = requests.post("http://127.0.0.1:8000/api/v1/analyze", json=payload)
                    if response.status_code == 200:
                        st.session_state.analysis_res = response.json()
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to reach FastAPI backend: {e}")

st.markdown("---")

# --- Results Section ---
if st.session_state.analysis_res:
    res = st.session_state.analysis_res

    st.subheader("3. Audit Results & Compliance Guardrails")

    routing = res.get("lawyer_routing", {})
    metrics = res.get("hallucination_metrics", {})

    # OBJ-4 Lawyer Routing Banner Display
    if routing.get("mandatory_lawyer_review"):
        st.error("🚨 **MANDATORY LAWYER REVIEW REQUIRED** (Confidence / Consistency Threshold Breached)")
        for reason in routing.get("flag_reasons", []):
            st.write(f"- 🔴 {reason}")
    else:
        st.success("✅ **AUTOMATED PASS**: Entailment verified within safe hallucination thresholds.")

    # Analysis Text Output
    st.markdown("### LLM Risk Assessment")
    st.markdown(res.get("analysis", "No analysis content returned."))

    # OBJ-3 Guardrails Breakdown Component
    with st.expander("🔍 NLI Hallucination & Entailment Breakdown (DeBERTa-large-mnli)"):
        h_rate = metrics.get("hallucination_rate", 0.0) * 100
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="Hallucination Flag Rate", value=f"{h_rate:.1f}%")
        with m_col2:
            st.metric(label="Routing Tag", value=routing.get("tag", "N/A"))
        with m_col3:
            st.metric(label="Total Sentences Analyzed", value=len(metrics.get("sentence_results", [])))

        sentence_results = metrics.get("sentence_results", [])
        if sentence_results:
            st.markdown("#### Sentence-by-Sentence Entailment Analysis:")
            for idx, item in enumerate(sentence_results, 1):
                icon = "✅" if item.get("is_entailed") else "⚠️"
                st.write(f"{icon} **Sentence {idx}:** {item.get('sentence')}")
                st.caption(f"Status: `{item.get('top_label')}` | Entailment Score: `{item.get('entailment_score', 0.0):.4f}`")

    # Source Citations Component
    sources = res.get("sources")
    if sources:
        with st.expander("📚 Retrieved Source Chunks from ChromaDB"):
            for idx, src in enumerate(sources, 1):
                st.markdown(f"**Chunk {idx}:**")
                st.text(src)

    # Footer Disclaimer
    st.markdown("---")
    st.caption(f"📜 System Policy: {res.get('disclaimer', LEGAL_DISCLAIMER)}")