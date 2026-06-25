"""
Main Streamlit Chat Application.

WHY: Provides an intuitive, state-of-the-art UI for users to upload documents,
interact with the Agentic RAG assistant, view citations, inspect telemetry,
and provide real-time feedback.
"""

from __future__ import annotations

import uuid
import streamlit as st

from utils import (
    clear_documents,
    get_health_status,
    get_system_metrics,
    query_rag,
    submit_feedback,
    upload_document,
)

# Page configuration
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS Injection for Dark Mode & Sleek Aesthetics
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark glassmorphic container styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Custom headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom buttons */
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}

# Header Section
st.markdown('<div class="main-header">Enterprise Agentic RAG Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cloud-Native Document Intelligence & Safety-Guarded Search</div>', unsafe_allow_html=True)

# ==========================================
# Sidebar Settings, Ingestion & Telemetry
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.markdown("### System Telemetry")
    
    # 1. Health Status
    health = get_health_status()
    if health.get("status") == "healthy":
        st.success(f"🟢 API Connected (v{health.get('version', '1.0')})")
    else:
        st.error(f"🔴 Backend Offline: {health.get('message', 'Unknown Error')}")
    
    # 2. Operational Metrics Dashboard
    st.markdown("---")
    st.markdown("#### Live Stats")
    metrics = get_system_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics.get("total_queries", 0)}</div><div class="metric-label">Queries</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics.get("avg_latency_ms", 0.0):.0f}ms</div><div class="metric-label">Latency</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics.get("cache_hit_rate", 0.0)*100:.0f}%</div><div class="metric-label">Cache Hit</div></div>',
            unsafe_allow_html=True,
        )
        tokens = metrics.get("token_usage", {}).get("total_tokens", 0)
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{tokens:,}</div><div class="metric-label">Tokens</div></div>',
            unsafe_allow_html=True,
        )

    # 3. Document Ingestion Section
    st.markdown("---")
    st.markdown("#### Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload reference documents (Max 16MB)",
        type=["pdf", "docx", "pptx", "html", "txt"],
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process & Ingest Document", use_container_width=True):
            with st.spinner("Parsing, cleaning, and indexing file..."):
                file_bytes = uploaded_file.read()
                result = upload_document(file_bytes, uploaded_file.name)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Success! {result.get('message')}")
                    st.toast(f"Ingested {uploaded_file.name} successfully!")

    # 4. Session Controls
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.feedback_submitted = {}
        st.toast("Chat reset successfully!")
        st.rerun()

    if st.button("🗑️ Clear Ingested Documents", use_container_width=True):
        with st.spinner("Clearing knowledge base..."):
            result = clear_documents()
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Successfully cleared all ingested documents!")
                st.toast("Knowledge base reset!")
                st.rerun()

# ==========================================
# Main Conversation Area
# ==========================================
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Citations if available
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 View Document Citations"):
                for cite in msg["citations"]:
                    source = cite.get("source", "Unknown Document")
                    page = f" (Page {cite['page']})" if cite.get("page") else ""
                    score = cite.get("relevance_score", 0.0)
                    
                    st.markdown(
                        f"**Source**: `{source}`{page} | **Relevance**: `{score:.2f}`"
                    )
                    st.info(cite.get("content_snippet", ""))
        
        # Display Response Metadata
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            st.markdown(
                f"<small style='color: #64748b;'>Confidence: **{meta.get('confidence', 0.0)*100:.1f}%** | Intent: **{meta.get('intent', 'unknown')}**</small>",
                unsafe_allow_html=True,
            )
            
            # Simple Feedback Widgets
            msg_id = msg.get("id", f"msg_{i}")
            if msg_id not in st.session_state.feedback_submitted:
                col_f1, col_f2 = st.columns([0.1, 0.9])
                with col_f1:
                    if st.button("👍", key=f"up_{msg_id}"):
                        submit_feedback(st.session_state.session_id, 5, "Thumbs up from UI", msg_id)
                        st.session_state.feedback_submitted[msg_id] = True
                        st.toast("Feedback recorded!")
                        st.rerun()
                with col_f2:
                    if st.button("👎", key=f"down_{msg_id}"):
                        submit_feedback(st.session_state.session_id, 1, "Thumbs down from UI", msg_id)
                        st.session_state.feedback_submitted[msg_id] = True
                        st.toast("Feedback recorded!")
                        st.rerun()

# User Input Box
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Agent planning and retrieving documents..."):
            response = query_rag(prompt, st.session_state.session_id)
            
            if "error" in response:
                message_placeholder.error(response["error"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Failed to get response: {response['error']}"}
                )
            else:
                answer = response.get("answer", "")
                citations = response.get("citations", [])
                confidence = response.get("confidence", 0.0)
                intent = response.get("intent", "query")
                
                message_placeholder.markdown(answer)
                
                # Render Citations in UI
                if citations:
                    with st.expander("📚 View Document Citations"):
                        for cite in citations:
                            # Handle dictionary or object output
                            src_name = getattr(cite, "source", None) or cite.get("source", "Unknown Document")
                            pg = getattr(cite, "page", None) or cite.get("page")
                            snippet = getattr(cite, "content_snippet", None) or cite.get("content_snippet", "")
                            rel_score = getattr(cite, "relevance_score", None) or cite.get("relevance_score", 0.0)
                            
                            page_text = f" (Page {pg})" if pg else ""
                            st.markdown(
                                f"**Source**: `{src_name}`{page_text} | **Relevance**: `{rel_score:.2f}`"
                            )
                            st.info(snippet)

                # Save assistant message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "metadata": {
                            "confidence": confidence,
                            "intent": intent,
                        },
                        "id": f"msg_{len(st.session_state.messages)}",
                    }
                )
                st.rerun()
