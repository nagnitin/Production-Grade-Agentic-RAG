"""
Streamlit RAGAS Evaluation Dashboard.

WHY: Provides a premium dashboard for ML/AI engineers to schedule evaluations,
analyze RAGAS quality metrics (faithfulness, answer relevance, context recall, context precision),
and drill down into sample failures to iterate on chunks, prompts, and retrieval models.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from utils import (
    get_evaluation_runs,
    get_evaluation_run_details,
    trigger_evaluation,
)

# Page Configuration
st.set_page_config(
    page_title="RAGAS Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark glassmorphic style matching the main app
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-header">RAGAS Evaluation Control Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Telemetry, Quality Indicators, and Validation Benchmarks</div>', unsafe_allow_html=True)

# ==========================================
# Sidebar: Trigger New Evaluation Run
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart.png", width=64)
    st.markdown("### Launch Benchmark Run")
    
    dataset_choice = st.selectbox(
        "Select Golden Dataset",
        ["default", "security"],
        help="Predefined QA test suites with ground truth answers",
    )
    
    st.markdown("#### Evaluation Metrics")
    m_faithfulness = st.checkbox("Faithfulness (Hallucination scan)", value=True)
    m_relevance = st.checkbox("Answer Relevance", value=True)
    m_precision = st.checkbox("Context Precision", value=True)
    m_recall = st.checkbox("Context Recall", value=True)
    
    # Compile selected metrics
    selected_metrics = []
    if m_faithfulness:
        selected_metrics.append("faithfulness")
    if m_relevance:
        selected_metrics.append("answer_relevance")
    if m_precision:
        selected_metrics.append("context_precision")
    if m_recall:
        selected_metrics.append("context_recall")

    if st.button("🚀 Trigger Evaluation Run", use_container_width=True):
        if not selected_metrics:
            st.error("Please select at least one metric!")
        else:
            with st.spinner("Queueing evaluation pipeline..."):
                res = trigger_evaluation(dataset_choice, selected_metrics)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"Run triggered! ID: {res.get('run_id')[:8]}...")
                    st.toast("Evaluation queued successfully!")
                    st.rerun()

# ==========================================
# Main Dashboard Content
# ==========================================
runs = get_evaluation_runs()

if not runs:
    st.info("No evaluation runs detected. Trigger a new run from the sidebar configuration panel to populate the dashboard.")
else:
    # 1. Runs Table Overview
    st.markdown("### Past Run History")
    
    # Build dataframe for summary representation
    run_records = []
    for run in runs:
        metrics_dict = run.get("metrics", {})
        run_records.append({
            "Run ID": run.get("run_id"),
            "Dataset": run.get("dataset_name"),
            "Status": run.get("status"),
            "Samples": run.get("num_samples", 0),
            "Duration (s)": run.get("duration_seconds"),
            "Faithfulness": f"{metrics_dict.get('faithfulness', 0.0):.2f}" if "faithfulness" in metrics_dict else "-",
            "Relevancy": f"{metrics_dict.get('answer_relevance', 0.0):.2f}" if "answer_relevance" in metrics_dict else "-",
            "Precision": f"{metrics_dict.get('context_precision', 0.0):.2f}" if "context_precision" in metrics_dict else "-",
            "Recall": f"{metrics_dict.get('context_recall', 0.0):.2f}" if "context_recall" in metrics_dict else "-",
        })
    
    df_runs = pd.DataFrame(run_records)
    st.dataframe(df_runs, use_container_width=True, hide_index=True)

    # 2. Detailed Drill-down Section
    st.markdown("---")
    st.markdown("### Run Analyzer & Sample Inspector")
    
    # Allow user to select a run to inspect
    run_options = {f"{r['dataset_name']} - {r['run_id'][:8]} ({r['status']})": r["run_id"] for r in runs}
    selected_run_label = st.selectbox("Select Evaluation Run to Analyze", list(run_options.keys()))
    selected_run_id = run_options[selected_run_label]
    
    with st.spinner("Fetching run telemetry..."):
        run_details = get_evaluation_run_details(selected_run_id)
        
    if "error" in run_details:
        st.error(run_details["error"])
    else:
        status = run_details.get("status", "unknown")
        
        if status == "running" or status == "pending":
            st.info(f"⏳ This run is currently in state '{status}'. Please wait a minute and refresh the page to view final calculated scores.")
        elif status == "failed":
            st.error("❌ This evaluation run failed.")
            st.code(run_details.get("error", "Unknown processing error"))
        elif status == "completed":
            # Display run-level aggregate metric cards
            metrics_summary = run_details.get("metrics", {})
            
            # Form grid layout for metrics
            cols = st.columns(4)
            labels = ["Faithfulness", "Answer Relevance", "Context Precision", "Context Recall"]
            metric_keys = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
            
            for col, label, key in zip(cols, labels, metric_keys):
                with col:
                    val = metrics_summary.get(key)
                    val_str = f"{val:.2f}" if val is not None else "N/A"
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{val_str}</div><div class="metric-label">{label}</div></div>',
                        unsafe_allow_html=True,
                    )
            
            # 3. Sample-by-sample analysis
            st.markdown("#### Sample Outcomes")
            results = run_details.get("results", [])
            
            if not results:
                st.info("No detailed records found for this run.")
            else:
                # Render questions inside list
                sample_options = {f"Q: {r['question'][:60]}...": idx for idx, r in enumerate(results)}
                selected_sample_label = st.selectbox("Inspect Individual Sample Details", list(sample_options.keys()))
                selected_sample_idx = sample_options[selected_sample_label]
                sample_data = results[selected_sample_idx]
                
                # Render comparative blocks
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown("**User Question**")
                    st.info(sample_data.get("question"))
                    
                    st.markdown("**Ground Truth Answer**")
                    st.success(sample_data.get("ground_truth", "N/A"))
                    
                with c_col2:
                    st.markdown("**Generated Bot Answer**")
                    st.warning(sample_data.get("generated_answer"))
                    
                    # Display sample level metrics
                    st.markdown("**Sample Scorecard**")
                    s_metrics = sample_data.get("metrics", {})
                    s_metric_str = " | ".join([f"{k}: **{v:.2f}**" for k, v in s_metrics.items()])
                    st.markdown(s_metric_str or "*No metrics computed*", unsafe_allow_html=True)
                
                # Context listing
                st.markdown("**Retrieved Document Snippets / Contexts**")
                contexts = sample_data.get("contexts", [])
                if not contexts:
                    st.info("No documents retrieved for this sample.")
                else:
                    for idx, ctx in enumerate(contexts):
                        with st.expander(f"Context Chunk #{idx+1}"):
                            st.write(ctx)
