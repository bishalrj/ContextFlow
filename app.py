from __future__ import annotations

import io
import os
import sys
import time
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union, Generator
import streamlit as st
from dotenv import load_dotenv

# Safely resolve project root and append src/ to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

try:
    from src.database import InMemoryDatabaseEngine
    from src.backend import OrchestrationBrain
except ImportError:
    from database import InMemoryDatabaseEngine
    from backend import OrchestrationBrain

# Page configuration
st.set_page_config(
    page_title="ContextFlow | AI Document Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode & Glassmorphism Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    [data-testid="stChatMessageAvatar"] {
        display: none !important;
    }
    [data-testid="stChatMessageContent"] {
        padding-left: 0px !important;
        margin-left: 0px !important;
    }
    .header-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.7) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(55, 65, 81, 0.6);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .stat-box {
        background-color: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .source-item {
        background: rgba(22, 27, 34, 0.6);
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(55, 65, 81, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Strict Telemetry Logging Engine
TELEMETRY_FILE = os.path.join(ROOT_DIR, "telemetry_logs.csv")


def log_telemetry_metrics(
    query: str,
    latency_sec: float,
    retrieved_chunks: int,
    response_length: int,
    feedback: str = "None",
    msg_id: Optional[str] = None
) -> str:
    """
    Captures query audit metrics and writes structured schema directly to telemetry_logs.csv.
    """
    file_exists = os.path.exists(TELEMETRY_FILE)
    if not msg_id:
        msg_id = str(uuid.uuid4())

    try:
        with open(TELEMETRY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "message_id", "query",
                    "latency_sec", "retrieved_chunks", "response_length", "feedback"
                ])
            timestamp = datetime.now().isoformat()
            writer.writerow([
                timestamp, msg_id, query, round(latency_sec, 4),
                retrieved_chunks, response_length, feedback
            ])
    except Exception as e:
        print(f"[TELEMETRY ERROR] Failed to write telemetry: {e}")

    return msg_id


def update_telemetry_feedback(target_msg_id: str, new_feedback: str) -> None:
    """
    Retroactively updates the feedback flag for a given message transaction.
    """
    if not os.path.exists(TELEMETRY_FILE):
        return

    rows: List[List[str]] = []
    updated = False
    try:
        with open(TELEMETRY_FILE, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                rows.append(headers)
            for row in reader:
                if len(row) >= 7 and row[1] == target_msg_id:
                    row[6] = new_feedback
                    updated = True
                rows.append(row)

        if updated:
            with open(TELEMETRY_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
    except Exception as e:
        print(f"[TELEMETRY ERROR] Failed to update feedback: {e}")


# Persistent Session Isolation
if "db_engine" not in st.session_state:
    st.session_state.db_engine = InMemoryDatabaseEngine(parent_chunk_size=1500, parent_chunk_overlap=300)
if "brain" not in st.session_state:
    st.session_state.brain = OrchestrationBrain()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "parent_map" not in st.session_state:
    st.session_state.parent_map = {}
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dynamic UX Logic & Sidebar
with st.sidebar:
    st.title("System Overview")
    st.markdown("---")
    st.markdown("### Upload Study Material")
    uploaded_file = st.file_uploader(
        "Upload PDF Notes",
        type=["pdf"],
        help="Upload your study material (lecture notes, textbook, or paper) to start asking questions."
    )

    if uploaded_file and uploaded_file.name != st.session_state.processed_file_name:
        with st.spinner(f"Processing study material '{uploaded_file.name}'..."):
            try:
                file_stream = io.BytesIO(uploaded_file.getvalue())
                v_store, p_map = st.session_state.db_engine.process_and_index_stream(
                    file_stream, uploaded_file.name
                )
                st.session_state.vector_store = v_store
                st.session_state.parent_map = p_map
                st.session_state.processed_file_name = uploaded_file.name
                st.session_state.messages = []
                st.toast(f" Loaded '{uploaded_file.name}' into knowledge base!")
            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.session_state.vector_store = None
                st.session_state.parent_map = {}

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("""
    - **AI Model**: Mistral Small
    - **Embedding Model**: Mistral Embed
    - **Vector Database**: FAISS
    - **Framework**: LangChain + Streamlit
    - **Session Analytics**: Conversation Logs
    """)
    st.markdown("---")
    retrieval_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=8, value=2)
    if st.button(" Clear Knowledge Base", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Dual-Pane Header Layout
st.markdown('<div class="header-card"></div>', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])
with col1:
    st.title("ContextFlow")
    st.markdown(
        "An AI-powered document assistant that transforms PDFs into interactive conversations "
        "using Retrieval-Augmented Generation (RAG) for accurate search, summarization, and context-aware question answering."
    )
with col2:
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown('<div class="stat-box"><div class="stat-label">AI MODEL</div><div class="stat-value">Mistral Small</div></div>', unsafe_allow_html=True)
    with sc2:
        status_val = "Loaded" if st.session_state.vector_store else "Waiting"
        st.markdown(f'<div class="stat-box"><div class="stat-label">KNOWLEDGE BASE</div><div class="stat-value">{status_val}</div></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(f'<div class="stat-box"><div class="stat-label">QUESTIONS ASKED</div><div class="stat-value">{len(st.session_state.messages)//2}</div></div>', unsafe_allow_html=True)

st.divider()

# Placeholder screen if no file uploaded
if not st.session_state.vector_store or not st.session_state.parent_map:
    st.info("Upload a PDF to begin exploring its contents with AI.")
    st.stop()

# Render interactive chat history
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            latency = message.get("latency", 0.0)
            sources = message.get("sources", [])
            msg_id = message.get("msg_id", str(idx))

            if latency > 0:
                st.caption(f" Generated in **{latency:.2f}s** | Based on **{len(sources)}** relevant document passages")

            if sources:
                with st.expander(" View Source Passages"):
                    for s_idx, source in enumerate(sources, 1):
                        meta = source.metadata if hasattr(source, "metadata") else {}
                        file_name = meta.get("file_name", "Uploaded Document")
                        page_num = meta.get("page_index", meta.get("parent_block_index", "N/A"))
                        passage = source.page_content if hasattr(source, "page_content") else str(source)

                        st.markdown(f"""
                        <div class="source-item">
                            <strong>[{s_idx}] File:</strong> <code>{file_name}</code> | <strong>Section/Page:</strong> <code>{page_num}</code><br/>
                            <span style="font-size: 0.88em; color: #cbd5e1;">"{passage[:350]}..."</span>
                        </div>
                        """, unsafe_allow_html=True)

            fb_col1, fb_col2, _ = st.columns([1.5, 1.5, 9])
            current_fb = message.get("feedback")
            with fb_col1:
                up_type = "primary" if current_fb == "thumbs_up" else "secondary"
                if st.button(" Useful", key=f"up_{idx}", type=up_type, use_container_width=True):
                    message["feedback"] = "thumbs_up"
                    update_telemetry_feedback(msg_id, "thumbs_up")
                    st.toast(" Thank you for your feedback!")
                    st.rerun()
            with fb_col2:
                down_type = "primary" if current_fb == "thumbs_down" else "secondary"
                if st.button(" Inaccurate", key=f"down_{idx}", type=down_type, use_container_width=True):
                    message["feedback"] = "thumbs_down"
                    update_telemetry_feedback(msg_id, "thumbs_down")
                    st.toast(" Thank you for your feedback!")
                    st.rerun()

# User Chat Input
if prompt := st.chat_input("Ask anything about your uploaded notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        try:
            token_generator, sources = st.session_state.brain.stream_question(
                user_query=prompt,
                vector_store=st.session_state.vector_store,
                parent_map=st.session_state.parent_map,
                k=retrieval_k
            )
            answer = st.write_stream(token_generator)
        except Exception as e:
            answer = f" Streaming Error: {str(e)}"
            st.error(answer)
            sources = []

        latency = time.time() - start_time
        st.caption(f" Generated in **{latency:.2f}s** | Based on **{len(sources)}** relevant document passages")

        if sources:
            with st.expander(" View Source Passages"):
                for s_idx, source in enumerate(sources, 1):
                    meta = source.metadata if hasattr(source, "metadata") else {}
                    file_name = meta.get("file_name", "Uploaded Document")
                    page_num = meta.get("page_index", meta.get("parent_block_index", "N/A"))
                    passage = source.page_content if hasattr(source, "page_content") else str(source)

                    st.markdown(f"""
                    <div class="source-item">
                        <strong>[{s_idx}] File:</strong> <code>{file_name}</code> | <strong>Section/Page:</strong> <code>{page_num}</code><br/>
                        <span style="font-size: 0.88em; color: #cbd5e1;">"{passage[:350]}..."</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Log transaction telemetry
        msg_id = log_telemetry_metrics(
            query=prompt,
            latency_sec=latency,
            retrieved_chunks=len(sources),
            response_length=len(answer) if answer else 0,
            feedback="None"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "latency": latency,
            "sources": sources,
            "msg_id": msg_id,
            "feedback": None
        })
        st.rerun()

# Subtle Footer
st.markdown('<div class="footer">Built with Streamlit, LangChain, FAISS, and Mistral AI</div>', unsafe_allow_html=True)
