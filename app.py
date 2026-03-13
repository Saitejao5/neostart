"""
app.py — ScholarBot: AI Research & Education Assistant
OpenRouter Edition

Features:
  ✅ RAG — upload PDFs/DOCX/TXT, vector search
  ✅ Live Web Search — Serper + Tavily fallback
  ✅ Concise / Detailed response modes
  ✅ OpenRouter LLM
  ✅ Source citation pills
"""

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from config.config import (
    APP_TITLE,
    APP_ICON,
    RESPONSE_MODES,
    OPENROUTER_API_KEY,
    SERPER_API_KEY,
    TAVILY_API_KEY,
)

from models.llm import get_llm_response
from utils.rag import VectorStore, build_rag_context
from utils.web_search import web_search, format_search_results
from utils.chat import build_prompt_with_context, trim_history, build_system_prompt


# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
)


# ── Session State ───────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "ingested_files": [],
        "response_mode": "Concise",
        "use_rag": True,
        "use_web_search": False,
        "model_override": "",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:

    st.title("🎓 ScholarBot")
    st.caption("AI Research Assistant")

    st.divider()

    # ── Response Mode ───────────────────────────────────────
    st.subheader("⚡ Response Mode")

    mode = st.radio(
        "Mode",
        options=["Concise", "Detailed"],
        index=["Concise", "Detailed"].index(st.session_state.response_mode),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.session_state.response_mode = mode

    st.divider()

    # ── Features ────────────────────────────────────────────
    st.subheader("🔧 Features")

    st.session_state.use_rag = st.toggle(
        "Document RAG", value=st.session_state.use_rag
    )

    st.session_state.use_web_search = st.toggle(
        "Live Web Search", value=st.session_state.use_web_search
    )

    st.divider()

    # ── Upload Documents ───────────────────────────────────
    st.subheader("📁 Upload Study Materials")

    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("Index Documents", use_container_width=True):

            with st.spinner("Indexing documents..."):
                try:
                    vs = VectorStore()

                    temp_paths = []

                    for uf in uploaded:
                        suffix = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.read())
                            temp_paths.append(tmp.name)

                    n_chunks = vs.add_documents(temp_paths)

                    for tp in temp_paths:
                        os.unlink(tp)

                    st.session_state.vector_store = vs
                    st.session_state.ingested_files = [uf.name for uf in uploaded]

                    st.success(f"Indexed {n_chunks} chunks")

                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    if st.session_state.ingested_files:

        st.subheader("📚 Indexed Files")

        for f in st.session_state.ingested_files:
            st.caption(f)

    st.divider()

    # ── Controls ───────────────────────────────────────────
    st.subheader("🗑 Controls")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Documents", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.ingested_files = []
        st.rerun()

    st.divider()

    # ── API Status ─────────────────────────────────────────
    st.subheader("🔑 API Status")

    api_status = [
        ("OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Serper", bool(SERPER_API_KEY)),
        ("Tavily", bool(TAVILY_API_KEY)),
    ]

    for name, status in api_status:
        st.write(f"{name}: {'✅ Connected' if status else '❌ Missing'}")


# ════════════════════════════════════════════════════════════
# MAIN CHAT
# ════════════════════════════════════════════════════════════

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("Upload documents and ask research questions.")


# ── Render Chat History ────────────────────────────────────
for msg in st.session_state.messages:

    role = msg["role"]

    with st.chat_message(role):

        st.markdown(msg["content"])

        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))


# ── Chat Input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a research question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                rag_context = ""
                web_context = ""
                sources_used = []

                # ── RAG Retrieval ──────────────────────────
                if st.session_state.use_rag and st.session_state.vector_store:

                    retrieved = st.session_state.vector_store.retrieve(prompt)

                    if retrieved:

                        rag_context = build_rag_context(retrieved)

                        sources_used = list({r["source"] for r in retrieved})


                # ── Web Search ─────────────────────────────
                if st.session_state.use_web_search:

                    results = web_search(prompt)

                    if results:

                        web_context = format_search_results(results)

                        sources_used.extend([r["url"] for r in results[:2]])


                # ── Build Prompt ───────────────────────────
                enriched_prompt = build_prompt_with_context(
                    user_query=prompt,
                    rag_context=rag_context,
                    web_context=web_context,
                    response_mode=st.session_state.response_mode,
                )

                trimmed_history = trim_history(st.session_state.messages[:-1])

                llm_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in trimmed_history
                ]

                llm_messages.append({"role": "user", "content": enriched_prompt})

                system_prompt = build_system_prompt(st.session_state.response_mode)

                model = st.session_state.model_override.strip() or None

                response = get_llm_response(
                    messages=llm_messages,
                    model=model,
                    system_prompt=system_prompt,
                )

                st.markdown(response)

                if sources_used:
                    st.caption("Sources: " + ", ".join(sources_used))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources_used,
                    }
                )

            except Exception as e:

                error = f"Error: {e}"

                st.error(error)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error}
                )