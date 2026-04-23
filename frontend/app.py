"""
app.py — Streamlit frontend for the Agentic Hybrid RAG system.

Provides a web UI with three tabs:
  - Search:        Raw document retrieval (no generation), shows ranked results
  - Expert Answer: Full RAG pipeline with optional agents, shows answer + sources
  - System:        Backend health check and pipeline overview

Configuration (retrieval mode, agents, top-k) is managed via the sidebar.
Results are stored in session_state so they persist across Streamlit reruns
(Streamlit reruns the entire script on every user interaction).

Backend communication is via HTTP GET requests to the FastAPI server.
BACKEND_URL defaults to localhost but can be overridden via environment variable.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


st.set_page_config(
    page_title="Agentic Hybrid RAG",
    page_icon="🔎",
    layout="wide",
)

# Read backend URL from environment so the same image works in Docker or locally
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# ---------------------------------------------------------------------------
# API call functions
# Each function maps 1:1 to one backend endpoint.
# raise_for_status() converts HTTP error codes (4xx, 5xx) into Python exceptions
# so the caller can catch them and show an st.error() message.
# ---------------------------------------------------------------------------

def run_search(query: str, mode: str, limit: int) -> Dict[str, Any]:
    """Call /api/v1/search — retrieval only, no generation."""
    url = f"{BACKEND_URL}/api/v1/search"
    response = requests.get(
        url,
        params={"q": query, "mode": mode, "limit": limit},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_answer(query: str, mode: str, limit: int, use_agent: bool, use_rewriter) -> Dict[str, Any]:
    """Call /api/v1/answer — full RAG pipeline with optional agents."""
    url = f"{BACKEND_URL}/api/v1/answer"
    response = requests.get(
        url,
        params={"q": query, "mode": mode, "limit": limit, "use_agent": use_agent, "use_rewriter": use_rewriter},
        timeout=120, # longer timeout: generation can take several seconds
    )
    response.raise_for_status()
    return response.json()


def run_health() -> Dict[str, Any]:
    """Call /health — returns backend initialization state."""
    url = f"{BACKEND_URL}/health"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def short_text(text: Optional[str], max_len: int = 220) -> str:
    """
      Truncate text to max_len characters for display in result cards.

      Replaces newlines with spaces (keeps cards single-line) and appends
      ellipsis when truncated so the user knows more content exists.
      """

    if not text:
        return "—"
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def init_state() -> None:
    """
       Initialize session_state keys on first run.

       Streamlit reruns the entire script on every interaction, so session_state
       is used to persist data between reruns. Keys are only set if not already
       present — this avoids overwriting results on subsequent reruns.
       """

    defaults = {
        "last_search_payload": None,
        "last_search_latency": None,
        "last_answer_payload": None,
        "last_answer_latency": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results() -> None:
    """Reset all cached results in session_state."""
    st.session_state["last_search_payload"] = None
    st.session_state["last_search_latency"] = None
    st.session_state["last_answer_payload"] = None
    st.session_state["last_answer_latency"] = None

# ---------------------------------------------------------------------------
# Rendering components
# ---------------------------------------------------------------------------

def render_result_card(idx: int, result: Dict[str, Any]) -> None:
    """
        Render a single retrieved document as a bordered card.

        Shows title, retrieval metadata (method, rank, score), document IDs,
        tags, and truncated question/answer previews. For hybrid results,
        also shows per-system debug scores (BM25 raw score, vector similarity,
        and which systems returned the document).

        Full text is available via an expander to avoid cluttering the view.
        """
    title = result.get("title") or "Untitled"
    score = result.get("retrieval_score")
    method = result.get("retrieval_method", "unknown")
    tags = result.get("tags", [])
    score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)

    question_text = result.get("question_text") or "—"
    answer_body = result.get("answer_body") or "—"

    with st.container(border=True):
        top_left, top_right = st.columns([5, 2])

        with top_left:
            st.markdown(f"### {idx}. {title}")

        with top_right:
            st.markdown(
                f"""
**Method:** `{method}`  
**Rank:** `{result.get('rank')}`  
**Score:** `{score_str}`
"""
            )

        meta1, meta2, meta3 = st.columns([1, 1, 2])
        with meta1:
            st.write(f"**Question ID:** {result.get('question_id')}")
        with meta2:
            st.write(f"**Answer ID:** {result.get('answer_id')}")
        with meta3:
            st.write(f"**Tags:** {', '.join(tags) if tags else '—'}")

        # Hybrid-only debug section: shows raw scores from each sub-system
        # and which systems contributed this document to the fused ranking
        if method == "hybrid":
            st.markdown("#### Hybrid debug")
            d1, d2, d3, d4 = st.columns(4)
            d1.write(f"BM25: {result.get('bm25_score')}")
            d2.write(f"Vector: {result.get('vector_score')}")
            d3.write(f"In BM25: {result.get('found_in_bm25')}")
            d4.write(f"In Vector: {result.get('found_in_vector')}")

        q_col, a_col = st.columns(2)

        with q_col:
            st.markdown("#### Question")
            st.write(short_text(question_text, 500))

        with a_col:
            st.markdown("#### Answer")
            st.write(short_text(answer_body, 700))

        # Full text hidden by default — expander keeps the page scannable
        with st.expander("Show full source document"):
            st.markdown("##### Full Question")
            st.write(question_text)
            st.markdown("##### Full Answer")
            st.write(answer_body)

            combined = result.get("combined_text")
            if combined:
                st.markdown("##### Combined Text")
                st.text(combined)


def render_search_results(payload: Dict[str, Any], latency: float) -> None:
    """
       Render the response from /api/v1/search.

       Shows summary metrics (mode, count, limit, latency) and renders
       one result card per retrieved document.
       """

    results = payload.get("data", [])
    meta = payload.get("meta", {})

    st.markdown("## Retrieved Documents")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mode", str(meta.get("mode", "—")).upper())
    m2.metric("Retrieved", len(results))
    m3.metric("Limit", meta.get("limit", "—"))
    m4.metric("Latency", f"{latency:.2f}s")

    st.caption(f"Query: {meta.get('query', '—')}")

    if not results:
        st.info("No documents found.")
        return

    for i, result in enumerate(results, start=1):
        render_result_card(i, result)


def render_answer_results(payload: Dict[str, Any], latency: float) -> None:
    """
       Render the response from /api/v1/answer.

       Shows agent metadata (which agents ran, selected mode), the generated
       answer, and source documents used for grounding.

       If the Query Rewriter was used and actually changed the query,
       shows the original → rewritten transformation so the user can see
       what was sent to retrieval.
       """
    meta = payload.get("meta", {})
    data = payload.get("data", {})
    answer = data.get("answer", "")
    sources = data.get("sources", [])

    # Agent metadata lives in meta (set by the backend router)
    agent_used = meta.get("agent_used", False)
    final_mode = meta.get("mode", "—")

    rewritten = meta.get("rewritten_query", "")
    rewriter_used = meta.get("rewriter_used", False)

    st.markdown("## Expert Answer")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mode", final_mode)
    m2.metric("Agent used", "YES" if agent_used else "NO")
    m3.metric("Sources used", len(sources))
    m4.metric("Limit", meta.get("limit", "—"))
    m5.metric("Latency", f"{latency:.2f}s")

    # Show rewrite only if rewriter ran AND actually changed the query
    if rewriter_used and rewritten != meta.get("query"):
        st.caption(f"Original: {meta.get('query')}  →  Rewritten: **{rewritten}**")
    else:
        st.caption(f"Query: {meta.get('query', '—')}")

    with st.container(border=True):
        st.markdown("### Generated Answer")
        st.write(answer or "No answer returned.")

    st.markdown("## Supporting Sources")

    if not sources:
        st.info("No supporting sources returned.")
        return

    for i, result in enumerate(sources, start=1):
        render_result_card(i, result)

# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    st.title("🔎 Agentic Hybrid RAG")
    st.caption("Search technical documents or generate a grounded expert answer over StackOverflow Q&A.")

    # Sidebar: all configuration controls
    # mode is passed to the backend but ignored when use_agent=True
    with st.sidebar:
        st.header("Configuration")
        mode = st.selectbox("Retrieval mode", ["bm25", "vector", "hybrid"], index=2)
        use_agent = st.checkbox("Use Agent Router", value=False)
        use_rewriter = st.checkbox("Use Query Rewriter", value=False)

        limit = st.slider("Top-k documents", min_value=1, max_value=10, value=4)
        st.markdown("---")
        st.write(f"**Backend:** `{BACKEND_URL}`")
        if st.button("Clear results", use_container_width=True): # force immediate UI refresh after clearing
            clear_results()
            st.rerun()

        st.markdown("---")
        st.markdown("**Example queries**")
        st.caption("python list")
        st.caption("difference between tuple and list")
        st.caption("flatten nested list python")
        st.caption("how to use generators in python")

    query = st.text_input(
        "Enter your technical question",
        placeholder="e.g. How to use python generators?",
    )

    tab_search, tab_answer, tab_system = st.tabs(["Search", "Expert Answer", "System"])

    # --- Tab 1: Raw retrieval ---
    with tab_search:
        run_retrieval = st.button("Retrieve documents", use_container_width=True)

        if run_retrieval:
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                try:
                    start = time.time()
                    payload = run_search(query=query.strip(), mode=mode, limit=limit)
                    latency = time.time() - start

                    st.session_state["last_search_payload"] = payload
                    st.session_state["last_search_latency"] = latency
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")

        if st.session_state["last_search_payload"] is not None:
            render_search_results(
                st.session_state["last_search_payload"],
                st.session_state["last_search_latency"],
            )

    # --- Tab 2: Full RAG (retrieval + generation + agents) ---
    with tab_answer:
        run_generation = st.button("Generate expert answer", use_container_width=True)

        if run_generation:
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                try:
                    start = time.time()
                    payload = run_answer(query=query.strip(), mode=mode, limit=limit, use_agent=use_agent, use_rewriter=use_rewriter)
                    latency = time.time() - start

                    st.session_state["last_answer_payload"] = payload
                    st.session_state["last_answer_latency"] = latency
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        if st.session_state["last_answer_payload"] is not None:
            render_answer_results(
                st.session_state["last_answer_payload"],
                st.session_state["last_answer_latency"],
            )
            
    # --- Tab 3: System health and pipeline overview ---
    with tab_system:
        st.markdown("## System Status")
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("Check backend health", use_container_width=True):
                try:
                    payload = run_health()
                    st.success("Backend is healthy.")
                    st.json(payload)
                except Exception as e:
                    st.error(f"Health check failed: {e}")

        with c2:
            st.info(
                "Current frontend supports two modes:\n"
                "- Retrieve documents\n"
                "- Generate expert answer from retrieved sources"
            )

        st.markdown("## Current Flow")
        st.code(
            "query -> retrieval (bm25/vector/hybrid) -> top-k docs -> LLM answer -> supporting sources"
        )


if __name__ == "__main__":
    main()