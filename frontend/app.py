import time
from typing import Any, Dict, List

import requests
import streamlit as st


st.set_page_config(
    page_title="Agentic Hybrid RAG",
    page_icon="🔎",
    layout="wide",
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def run_search(backend_base_url: str, query: str, mode: str, limit: int) -> Dict[str, Any]:
    url = f"{backend_base_url.rstrip('/')}/api/v1/search"
    params = {
        "q": query,
        "mode": mode,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def run_healthcheck(backend_base_url: str) -> Dict[str, Any]:
    url = f"{backend_base_url.rstrip('/')}/health"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def short_text(text: str | None, max_len: int = 280) -> str:
    if not text:
        return "—"
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def render_result_card(idx: int, result: Dict[str, Any]) -> None:
    title = result.get("title") or "Untitled result"
    score = result.get("retrieval_score")
    method = result.get("retrieval_method", "unknown")
    tags = result.get("tags", [])
    question_id = result.get("question_id")
    answer_id = result.get("answer_id")
    rank = result.get("rank")

    bm25_score = result.get("bm25_score")
    vector_score = result.get("vector_score")
    found_in_bm25 = result.get("found_in_bm25")
    found_in_vector = result.get("found_in_vector")

    with st.container(border=True):
        top_left, top_right = st.columns([5, 2])

        with top_left:
            st.markdown(f"### {idx}. {title}")

        with top_right:
            if isinstance(score, (int, float)):
                score_str = f"{score:.6f}"
            else:
                score_str = str(score)
            st.markdown(
                f"""
**Method:** `{method}`  
**Rank:** `{rank}`  
**Score:** `{score_str}` """ )

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.write(f"**Question ID:** {question_id}")
            st.write(f"**Answer ID:** {answer_id}")
        with meta_col2:
            st.write(f"**Tags:** {', '.join(tags) if tags else '—'}")

        if method == "hybrid":
            debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)
            with debug_col1:
                st.write(f"**BM25 score:** {bm25_score if bm25_score is not None else '—'}")
            with debug_col2:
                st.write(f"**Vector score:** {vector_score if vector_score is not None else '—'}")
            with debug_col3:
                st.write(f"**In BM25:** {found_in_bm25 if found_in_bm25 is not None else '—'}")
            with debug_col4:
                st.write(f"**In Vector:** {found_in_vector if found_in_vector is not None else '—'}")

        q_col, a_col = st.columns(2)

        with q_col:
            st.markdown("#### Question")
            st.write(result.get("question_text") or "—")

        with a_col:
            st.markdown("#### Answer")
            st.write(result.get("answer_body") or "—")

        with st.expander("Show combined document"):
            st.text(result.get("combined_text") or "—")


def main() -> None:
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {}
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = "hybrid"

    st.title("🔎 Agentic Hybrid RAG")
    st.caption("Explore BM25, vector, and hybrid retrieval over StackOverflow Q&A.")

    with st.sidebar:
        st.header("Configuration")

        backend_base_url = st.text_input(
            "Backend URL",
            value=DEFAULT_BACKEND_URL,
            help="Local backend: http://127.0.0.1:8000",
        )

        mode = st.selectbox(
            "Retrieval mode",
            options=["bm25", "vector", "hybrid"],
            index=2,
        )

        limit = st.slider("Results limit", min_value=1, max_value=20, value=5)

        st.markdown("---")

        if st.button("Check backend health", use_container_width=True):
            try:
                health = run_healthcheck(backend_base_url)
                st.success(f"Backend is up: {health}")
            except Exception as e:
                st.error(f"Health check failed: {e}")

        st.markdown("---")
        st.markdown(
            """
**Tips**
- Use **BM25** for exact keyword matches.
- Use **Vector** for semantic similarity.
- Use **Hybrid** for stronger overall retrieval.
"""
        )

    st.markdown("## Search")
    query = st.text_input(
        "Enter your technical question",
        value=st.session_state.last_query,
        placeholder="e.g. How to use python generators?",
    )

    search_col1, search_col2 = st.columns([1, 5])
    with search_col1:
        search_clicked = st.button("Search", use_container_width=True)
    with search_col2:
        st.info(f"Current mode: **{mode.upper()}**")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                start_time = time.time()
                payload = run_search(
                    backend_base_url=backend_base_url,
                    query=query.strip(),
                    mode=mode,
                    limit=limit,
                )
                latency = time.time() - start_time

                results = payload.get("data", [])
                meta = payload.get("meta", {})

                st.session_state.last_results = results
                st.session_state.last_meta = {
                    **meta,
                    "latency_sec": latency,
                }
                st.session_state.last_query = query.strip()
                st.session_state.last_mode = mode

            except requests.HTTPError as e:
                response_text = e.response.text if e.response is not None else str(e)
                st.error(f"Request failed: {response_text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    results: List[Dict[str, Any]] = st.session_state.last_results
    meta: Dict[str, Any] = st.session_state.last_meta

    if results:
        st.markdown("## Results")

        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Mode", str(meta.get("mode", st.session_state.last_mode)).upper())
        stat2.metric("Retrieved", meta.get("retrieved_count", len(results)))
        stat3.metric("Limit", meta.get("limit", "—"))
        stat4.metric("Latency", f"{meta.get('latency_sec', 0):.2f}s")

        st.caption(f"Query: {meta.get('query', st.session_state.last_query)}")

        st.markdown("### Top matches")
        for idx, result in enumerate(results, start=1):
            render_result_card(idx, result)

    elif st.session_state.last_query:
        st.info("No results found for the last query.")

    st.markdown("---")
    st.markdown(
        """
### What this UI shows
- **BM25**: lexical retrieval from Elasticsearch
- **Vector**: semantic retrieval from Qdrant
- **Hybrid**: fused ranking from both retrievers
"""
    )


if __name__ == "__main__":
    main()