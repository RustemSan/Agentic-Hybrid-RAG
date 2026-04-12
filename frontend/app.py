import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st


st.set_page_config(
    page_title="Agentic Hybrid RAG",
    page_icon="🔎",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def run_search(query: str, mode: str, limit: int) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/api/v1/search"
    response = requests.get(
        url,
        params={"q": query, "mode": mode, "limit": limit},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_answer(query: str, mode: str, limit: int) -> Dict[str, Any]:
    url = f"{BACKEND_URL}/api/v1/answer"
    response = requests.get(
        url,
        params={"q": query, "mode": mode, "limit": limit},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def render_result_card(idx: int, result: Dict[str, Any]) -> None:
    title = result.get("title") or "Untitled"
    score = result.get("retrieval_score")
    method = result.get("retrieval_method", "unknown")
    tags = result.get("tags", [])
    score_str = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)

    with st.container(border=True):
        st.markdown(f"### {idx}. {title}")
        st.markdown(
            f"""
**Method:** `{method}`  
**Rank:** `{result.get('rank')}`  
**Score:** `{score_str}`
"""
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Question ID:** {result.get('question_id')}")
            st.write(f"**Tags:** {', '.join(tags) if tags else '—'}")
            st.markdown("#### Question")
            st.write(result.get("question_text") or "—")

        with col2:
            st.write(f"**Answer ID:** {result.get('answer_id')}")
            st.markdown("#### Answer")
            st.write(result.get("answer_body") or "—")

        if method == "hybrid":
            st.markdown("#### Hybrid debug")
            d1, d2, d3, d4 = st.columns(4)
            d1.write(f"BM25: {result.get('bm25_score')}")
            d2.write(f"Vector: {result.get('vector_score')}")
            d3.write(f"In BM25: {result.get('found_in_bm25')}")
            d4.write(f"In Vector: {result.get('found_in_vector')}")


def main() -> None:
    st.title("🔎 Agentic Hybrid RAG")
    st.caption("Retrieve documents and generate grounded answers over StackOverflow Q&A.")

    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox("Retrieval mode", ["bm25", "vector", "hybrid"], index=2)
        limit = st.slider("Top-k documents", 1, 10, 4)

    query = st.text_input(
        "Enter your technical question",
        placeholder="e.g. How to use python generators?",
    )

    c1, c2 = st.columns(2)
    retrieve_clicked = c1.button("Retrieve documents", use_container_width=True)
    answer_clicked = c2.button("Generate answer", use_container_width=True)

    if retrieve_clicked:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                start = time.time()
                payload = run_search(query=query.strip(), mode=mode, limit=limit)
                latency = time.time() - start

                results = payload.get("data", [])
                meta = payload.get("meta", {})

                st.success(
                    f"Retrieved {len(results)} docs | mode={meta.get('mode')} | latency={latency:.2f}s"
                )

                for i, result in enumerate(results, start=1):
                    render_result_card(i, result)

            except Exception as e:
                st.error(f"Retrieval failed: {e}")

    if answer_clicked:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                start = time.time()
                payload = run_answer(query=query.strip(), mode=mode, limit=limit)
                latency = time.time() - start

                meta = payload.get("meta", {})
                data = payload.get("data", {})
                answer = data.get("answer", "")
                sources = data.get("sources", [])

                st.success(
                    f"Answer generated | mode={meta.get('mode')} | sources={len(sources)} | latency={latency:.2f}s"
                )

                st.markdown("## Generated answer")
                st.write(answer)

                st.markdown("## Supporting sources")
                for i, result in enumerate(sources, start=1):
                    render_result_card(i, result)

            except Exception as e:
                st.error(f"Generation failed: {e}")


if __name__ == "__main__":
    main()