import streamlit as st
import requests
import time

st.set_page_config(page_title="Agentic RAG Explorer", layout="wide")

st.title("🔍 Agentic Hybrid RAG")

st.sidebar.header("Settings")
limit = st.sidebar.slider("Results Limit", 1, 20, 5)

query = st.text_input("Enter your technical question:", placeholder="e.g. How to use python generators?")

col1, col2, col3 = st.columns(3)

mode = None
if col1.button("🎯 BM25 Search (Flow A)"):
    mode = "bm25"
if col2.button("🤖 Agentic RAG (Flow B)"):
    mode = "agent"
if col3.button("🧬 Hybrid RAG (Flow C)"):
    mode = "hybrid"

if mode and query:
    start_time = time.time()

    try:
        backend_url = f"http://so-backend:8000/api/v1/search"
        params = {"q": query, "limit": limit}

        with st.spinner(f"Running {mode.upper()}..."):
            response = requests.get(backend_url, params=params)

        end_time = time.time()
        latency = end_time - start_time

        if response.status_code == 200:
            full_response = response.json()

            search_results = full_response.get("data", [])

            st.success(f"Done in {latency:.3f} seconds. Found {full_response['meta']['retrieved_count']} docs.")

            if not search_results:
                st.info("No results found.")
            else:
                for i, res in enumerate(search_results):
                    title = res.get('title', 'No Title')
                    score = res.get('retrieval_score', 0)

                    with st.expander(f"Result {i + 1}: {title} (Score: {score:.2f})"):
                        st.write(f"**Question ID:** {res.get('question_id')}")
                        st.write(f"**Tags:** {', '.join(res.get('tags', []))}")
                        st.markdown(res.get('answer_body', ''))

    except Exception as e:
        st.error(f"Connection failed: {e}")

elif mode and not query:
    st.warning("Please enter a query first.")