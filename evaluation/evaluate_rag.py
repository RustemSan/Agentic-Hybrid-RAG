import requests
import time
import pandas as pd

BACKEND_URL = "http://localhost:8000/api/v1/answer"

QUESTIONS = [
    "Are tuples mutable or immutable in Python?",
    "How does the Python interpreter work under the hood?",
    "Why do I get ModuleNotFoundError even though the module is installed?",
    "What is the best way to parse a string in Python?",
    "How to properly open and read a binary file in Python?",
    "Which library should I use to execute SQL scripts in Python?",
    "What is the difference between deep copy and shallow copy?",
    "How do I sort a list of dictionaries by a specific key value?",
    "What is the Global Interpreter Lock (GIL) and how does it affect multithreading?",
    "How to write a custom decorator that takes arguments?",
    "What is the exact difference between __new__ and __init__ methods?",
    "How to read a massive CSV file in Pandas without running out of RAM?",
    "What do *args and **kwargs mean, and when should I use them?",
    "How to safely evaluate a string representation of a dictionary without using eval()?",
    "How can I catch multiple specific exceptions in a single try-except block?"
]

# Тестируем три разных режима
MODES = ["bm25", "vector", "hybrid"]
results = []

print(
    f"Starting FULL evaluation: {len(QUESTIONS)} questions x {len(MODES)} modes = {len(QUESTIONS) * len(MODES)} requests.\n")

for mode in MODES:
    print(f"========== TESTING MODE: {mode.upper()} ==========")
    for idx, query in enumerate(QUESTIONS, 1):
        print(f"[{idx}/{len(QUESTIONS)}] Query: {query}")

        params = {
            "q": query,
            "mode": mode,
            "limit": 5,
            "use_agent": False,
            "use_rewriter": False
        }

        start_time = time.time()

        try:
            response = requests.get(BACKEND_URL, params=params)
            response.raise_for_status()
            data = response.json()

            latency = round(time.time() - start_time, 2)
            answer = data.get("data", {}).get("answer", "No answer")
            docs = data.get("data", {}).get("sources", [])

            results.append({
                "Mode": mode.upper(),
                "Question": query,
                "Latency (sec)": latency,
                "Docs Retrieved": len(docs),
                "LLM Answer": answer,
                "Status": "OK"
            })
            print(f" -> Success. Latency: {latency}s\n")

        except requests.exceptions.RequestException as e:
            latency = round(time.time() - start_time, 2)
            results.append({
                "Mode": mode.upper(),
                "Question": query,
                "Latency (sec)": latency,
                "Docs Retrieved": 0,
                "LLM Answer": "Error",
                "Status": "Error"
            })
            print(f" -> Error\n")

        time.sleep(10)

df = pd.DataFrame(results)
output_file = "full_rag_benchmark.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Evaluation complete! Results saved to {output_file}")