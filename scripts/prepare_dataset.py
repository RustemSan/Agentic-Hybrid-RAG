import json
import re
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup

from backend.app.db import get_connection


OUTPUT_PATH = Path("data/processed/documents.jsonl")

# Пороговые фильтры
MIN_QUESTION_LEN = 40
MIN_ANSWER_LEN = 60
MIN_QUESTION_SCORE = 0
MIN_ANSWER_SCORE = 0

# Ограничение на количество документов в итоговом датасете.
MAX_DOCUMENTS = None

def clean_html(html_text: str) -> str:
    if not html_text:
        return ""

    html_text = unescape(html_text)

    try:


        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            # грубый fallback: просто убрать теги regex'ом
            text = re.sub(r"<[^>]+>", " ", html_text)
            text = text.replace("\xa0", " ")
            text = re.sub(r"\s+", " ", text).strip()
            return text

    for tag in soup.find_all(["pre", "code"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text.strip()

def parse_tags(tag_string: str) -> list[str]:
    if not tag_string:
        return []

    tag_string = tag_string.strip("|")
    tags = tag_string.split("|")
    return [t for t in tags if t]


def build_question_text(title: str, question_body: str) -> str:
    parts = []
    if title:
        parts.append(title.strip())
    if question_body:
        parts.append(question_body.strip())
    return " ".join(parts).strip()


def build_combined_text(title: str, question_body: str, answer_body: str) -> str:
    parts = []

    if title:
        parts.append(f"TITLE:\n{title.strip()}")

    if question_body:
        parts.append(f"QUESTION:\n{question_body.strip()}")

    if answer_body:
        parts.append(f"ANSWER:\n{answer_body.strip()}")

    return "\n\n".join(parts).strip()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cur = conn.cursor()

    query = """
    SELECT
        q.id AS question_id,
        q.title,
        q.body AS question_body_html,
        q.tags,
        q.score AS question_score,
        q.view_count,
        q.answer_count,
        q.favorite_count,
        q.creation_date AS question_creation_date,

        a.id AS answer_id,
        a.body AS answer_body_html,
        a.score AS answer_score,
        a.creation_date AS answer_creation_date

    FROM posts q
    JOIN posts a
        ON a.id = q.accepted_answer_id

    WHERE q.post_type_id = 1
      AND q.accepted_answer_id IS NOT NULL
      AND q.title IS NOT NULL
      AND q.body IS NOT NULL
      AND a.body IS NOT NULL
      AND COALESCE(q.score, 0) >= %s
      AND COALESCE(a.score, 0) >= %s

    ORDER BY q.id
    """

    cur.execute(query, (MIN_QUESTION_SCORE, MIN_ANSWER_SCORE))

    written = 0
    scanned = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        while True:
            rows = cur.fetchmany(5000)
            if not rows:
                break

            for row in rows:
                scanned += 1

                (
                    question_id,
                    title,
                    question_body_html,
                    tag_string,
                    question_score,
                    view_count,
                    answer_count,
                    favorite_count,
                    question_creation_date,
                    answer_id,
                    answer_body_html,
                    answer_score,
                    answer_creation_date,
                ) = row

                title = (title or "").strip()
                question_body = clean_html(question_body_html or "")
                answer_body = clean_html(answer_body_html or "")
                tags = parse_tags(tag_string or "")

                question_text = build_question_text(title, question_body)
                combined_text = build_combined_text(title, question_body, answer_body)

                if len(question_text) < MIN_QUESTION_LEN:
                    continue

                if len(answer_body) < MIN_ANSWER_LEN:
                    continue

                doc = {
                    "question_id": question_id,
                    "answer_id": answer_id,
                    "title": title,
                    "question_body": question_body,
                    "answer_body": answer_body,
                    "question_text": question_text,
                    "combined_text": combined_text,
                    "tags": tags,
                    "question_score": question_score or 0,
                    "answer_score": answer_score or 0,
                    "view_count": view_count or 0,
                    "answer_count": answer_count or 0,
                    "favorite_count": favorite_count or 0,
                    "question_creation_date": question_creation_date.isoformat() if question_creation_date else None,
                    "answer_creation_date": answer_creation_date.isoformat() if answer_creation_date else None,
                }

                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

                if written % 10000 == 0:
                    print(f"Written {written} documents (scanned {scanned})...", flush=True)

                if MAX_DOCUMENTS is not None and written >= MAX_DOCUMENTS:
                    print(f"Reached MAX_DOCUMENTS={MAX_DOCUMENTS}", flush=True)
                    cur.close()
                    conn.close()
                    print(f"Done. Written {written} documents to {OUTPUT_PATH}", flush=True)
                    return

    cur.close()
    conn.close()
    print(f"Done. Written {written} documents to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()