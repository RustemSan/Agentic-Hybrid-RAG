import xml.etree.ElementTree as ET
from datetime import datetime

from backend.app.db import get_connection

EXCLUDED_POST_IDS = {1000000001, 1000000010}

# Сначала ставь 10_000, потом 100_000, потом 1_000_000, потом 5_000_000
IMPORT_LIMIT = 5_000_000

# commit каждые N реально вставленных подходящих постов
COMMIT_EVERY = 50_000


def parse_datetime(value: str):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def parse_int(value: str):
    if value is None:
        return None
    return int(value)


def main():
    conn = get_connection()
    cur = conn.cursor()

    xml_path = "data/raw/Posts.xml"

    processed_xml_rows = 0
    matched_posts = 0           # подошло под фильтр
    inserted_rows = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        processed_xml_rows += 1

        post_id = parse_int(elem.attrib.get("Id"))
        post_type_id = parse_int(elem.attrib.get("PostTypeId"))

        if post_id is None or post_type_id is None:
            elem.clear()
            continue

        if post_id in EXCLUDED_POST_IDS:
            elem.clear()
            continue

        # берем только questions и answers
        if post_type_id not in (1, 2):
            elem.clear()
            continue

        matched_posts += 1

        parent_id = parse_int(elem.attrib.get("ParentId"))
        accepted_answer_id = parse_int(elem.attrib.get("AcceptedAnswerId"))
        creation_date = parse_datetime(elem.attrib.get("CreationDate"))
        score = parse_int(elem.attrib.get("Score"))
        view_count = parse_int(elem.attrib.get("ViewCount"))
        answer_count = parse_int(elem.attrib.get("AnswerCount"))
        comment_count = parse_int(elem.attrib.get("CommentCount"))
        favorite_count = parse_int(elem.attrib.get("FavoriteCount"))
        title = elem.attrib.get("Title")
        body = elem.attrib.get("Body")
        tags = elem.attrib.get("Tags")

        cur.execute(
            """
            INSERT INTO posts (
                id,
                post_type_id,
                parent_id,
                accepted_answer_id,
                creation_date,
                score,
                view_count,
                answer_count,
                comment_count,
                favorite_count,
                title,
                body,
                tags
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                post_id,
                post_type_id,
                parent_id,
                accepted_answer_id,
                creation_date,
                score,
                view_count,
                answer_count,
                comment_count,
                favorite_count,
                title,
                body,
                tags,
            ),
        )

        if cur.rowcount == 1:
            inserted_rows += 1

        if inserted_rows > 0 and inserted_rows % COMMIT_EVERY == 0:
            print(
                f"About to commit: processed_xml={processed_xml_rows}, "
                f"matched_posts={matched_posts}, inserted={inserted_rows}",
                flush=True,
            )
            conn.commit()
            print(
                f"Committed: processed_xml={processed_xml_rows}, "
                f"matched_posts={matched_posts}, inserted={inserted_rows}",
                flush=True,
            )

        elem.clear()

        # стопаем по реально вставленным полезным постам
        if IMPORT_LIMIT is not None and inserted_rows >= IMPORT_LIMIT:
            print(
                f"Reached import limit: inserted_rows={inserted_rows}, "
                f"processed_xml={processed_xml_rows}, matched_posts={matched_posts}",
                flush=True,
            )
            break

    conn.commit()
    cur.close()
    conn.close()

    print(
        f"Finished: processed_xml={processed_xml_rows}, "
        f"matched_posts={matched_posts}, inserted={inserted_rows}",
        flush=True,
    )


if __name__ == "__main__":
    main()
