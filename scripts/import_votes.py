import xml.etree.ElementTree as ET
from datetime import datetime

from backend.app.db import get_connection


def parse_creation_date(value: str):
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def main():
    conn = get_connection()
    cur = conn.cursor()

    xml_path = "data/raw/Votes.xml"

    processed = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        vote_id = elem.attrib.get("Id")
        post_id = elem.attrib.get("PostId")
        vote_type_id = elem.attrib.get("VoteTypeId")
        creation_date = elem.attrib.get("CreationDate")
        bounty_amount = elem.attrib.get("BountyAmount")

        if vote_id is None or post_id is None or vote_type_id is None:
            elem.clear()
            continue

        cur.execute(
            """
            INSERT INTO votes (
                id,
                post_id,
                vote_type_id,
                creation_date,
                bounty_amount
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                int(vote_id),
                int(post_id),
                int(vote_type_id),
                parse_creation_date(creation_date),
                int(bounty_amount) if bounty_amount is not None else None,
            ),
        )

        processed += 1

        if processed % 25000 == 0:
            conn.commit()
            print(f"Processed {processed} votes...")

        elem.clear()

    conn.commit()
    cur.close()
    conn.close()

    print(f"Finished. Processed {processed} votes.")


if __name__ == "__main__":
    main()