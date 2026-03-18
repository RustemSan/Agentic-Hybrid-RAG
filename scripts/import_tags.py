import xml.etree.ElementTree as ET

from backend.app.db import get_connection


def main():
    conn = get_connection()
    cur = conn.cursor()

    xml_path = "data/raw/Tags.xml"

    inserted = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "row":
            continue

        tag_id = elem.attrib.get("Id")
        tag_name = elem.attrib.get("TagName")
        count = elem.attrib.get("Count")

        if tag_id is None or tag_name is None:
            elem.clear()
            continue

        cur.execute(
            """
            INSERT INTO tags (id, tag_name, count)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                int(tag_id),
                tag_name,
                int(count) if count is not None else None,
            ),
        )

        inserted += 1

        if inserted % 1000 == 0:
            conn.commit()
            print(f"Inserted {inserted} tags...")

        elem.clear()

    conn.commit()
    cur.close()
    conn.close()

    print(f"Finished. Inserted/processed {inserted} tags.")


if __name__ == "__main__":
    main()