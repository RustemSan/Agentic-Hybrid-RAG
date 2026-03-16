from pathlib import Path
from backend.app.db import get_connection


def main():
    conn = get_connection()
    cur = conn.cursor()

    sql_path = Path("scripts/create_tables.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        cur.execute(f.read())

    conn.commit()
    cur.close()
    conn.close()

    print("Database schema created successfully.")


if __name__ == "__main__":
    main()