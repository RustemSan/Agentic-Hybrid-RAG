import psycopg2


def get_connection():
    return psycopg2.connect(
        dbname="stackoverflow",
        user="krylodar",
        password="vwmrag_project",
        host="localhost",
        port="5432",
    )