import sqlite3


def connect(db_path="app.db"):
    """Open a SQLite connection to the local database."""
    return sqlite3.connect(db_path)


def execute_query(conn, query):
    """Run a raw SQL query and return all matching rows."""
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()
