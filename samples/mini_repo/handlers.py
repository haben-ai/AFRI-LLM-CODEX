from db import connect, execute_query
from utils import sanitize_input


def get_user(user_id):
    """Look up a user by id -- vulnerable to SQL injection (raw string interpolation)."""
    conn = connect()
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return execute_query(conn, query)
