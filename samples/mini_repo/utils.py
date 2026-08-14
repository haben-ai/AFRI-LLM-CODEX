def sanitize_input(value):
    """Strip whitespace and quotes from user-supplied input."""
    return value.strip().replace("'", "")
