# CONTRACT: database/__init__.py
# Lazy import to prevent circular imports at package load time.

def get_db_manager():
    """
    Lazily import and return the DatabaseManager class.

    Returns:
        type[DatabaseManager]: The DatabaseManager class (not an instance).
    """
    from database.db_manager import DatabaseManager  # noqa: PLC0415
    return DatabaseManager
