import sqlite3
import datetime
import os

# Use /tmp for production (Render), local dir for development
if os.environ.get("RENDER"):
    DB_PATH = "/tmp/usage.db"
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "usage.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Devices table
    c.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            news_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            last_reset DATE
        )
    ''')
    conn.commit()
    conn.close()

def _reset_if_needed(cursor, table, id_column, id_value):
    today = datetime.date.today().isoformat()
    cursor.execute(f"SELECT last_reset FROM {table} WHERE {id_column} = ?", (id_value,))
    row = cursor.fetchone()
    if not row:
        cursor.execute(f"INSERT INTO {table} ({id_column}, last_reset) VALUES (?, ?)", (id_value, today))
    elif row['last_reset'] != today:
        cursor.execute(f"UPDATE {table} SET news_count = 0, image_count = 0, last_reset = ? WHERE {id_column} = ?", (today, id_value))

def check_and_increment_usage(device_id=None, scan_type="news", ip=None):
    """
    scan_type can be 'news' or 'image'
    Returns (allowed: bool, reason: str)
    """
    # Define daily limits
    DEVICE_LIMIT = 3

    # Use IP as fallback if no device fingerprint provided
    effective_id = device_id if device_id else ip
    if not effective_id:
        # No way to identify — allow but don't track
        return True, ""

    conn = get_db_connection()
    c = conn.cursor()

    try:
        _reset_if_needed(c, "devices", "device_id", effective_id)
        column = f"{scan_type}_count"
        c.execute(f"SELECT {column} FROM devices WHERE device_id = ?", (effective_id,))
        count = c.fetchone()[0]

        if count >= DEVICE_LIMIT:
            return False, f"You've used all {DEVICE_LIMIT} free {scan_type} scans for today. Please visit tomorrow or try a different device."

        c.execute(f"UPDATE devices SET {column} = {column} + 1 WHERE device_id = ?", (effective_id,))
        conn.commit()
        return True, ""

    finally:
        conn.close()

# Initialize on import
init_db()
