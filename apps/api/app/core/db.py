from psycopg_pool import ConnectionPool
from .settings import settings
from .logging import log_event, hash_text
import time

pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=5)

def query(sql: str, params=None):
    start = time.time()
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall() if cur.description else None
        latency = int((time.time() - start) * 1000)
        log_event(
            "db.query",
            sql_hash=hash_text(sql),
            latency_ms=latency,
            row_count=(len(rows) if rows is not None else 0),
        )
        return rows
    except Exception as ex:
        latency = int((time.time() - start) * 1000)
        log_event(
            "db.query",
            level="error",
            sql_hash=hash_text(sql),
            latency_ms=latency,
            error=str(ex),
        )
        raise
