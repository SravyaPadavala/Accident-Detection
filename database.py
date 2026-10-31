"""SQLite storage for incident records."""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("ACCIDENT_DB",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "accident_detection.db"))


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_path TEXT,
            source TEXT,
            vehicles INTEGER)""")


def add_incident(confidence, evidence_path=None, source="video", vehicles=0):
    init_db()
    with _conn() as c:
        cur = c.execute("INSERT INTO incidents (timestamp, confidence, evidence_path, source, vehicles) "
                        "VALUES (?,?,?,?,?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), float(confidence),
                         evidence_path, source, int(vehicles)))
        return cur.lastrowid


def list_incidents(limit=100):
    init_db()
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,))]


def delete_incident(incident_id):
    init_db()
    with _conn() as c:
        return c.execute("DELETE FROM incidents WHERE id=?", (incident_id,)).rowcount > 0


def get_stats():
    init_db()
    with _conn() as c:
        total, avg, last = c.execute(
            "SELECT COUNT(*), AVG(confidence), MAX(timestamp) FROM incidents").fetchone()
        today = c.execute("SELECT COUNT(*) FROM incidents WHERE date(timestamp)=date('now','localtime')").fetchone()[0]
    return {"total_incidents": total, "today_incidents": today,
            "avg_confidence": round(avg, 3) if avg else 0.0, "last_incident": last}
