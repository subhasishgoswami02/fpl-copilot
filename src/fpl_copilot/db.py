"""SQLite persistence. Recommendations are immutable: one per (manager, event), ever."""
import datetime
import json
import pathlib
import sqlite3
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,            -- recommend | grade | backtest
    event       INTEGER,
    started_utc TEXT NOT NULL,
    detail      TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    manager_id   INTEGER NOT NULL,
    event        INTEGER NOT NULL,
    run_id       TEXT NOT NULL,
    created_utc  TEXT NOT NULL,
    deadline_utc TEXT NOT NULL,
    payload      TEXT NOT NULL,           -- full decision JSON
    PRIMARY KEY (manager_id, event)
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id    TEXT NOT NULL,
    event     INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    ev        REAL, p10 REAL, p90 REAL,
    detail    TEXT,
    PRIMARY KEY (run_id, event, player_id)
);
CREATE TABLE IF NOT EXISTS gradings (
    manager_id  INTEGER NOT NULL,
    event       INTEGER NOT NULL,
    run_id      TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (manager_id, event)
);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class ImmutabilityError(RuntimeError):
    pass


class Db:
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def new_run(self, kind, event=None, detail=""):
        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO agent_runs VALUES (?,?,?,?,?)",
            (run_id, kind, event, utcnow(), detail),
        )
        self.conn.commit()
        return run_id

    # -- recommendations (immutable) ---------------------------------------
    def save_recommendation(self, manager_id, event, run_id, deadline_utc, payload):
        if self.get_recommendation(manager_id, event) is not None:
            raise ImmutabilityError(
                f"Recommendation for manager {manager_id} GW{event} already exists; "
                "recommendations are never overwritten."
            )
        self.conn.execute(
            "INSERT INTO recommendations VALUES (?,?,?,?,?,?)",
            (manager_id, event, run_id, utcnow(), deadline_utc, json.dumps(payload)),
        )
        self.conn.commit()

    def get_recommendation(self, manager_id, event):
        row = self.conn.execute(
            "SELECT * FROM recommendations WHERE manager_id=? AND event=?",
            (manager_id, event),
        ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        rec["payload"] = json.loads(rec["payload"])
        return rec

    def save_predictions(self, run_id, event, preds):
        rows = [
            (run_id, event, p["player_id"], p["ev"], p["p10"], p["p90"],
             json.dumps(p.get("detail", {})))
            for p in preds
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()

    def get_predictions(self, run_id, event):
        rows = self.conn.execute(
            "SELECT * FROM predictions WHERE run_id=? AND event=?", (run_id, event)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- gradings ------------------------------------------------------------
    def save_grading(self, manager_id, event, run_id, payload):
        self.conn.execute(
            "INSERT OR REPLACE INTO gradings VALUES (?,?,?,?,?)",
            (manager_id, event, run_id, utcnow(), json.dumps(payload)),
        )
        self.conn.commit()

    def get_grading(self, manager_id, event):
        row = self.conn.execute(
            "SELECT * FROM gradings WHERE manager_id=? AND event=?",
            (manager_id, event),
        ).fetchone()
        if row is None:
            return None
        g = dict(row)
        g["payload"] = json.loads(g["payload"])
        return g

    def all_gradings(self, manager_id):
        rows = self.conn.execute(
            "SELECT * FROM gradings WHERE manager_id=? ORDER BY event", (manager_id,)
        ).fetchall()
        out = []
        for r in rows:
            g = dict(r)
            g["payload"] = json.loads(g["payload"])
            out.append(g)
        return out

    # -- generic state --------------------------------------------------------
    def set_state(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value))
        )
        self.conn.commit()

    def get_state(self, key, default=None):
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default
