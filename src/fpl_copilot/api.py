"""FPL public API client with retries and immutable raw-snapshot persistence.

Every recommend run passes a snapshot_dir; the raw JSON that existed before the
deadline is written there and committed, so grading can never leak post-deadline data.
"""
import json
import pathlib
import time

import requests

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-copilot; personal research project)"}


class FplApiError(RuntimeError):
    pass


class FplApi:
    def __init__(self, snapshot_dir=None):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.snapshot_dir = pathlib.Path(snapshot_dir) if snapshot_dir else None

    def _get(self, path, snapshot_name=None, retries=4):
        url = f"{BASE}/{path}"
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    if self.snapshot_dir and snapshot_name:
                        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                        out = self.snapshot_dir / f"{snapshot_name}.json"
                        if not out.exists():  # snapshots are immutable
                            out.write_text(json.dumps(data, separators=(",", ":")))
                    return data
                if resp.status_code in (403, 429, 500, 502, 503):
                    last_err = f"HTTP {resp.status_code}"
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_err = str(exc)
                time.sleep(2 ** attempt)
        raise FplApiError(f"GET {path} failed after {retries} attempts: {last_err}")

    # -- game-wide ---------------------------------------------------------
    def bootstrap(self):
        return self._get("bootstrap-static/", "bootstrap")

    def fixtures(self):
        return self._get("fixtures/", "fixtures")

    def event_live(self, gw):
        return self._get(f"event/{gw}/live/", f"live_gw{gw}")

    def event_status(self):
        return self._get("event-status/")

    def element_summary(self, player_id):
        # not snapshotted individually (many small calls); features derived from
        # these are stored in the predictions payload instead
        return self._get(f"element-summary/{player_id}/")

    # -- manager-specific (all public, no login) ---------------------------
    def entry(self, manager_id):
        return self._get(f"entry/{manager_id}/", "entry")

    def entry_history(self, manager_id):
        return self._get(f"entry/{manager_id}/history/", "entry_history")

    def entry_picks(self, manager_id, gw):
        return self._get(f"entry/{manager_id}/event/{gw}/picks/", f"picks_gw{gw}")

    def entry_transfers(self, manager_id):
        return self._get(f"entry/{manager_id}/transfers/", "transfers")


def next_event(bootstrap):
    """The upcoming (not yet started) event, from bootstrap 'events'."""
    for ev in bootstrap["events"]:
        if ev.get("is_next"):
            return ev
    # fallback: first unfinished event
    for ev in bootstrap["events"]:
        if not ev["finished"] and not ev.get("is_current"):
            return ev
    return None


def last_finished_event(bootstrap):
    done = [ev for ev in bootstrap["events"] if ev["finished"]]
    return done[-1] if done else None
