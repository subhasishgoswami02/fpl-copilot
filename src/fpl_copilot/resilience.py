"""Resilience primitives for external API calls: a circuit breaker and a
last-known-good snapshot cache.

Not wired into FplApi or the CLI yet. FplApi._get already retries with
backoff (now with jitter, see api.py); this module is the next layer: fail
fast once an upstream is clearly down instead of retrying forever, and offer
a stale-but-usable fallback instead of a hard crash.

Intended integration (see 02-architecture/system-design.md in the
production-rebuild planning docs): cli.py's cmd_recommend / cmd_grade /
cmd_status would wrap their top-level api.bootstrap() / api.fixtures() calls
in resilient_call(), catching FplApiDegraded to log a degraded run instead of
crashing, and letting a plain RuntimeError/FplApiError through only when
there is truly no fallback available. That CLI wiring is deliberately left
for a follow-up change: it changes what a live scheduled run does on
deadline day, and that deserves its own review and a local test run, not a
silent change bundled into this foundation PR.
"""
import json
import pathlib
import time


class CircuitOpenError(RuntimeError):
    """Raised by CircuitBreaker.call when the circuit is open and the
    cooldown hasn't elapsed; the caller should not even attempt the network
    call."""


class CircuitBreaker:
    """Minimal 3-state breaker: closed -> open -> half-open -> closed.

    Opens after `failure_threshold` consecutive failures. Once open, calls
    are rejected immediately (via CircuitOpenError) until `reset_after`
    seconds have passed, at which point a single probe call is allowed
    (half-open); success closes the circuit, failure re-opens it and resets
    the cooldown clock.
    """

    def __init__(self, failure_threshold=4, reset_after=300):
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self._consecutive_failures = 0
        self._state = "closed"
        self._opened_at = None

    def _current_state(self):
        if self._state == "open" and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.reset_after:
                self._state = "half-open"
        return self._state

    @property
    def state(self):
        return self._current_state()

    def call(self, fn, *args, **kwargs):
        current = self._current_state()
        if current == "open":
            raise CircuitOpenError(
                f"circuit open, {self.reset_after}s cooldown not yet elapsed"
            )
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_failure(self):
        self._consecutive_failures += 1
        if self._state == "half-open" or (
            self._consecutive_failures >= self.failure_threshold
        ):
            self._state = "open"
            self._opened_at = time.monotonic()

    def _on_success(self):
        self._consecutive_failures = 0
        self._state = "closed"
        self._opened_at = None


class SnapshotCache:
    """Finds the most recent previously-committed snapshot for a given name
    across all past gameweek snapshot directories, for use as a degraded
    fallback when the live API is unreachable.

    Expects the layout fpl-copilot already uses:
    `snapshots_root/gwNN/name.json` (see api.py's FplApi(snapshot_dir=...)
    and config.yaml's snapshots_dir, and cli.py's f"gw{gw:02d}" convention).
    """

    def __init__(self, snapshots_root):
        self.snapshots_root = pathlib.Path(snapshots_root)

    def most_recent(self, name):
        """Returns (data, source_path) for the newest `name.json` found
        under snapshots_root/gw*/, or (None, None) if nothing is cached yet.

        "Newest" is by gameweek directory name (gw01, gw02, ...), which
        sorts correctly as long as gameweeks stay zero-padded to 2 digits,
        matching the existing convention.
        """
        if not self.snapshots_root.exists():
            return None, None
        candidates = sorted(self.snapshots_root.glob(f"gw*/{name}.json"))
        if not candidates:
            return None, None
        latest = candidates[-1]
        return json.loads(latest.read_text()), latest


class FplApiDegraded(RuntimeError):
    """Raised instead of letting the original error propagate when a live
    call failed but a cached snapshot was available to fall back to.
    Callers that want graceful degradation should catch this specifically;
    callers that don't will still see a clear exception rather than
    silently getting stale data.
    """

    def __init__(self, message, data, source_path):
        super().__init__(message)
        self.data = data
        self.source_path = source_path


def resilient_call(breaker, fetch_fn, cache, cache_name):
    """Runs fetch_fn() through the circuit breaker. On failure (or an open
    circuit), tries the snapshot cache for `cache_name` and raises
    FplApiDegraded with the cached payload if one exists, otherwise
    re-raises the original error.

    This is the intended call shape for the CLI/API layer integration
    described in system-design.md; not yet wired into cli.py.
    """
    try:
        return breaker.call(fetch_fn)
    except CircuitOpenError as exc:
        data, source = cache.most_recent(cache_name)
        if data is not None:
            raise FplApiDegraded(
                f"circuit open for '{cache_name}', using cached snapshot "
                f"from {source}",
                data, source,
            ) from exc
        raise
    except Exception as exc:
        data, source = cache.most_recent(cache_name)
        if data is not None:
            raise FplApiDegraded(
                f"live call for '{cache_name}' failed ({exc}), using cached "
                f"snapshot from {source}",
                data, source,
            ) from exc
        raise
