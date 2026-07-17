"""Pre-deadline squad reconstruction.

The public API only exposes picks AFTER a deadline. The current squad before the
upcoming deadline is therefore reconstructed as:

    picks at last deadline  +  transfers registered for the upcoming event

Bank is adjusted using the recorded in/out costs on the transfers endpoint (exact),
while hypothetical future sales use current price (approximation, documented).
"""


class SquadUnknown(RuntimeError):
    pass


def reconstruct_squad(api, db, manager_id, upcoming_event):
    # manual override (required before GW1, useful any time reconstruction drifts)
    manual = db.get_state(f"manual_squad_gw{upcoming_event}")
    if manual:
        return {
            "players": sorted(manual["players"]),
            "bank": manual.get("bank", 0),
            "source": "manual",
            "pending_transfers": [],
        }

    last_event = upcoming_event - 1
    if last_event < 1:
        raise SquadUnknown(
            "Before the GW1 deadline the public API exposes no squad. "
            "Run: python -m fpl_copilot.cli set-squad <15 comma-separated player ids> [--bank N]"
        )

    picks = api.entry_picks(manager_id, last_event)
    players = {p["element"] for p in picks["picks"]}
    bank = picks["entry_history"]["bank"]  # tenths of £m

    pending = [
        t for t in api.entry_transfers(manager_id) if t["event"] == upcoming_event
    ]
    for t in pending:
        players.discard(t["element_out"])
        players.add(t["element_in"])
        bank += t["element_out_cost"] - t["element_in_cost"]

    if len(players) != 15:
        raise SquadUnknown(
            f"Reconstructed squad has {len(players)} players (expected 15). "
            "A chip or unusual transfer history may be involved; use set-squad to correct."
        )

    return {
        "players": sorted(players),
        "bank": bank,
        "source": f"picks_gw{last_event}+{len(pending)}_transfers",
        "pending_transfers": pending,
    }


def track_free_transfers(db, entry_history, upcoming_event):
    """Estimate available free transfers (max 5 under current rules).

    Uses last event's recorded transfer count; can be corrected manually via
    `cli set-state free_transfers N`.
    """
    override = db.get_state("free_transfers_override")
    if override is not None:
        return int(override)
    ft = int(db.get_state("free_transfers", 1))
    last_processed = db.get_state("ft_last_event", 0)
    for row in entry_history.get("current", []):
        ev = row["event"]
        if ev <= last_processed or ev >= upcoming_event:
            continue
        used = row.get("event_transfers", 0)
        # paid hits don't consume next week's FT
        paid = row.get("event_transfers_cost", 0) // 4
        free_used = max(0, used - paid)
        ft = min(5, max(1, ft - free_used + 1))
        last_processed = ev
    db.set_state("free_transfers", ft)
    db.set_state("ft_last_event", last_processed)
    return ft
