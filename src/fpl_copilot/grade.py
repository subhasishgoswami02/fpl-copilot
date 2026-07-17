"""Post-gameweek self-grading.

Process and outcome are graded separately, and every grade is anchored against
naive baselines. "Predicted 64-72, scored 69" is calibration; "beat the
do-nothing baseline" is decision quality. Both are stored.
"""


def pinball_loss(actual, quantile_value, tau):
    diff = actual - quantile_value
    return tau * diff if diff >= 0 else (tau - 1) * diff


def grade_event(rec, actual_points, actual_picks, bootstrap_event, elements):
    """
    rec:            stored recommendation row (payload = decision dict)
    actual_points:  {player_id: gw points} from event/{gw}/live
    actual_picks:   picks payload for the manager for that GW (what he actually did)
    bootstrap_event: the event dict from bootstrap (has most_captained etc.)
    """
    d = rec["payload"]
    pred = {int(k): v for k, v in d["pred_by_id"].items()}

    # --- calibration: predicted XI range vs the XI's actual points ------------
    xi = d["xi"]
    cap = d["captain"]
    xi_actual = sum(actual_points.get(pid, 0) for pid in xi) + actual_points.get(cap, 0)
    rng = d["predicted_points"]
    range_hit = rng["p10"] <= xi_actual <= rng["p90"]

    # per-player quantile loss (players with a prediction and a result)
    losses = []
    for pid, p in pred.items():
        if pid not in actual_points:
            continue
        a = actual_points[pid]
        losses.append(pinball_loss(a, p["p10"], 0.10) + pinball_loss(a, p["p90"], 0.90))
    avg_pinball = round(sum(losses) / len(losses), 3) if losses else None

    # --- captain: outcome percentile within recommended XI --------------------
    xi_scores = sorted(actual_points.get(pid, 0) for pid in xi)
    cap_actual = actual_points.get(cap, 0)
    cap_percentile = round(
        sum(1 for s in xi_scores if s <= cap_actual) / len(xi_scores), 2
    ) if xi_scores else None

    # baseline: the crowd's captain
    crowd_cap = bootstrap_event.get("most_captained")
    crowd_cap_actual = actual_points.get(crowd_cap, 0) if crowd_cap else None
    beat_crowd_captain = (
        None if crowd_cap_actual is None else cap_actual >= crowd_cap_actual
    )

    # --- transfer: recommended move vs do-nothing and vs the template move ----
    transfer_grade = None
    if d.get("action") == "transfer" and d.get("transfer"):
        t = d["transfer"]
        in_pts = actual_points.get(t["in"], 0)
        out_pts = actual_points.get(t["out"], 0)
        template_in = bootstrap_event.get("most_transferred_in")
        template_pts = actual_points.get(template_in, 0) if template_in else None
        transfer_grade = {
            "in": t["in"], "out": t["out"],
            "in_actual": in_pts, "out_actual": out_pts,
            "delta_actual": in_pts - out_pts,          # vs do-nothing
            "delta_predicted_1gw": None,               # horizon EV isn't 1GW comparable
            "template_in": template_in,
            "template_in_actual": template_pts,
            "beat_template_move": None if template_pts is None else in_pts >= template_pts,
        }

    # --- bench order: any bench outfielder outscoring an XI player ------------
    bench = [pid for pid in d["bench_order"]]
    bench_errors = []
    if bench and elements:
        xi_min = min(
            (actual_points.get(pid, 0) for pid in xi
             if elements.get(pid, {}).get("element_type") != 1),
            default=0,
        )
        for pid in bench:
            if elements.get(pid, {}).get("element_type") == 1:
                continue
            if actual_points.get(pid, 0) > xi_min + 2:  # tolerance: noise vs real error
                bench_errors.append({"player": pid, "points": actual_points.get(pid, 0)})

    # --- did the manager follow the recommendation? ---------------------------
    followed = None
    if actual_picks:
        actual_xi = {p["element"] for p in actual_picks["picks"] if p["position"] <= 11}
        actual_cap = next(
            (p["element"] for p in actual_picks["picks"] if p["is_captain"]), None
        )
        followed = {
            "xi_overlap": len(actual_xi & set(xi)),
            "captain_followed": actual_cap == cap,
        }

    return {
        "event": rec["event"],
        "calibration": {
            "predicted": rng,
            "actual_xi_points": xi_actual,
            "range_hit": range_hit,
            "avg_pinball_loss": avg_pinball,
        },
        "captain": {
            "recommended": cap,
            "actual_points": cap_actual,
            "percentile_in_xi": cap_percentile,
            "crowd_captain": crowd_cap,
            "crowd_captain_points": crowd_cap_actual,
            "beat_crowd_captain": beat_crowd_captain,
        },
        "transfer": transfer_grade,
        "action_taken": d.get("action"),
        "bench_errors": bench_errors,
        "followed": followed,
    }


def season_scoreboard(gradings):
    """Aggregate calibration + decision quality across all graded events."""
    if not gradings:
        return {"graded_events": 0}
    n = len(gradings)
    hits = sum(1 for g in gradings if g["payload"]["calibration"]["range_hit"])
    cap_beats = [
        g["payload"]["captain"]["beat_crowd_captain"]
        for g in gradings
        if g["payload"]["captain"]["beat_crowd_captain"] is not None
    ]
    transfers = [g["payload"]["transfer"] for g in gradings if g["payload"]["transfer"]]
    positive_transfers = sum(1 for t in transfers if t["delta_actual"] > 0)
    pinballs = [
        g["payload"]["calibration"]["avg_pinball_loss"]
        for g in gradings
        if g["payload"]["calibration"]["avg_pinball_loss"] is not None
    ]
    return {
        "graded_events": n,
        "range_coverage": round(hits / n, 2),          # target ~0.80 for p10-p90
        "captain_vs_crowd": (
            round(sum(cap_beats) / len(cap_beats), 2) if cap_beats else None
        ),
        "transfers_made": len(transfers),
        "transfers_beat_do_nothing": positive_transfers,
        "avg_pinball_loss": (
            round(sum(pinballs) / len(pinballs), 3) if pinballs else None
        ),
    }
