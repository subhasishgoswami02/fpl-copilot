"""Decision engine: transfer options (including roll), captain/vice, XI and bench order.

Deterministic. Produces ranked options with EV deltas; the LLM only narrates.
"""

POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_SHAPE = {1: 2, 2: 5, 3: 5, 4: 3}
# transfer only if best option beats roll by this much horizon EV
MODE_THRESHOLDS = {"safe": 6.0, "balanced": 4.0, "aggressive": 2.5}
MAX_PER_CLUB = 3


def pick_xi(squad_preds, elements):
    """Best valid XI by single-GW EV. Formation: 1 GK, >=3 DEF, >=2 MID, >=1 FWD, 11 total."""
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for p in squad_preds:
        by_pos[elements[p["player_id"]]["element_type"]].append(p)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p["ev"], reverse=True)

    xi = [by_pos[1][0]] if by_pos[1] else []
    xi += by_pos[2][:3] + by_pos[3][:2] + by_pos[4][:1]
    chosen = {p["player_id"] for p in xi}
    remaining = sorted(
        (p for pos in (2, 3, 4) for p in by_pos[pos] if p["player_id"] not in chosen),
        key=lambda p: p["ev"], reverse=True,
    )
    # positional caps: DEF<=5, MID<=5, FWD<=3
    counts = {2: 3, 3: 2, 4: 1}
    caps = {2: 5, 3: 5, 4: 3}
    for p in remaining:
        if len(xi) == 11:
            break
        pos = elements[p["player_id"]]["element_type"]
        if counts[pos] < caps[pos]:
            xi.append(p)
            counts[pos] += 1

    bench_gk = by_pos[1][1:2]
    bench_out = [p for p in remaining if p not in xi]
    bench = bench_gk + sorted(bench_out, key=lambda p: p["ev"], reverse=True)
    return xi, bench


def captain_picks(xi, elements):
    ranked = sorted(xi, key=lambda p: p["ev"], reverse=True)
    captain, vice = ranked[0], ranked[1]
    differential = None
    for p in sorted(xi, key=lambda p: p["p90"], reverse=True):
        el = elements[p["player_id"]]
        if float(el.get("selected_by_percent", "100") or 100) < 10.0:
            differential = p
            break
    return captain, vice, differential


def transfer_options(squad, bank, predictor, elements, event, horizon, top_n=5):
    """Rank single-transfer options against rolling. EVs are horizon-weighted."""
    squad_set = set(squad)
    club_count = {}
    for pid in squad:
        club = elements[pid]["team"]
        club_count[club] = club_count.get(club, 0) + 1

    h_ev = {pid: predictor.horizon_ev(pid, event, horizon) for pid in squad}

    options = []
    for out_id in squad:
        out_el = elements[out_id]
        budget = out_el["now_cost"] + bank  # sell price approximated by current price
        pool = [
            el for el in elements.values()
            if el["element_type"] == out_el["element_type"]
            and el["id"] not in squad_set
            and el["now_cost"] <= budget
            and el.get("status", "a") not in ("u", "n")
        ]
        # cheap pre-filter before expensive horizon EV: form + points signal
        pool.sort(key=lambda e: (float(e.get("form", 0) or 0), e.get("total_points", 0)),
                  reverse=True)
        for cand in pool[:12]:
            new_count = club_count.get(cand["team"], 0) + 1
            if cand["team"] != out_el["team"] and new_count > MAX_PER_CLUB:
                continue
            cand_ev = predictor.horizon_ev(cand["id"], event, horizon)
            delta = round(cand_ev - h_ev[out_id], 2)
            if delta <= 0:
                continue
            options.append({
                "out": out_id, "in": cand["id"], "delta_horizon_ev": delta,
                "out_ev": h_ev[out_id], "in_ev": cand_ev,
                "cost_change": cand["now_cost"] - out_el["now_cost"],
            })
    options.sort(key=lambda o: o["delta_horizon_ev"], reverse=True)
    return options[:top_n]


def decide(squad_info, predictor, elements, event, horizon, mode, free_transfers):
    squad = squad_info["players"]
    squad_preds = [predictor.predict_event(pid, event) for pid in squad]
    pred_by_id = {p["player_id"]: p for p in squad_preds}

    xi, bench = pick_xi(squad_preds, elements)
    captain, vice, differential = captain_picks(xi, elements)

    options = transfer_options(
        squad, squad_info["bank"], predictor, elements, event, horizon
    )
    threshold = MODE_THRESHOLDS.get(mode, 4.0)
    best = options[0] if options else None
    action = "transfer" if (best and best["delta_horizon_ev"] >= threshold) else "roll"

    # predicted range for the recommended XI with captain doubled
    ev_total = sum(p["ev"] for p in xi) + captain["ev"]
    var = sum(((p["p90"] - p["p10"]) / 2.5631) ** 2 for p in xi)
    var += ((captain["p90"] - captain["p10"]) / 2.5631) ** 2  # captain counts twice
    sigma = var ** 0.5
    p10_total = max(0.0, ev_total - 1.2816 * sigma)
    p90_total = ev_total + 1.2816 * sigma

    return {
        "event": event,
        "mode": mode,
        "free_transfers": free_transfers,
        "action": action,
        "transfer": best if action == "transfer" else None,
        "alternatives": options,
        "roll_rationale": (
            None if action == "transfer" else
            f"Best available move gains only {best['delta_horizon_ev'] if best else 0} "
            f"horizon EV, below the {mode} threshold of {threshold}. "
            "Rolling preserves flexibility."
        ),
        "captain": captain["player_id"],
        "vice_captain": vice["player_id"],
        "differential_captain": differential["player_id"] if differential else None,
        "xi": [p["player_id"] for p in xi],
        "bench_order": [p["player_id"] for p in bench],
        "predicted_points": {
            "ev": round(ev_total, 1),
            "p10": round(p10_total, 1),
            "p90": round(p90_total, 1),
        },
        "squad_predictions": squad_preds,
        "pred_by_id": {str(k): v for k, v in pred_by_id.items()},
    }
