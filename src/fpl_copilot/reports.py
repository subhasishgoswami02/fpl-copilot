"""Markdown report rendering. These files are the public record."""
import json
import pathlib


def _name_map(elements):
    return {str(pid): el["web_name"] for pid, el in elements.items()}


def write_recommendation_report(reports_dir, decision, memo, elements, deadline_utc,
                                created_utc, run_id):
    names = _name_map(elements)
    d = decision
    lines = [
        f"# GW{d['event']} Recommendation",
        "",
        f"Committed: `{created_utc}` | Deadline: `{deadline_utc}` | Run: `{run_id}`",
        "",
        f"**Action:** {'Transfer' if d['action'] == 'transfer' else 'Roll the transfer'}"
        + (
            f" — {names.get(str(d['transfer']['out']))} → "
            f"{names.get(str(d['transfer']['in']))} "
            f"(+{d['transfer']['delta_horizon_ev']} horizon EV)"
            if d["action"] == "transfer" and d["transfer"] else ""
        ),
        f"**Captain:** {names.get(str(d['captain']))} | "
        f"**Vice:** {names.get(str(d['vice_captain']))}"
        + (
            f" | **Differential:** {names.get(str(d['differential_captain']))}"
            if d.get("differential_captain") else ""
        ),
        f"**Predicted XI points (captain doubled):** {d['predicted_points']['ev']} "
        f"(80% range: {d['predicted_points']['p10']}–{d['predicted_points']['p90']})",
        f"**Mode:** {d['mode']} | **Free transfers:** {d['free_transfers']}",
        "",
        "## Memo",
        "",
        memo,
        "",
        "## Alternatives considered",
        "",
    ]
    for o in d.get("alternatives", []):
        lines.append(
            f"- {names.get(str(o['out']))} → {names.get(str(o['in']))}: "
            f"+{o['delta_horizon_ev']} horizon EV"
        )
    if not d.get("alternatives"):
        lines.append("- (no positive-EV moves found)")
    lines += ["- Roll the transfer: +0 EV now, +1 flexibility next week", "",
              "## Player predictions (this GW)", "",
              "| Player | EV | p10 | p90 | xMins |", "|---|---|---|---|---|"]
    for p in sorted(d["squad_predictions"], key=lambda p: p["ev"], reverse=True):
        lines.append(
            f"| {names.get(str(p['player_id']))} | {p['ev']} | {p['p10']} | "
            f"{p['p90']} | {p['detail'].get('exp_minutes', '-')} |"
        )
    path = pathlib.Path(reports_dir) / f"gw{d['event']:02d}-recommendation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

    # machine-readable predictions alongside, for public verifiability
    (pathlib.Path(reports_dir) / f"gw{d['event']:02d}-predictions.json").write_text(
        json.dumps({"run_id": run_id, "created_utc": created_utc,
                    "deadline_utc": deadline_utc, "decision": d}, indent=1)
    )
    return path


def write_grading_report(reports_dir, grading, narrative, scoreboard, elements,
                         new_hypotheses):
    names = _name_map(elements)
    g = grading
    cal = g["calibration"]
    cap = g["captain"]
    lines = [
        f"# GW{g['event']} Grading",
        "",
        f"**Predicted:** {cal['predicted']['ev']} "
        f"({cal['predicted']['p10']}–{cal['predicted']['p90']}) | "
        f"**Actual:** {cal['actual_xi_points']} | "
        f"**Range {'HIT' if cal['range_hit'] else 'MISS'}**",
        f"**Avg pinball loss:** {cal['avg_pinball_loss']}",
        "",
        f"**Captain:** {names.get(str(cap['recommended']), cap['recommended'])} "
        f"scored {cap['actual_points']} "
        f"(percentile in XI: {cap['percentile_in_xi']}); crowd captain "
        f"{names.get(str(cap['crowd_captain']), cap['crowd_captain'])} scored "
        f"{cap['crowd_captain_points']} — "
        f"{'beat' if cap['beat_crowd_captain'] else 'lost to'} the crowd"
        if cap["beat_crowd_captain"] is not None else
        f"**Captain:** {names.get(str(cap['recommended']), cap['recommended'])} "
        f"scored {cap['actual_points']}",
        "",
    ]
    if g.get("transfer"):
        t = g["transfer"]
        lines += [
            f"**Transfer:** {names.get(str(t['out']), t['out'])} "
            f"({t['out_actual']} pts) → {names.get(str(t['in']), t['in'])} "
            f"({t['in_actual']} pts): {t['delta_actual']:+d} vs do-nothing",
        ]
    else:
        lines += [f"**Action:** {g.get('action_taken', 'roll')} (no transfer to grade)"]
    if g.get("bench_errors"):
        lines += ["", "**Bench errors:**"] + [
            f"- {names.get(str(b['player']), b['player'])} scored {b['points']} on the bench"
            for b in g["bench_errors"]
        ]
    lines += [
        "", "## Postmortem", "", narrative, "",
        "## New rule hypotheses",
        "",
        ("\n".join(f"- `{h}`" for h in new_hypotheses)
         if new_hypotheses else "(none proposed)"),
        "", "## Season scoreboard", "",
        "```json", json.dumps(scoreboard, indent=1), "```",
    ]
    path = pathlib.Path(reports_dir) / f"gw{g['event']:02d}-grading.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
