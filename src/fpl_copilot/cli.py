"""CLI orchestrator. Commands:

  recommend [--if-due] [--force]   pre-deadline: snapshot, predict, decide, commit report
  grade     [--if-ready] [--force] post-GW: grade vs actuals + baselines, postmortem
  status                           next deadline, squad state, season scoreboard
  set-squad <ids> [--bank N]       manual squad (needed before GW1)
  set-state <key> <value>          e.g. set-state free_transfers_override 2
  backtest  [--season 2025-26]     calibration metrics on historical data
  validate-rules [--season ...]    promote/reject rule hypotheses via backtest
"""
import argparse
import datetime
import json
import pathlib
import sys

import yaml

from . import api as fpl_api
from .api import FplApi
from .db import Db
from .decide import decide
from .grade import grade_event, season_scoreboard
from .llm import postmortem, write_memo
from .model import Predictor
from .rules import active_rules, add_hypotheses, load_rules, save_rules
from .squad import SquadUnknown, reconstruct_squad, track_free_transfers

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_deadline(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
def cmd_recommend(args, cfg):
    db = Db(ROOT / cfg["db_path"])
    manager_id = cfg["manager_id"]

    probe = FplApi()
    bootstrap = probe.bootstrap()
    event = fpl_api.next_event(bootstrap)
    if event is None:
        print("No upcoming event (season over or not started).")
        return 0
    gw = event["id"]
    deadline = _parse_deadline(event["deadline_time"])
    hours_left = (deadline - _utcnow()).total_seconds() / 3600

    if args.if_due:
        if hours_left > cfg["recommend_window_hours"]:
            print(f"GW{gw} deadline is {hours_left:.0f}h away; outside "
                  f"{cfg['recommend_window_hours']}h window. Nothing to do.")
            return 0
        if hours_left < 0:
            print(f"GW{gw} deadline has passed. Nothing to do.")
            return 0
        if db.get_recommendation(manager_id, gw):
            print(f"GW{gw} already recommended (immutable). Nothing to do.")
            return 0
    elif db.get_recommendation(manager_id, gw) and not args.force:
        print(f"GW{gw} recommendation exists and is immutable. Use a new GW.")
        return 1

    # snapshot everything used for this decision, pre-deadline
    snap_dir = ROOT / cfg["snapshots_dir"] / f"gw{gw:02d}"
    apic = FplApi(snapshot_dir=snap_dir)
    bootstrap = apic.bootstrap()
    fixtures = apic.fixtures()
    elements = {e["id"]: e for e in bootstrap["elements"]}

    try:
        squad_info = reconstruct_squad(apic, db, manager_id, gw)
    except SquadUnknown as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        history = apic.entry_history(manager_id)
        free_transfers = track_free_transfers(db, history, gw)
    except Exception:
        free_transfers = int(db.get_state("free_transfers", 1))

    rules = active_rules(load_rules(ROOT / cfg["rules_path"]))
    predictor = Predictor(bootstrap, fixtures, rules=rules)

    # per-match history for squad + a bounded candidate pool (API courtesy)
    pool_ids = set(squad_info["players"])
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for el in bootstrap["elements"]:
        if el.get("status", "a") in ("u", "n") or el.get("minutes", 0) == 0:
            continue
        by_pos[el["element_type"]].append(el)
    for pos, els in by_pos.items():
        els.sort(key=lambda e: float(e.get("form", 0) or 0), reverse=True)
        pool_ids.update(e["id"] for e in els[:15])
    for pid in sorted(pool_ids):
        try:
            predictor.add_history(pid, apic.element_summary(pid))
        except Exception:
            pass  # predictor degrades gracefully to season aggregates

    run_id = db.new_run("recommend", gw, detail=f"squad={squad_info['source']}")
    decision = decide(squad_info, predictor, elements, gw,
                      cfg["horizon_gws"], cfg["mode"], free_transfers)

    names = {str(pid): elements[pid]["web_name"]
             for pid in pool_ids if pid in elements}
    memo = write_memo(decision, names, cfg["llm_model"])

    db.save_recommendation(manager_id, gw, run_id, event["deadline_time"], decision)
    db.save_predictions(run_id, gw, decision["squad_predictions"])

    from .reports import write_recommendation_report
    path = write_recommendation_report(
        ROOT / cfg["reports_dir"], decision, memo, elements,
        event["deadline_time"], _utcnow().isoformat(timespec="seconds"), run_id,
    )
    print(f"GW{gw} recommendation committed: {path}")
    print(f"Action: {decision['action']} | Captain: "
          f"{names.get(str(decision['captain']))} | Range: "
          f"{decision['predicted_points']['p10']}-{decision['predicted_points']['p90']}")
    return 0


# --------------------------------------------------------------------------
def cmd_grade(args, cfg):
    db = Db(ROOT / cfg["db_path"])
    manager_id = cfg["manager_id"]
    apic = FplApi()
    bootstrap = apic.bootstrap()
    elements = {e["id"]: e for e in bootstrap["elements"]}

    last = fpl_api.last_finished_event(bootstrap)
    if last is None:
        print("No finished event yet.")
        return 0
    gw = last["id"]

    rec = db.get_recommendation(manager_id, gw)
    if rec is None:
        print(f"No recommendation was logged for GW{gw}; nothing to grade.")
        return 0
    if db.get_grading(manager_id, gw) and not args.force:
        print(f"GW{gw} already graded. Nothing to do.")
        return 0

    if args.if_ready:
        status = apic.event_status()
        rows = [s for s in status.get("status", []) if s.get("event") == gw]
        if rows and not all(s.get("bonus_added") for s in rows):
            print(f"GW{gw} bonus not confirmed yet. Will retry next run.")
            return 0

    live = apic.event_live(gw)
    actual_points = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
    try:
        actual_picks = apic.entry_picks(manager_id, gw)
    except Exception:
        actual_picks = None

    run_id = db.new_run("grade", gw)
    grading = grade_event(rec, actual_points, actual_picks, last, elements)
    db.save_grading(manager_id, gw, run_id, grading)

    scoreboard = season_scoreboard(db.all_gradings(manager_id))
    pm = postmortem(grading, scoreboard, cfg["llm_model"])
    new_ids = add_hypotheses(ROOT / cfg["rules_path"], pm["hypotheses"],
                             created_from=f"GW{gw} grading")

    from .reports import write_grading_report
    path = write_grading_report(ROOT / cfg["reports_dir"], grading,
                                pm["narrative"], scoreboard, elements, new_ids)
    print(f"GW{gw} graded: {path}")
    cal = grading["calibration"]
    print(f"Range {'HIT' if cal['range_hit'] else 'MISS'}: predicted "
          f"{cal['predicted']['p10']}-{cal['predicted']['p90']}, "
          f"actual {cal['actual_xi_points']}")
    if new_ids:
        print(f"New hypotheses queued for validation: {new_ids}")
    return 0


# --------------------------------------------------------------------------
def cmd_status(args, cfg):
    db = Db(ROOT / cfg["db_path"])
    apic = FplApi()
    bootstrap = apic.bootstrap()
    ev = fpl_api.next_event(bootstrap)
    if ev:
        deadline = _parse_deadline(ev["deadline_time"])
        hours = (deadline - _utcnow()).total_seconds() / 3600
        rec = db.get_recommendation(cfg["manager_id"], ev["id"])
        print(f"Next: GW{ev['id']} deadline {ev['deadline_time']} ({hours:.1f}h away)")
        print(f"Recommendation logged: {'yes' if rec else 'no'}")
    print(json.dumps(season_scoreboard(db.all_gradings(cfg["manager_id"])), indent=1))
    return 0


def cmd_set_squad(args, cfg):
    db = Db(ROOT / cfg["db_path"])
    ids = [int(x) for x in args.players.split(",")]
    if len(ids) != 15:
        print(f"Need exactly 15 player ids, got {len(ids)}.")
        return 1
    apic = FplApi()
    bootstrap = apic.bootstrap()
    ev = fpl_api.next_event(bootstrap)
    gw = ev["id"] if ev else 1
    db.set_state(f"manual_squad_gw{gw}", {"players": ids, "bank": args.bank})
    print(f"Manual squad stored for GW{gw} (bank={args.bank} tenths).")
    return 0


def cmd_set_state(args, cfg):
    db = Db(ROOT / cfg["db_path"])
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    db.set_state(args.key, value)
    print(f"state[{args.key}] = {value}")
    return 0


def cmd_backtest(args, cfg):
    from .backtest import load_season, metrics, replay
    rows = load_season(args.season)
    rules = active_rules(load_rules(ROOT / cfg["rules_path"]))
    base = metrics(replay(rows, rules=[]))
    with_rules = metrics(replay(rows, rules=rules))
    print(f"Season {args.season}: {len(rows)} player-GW rows")
    print("Base model:      ", json.dumps(base))
    print("With active rules:", json.dumps(with_rules))
    return 0


def cmd_validate_rules(args, cfg):
    from .backtest import load_season, validate_rule
    path = ROOT / cfg["rules_path"]
    rules = load_rules(path)
    hypotheses = [r for r in rules if r.get("status") == "hypothesis"]
    if not hypotheses:
        print("No hypotheses to validate.")
        return 0
    rows = load_season(args.season)
    base_rules = active_rules(rules)
    for hyp in hypotheses:
        result = validate_rule(rows, hyp, base_rules=base_rules)
        hyp["validation"] = {**result, "season": args.season,
                             "event": args.event or 0}
        hyp["status"] = "active" if result["improved"] else "rejected"
        print(f"{hyp['id']}: {'PROMOTED' if result['improved'] else 'REJECTED'} "
              f"(pinball {result['baseline'].get('avg_pinball_loss')} -> "
              f"{result['with_rule'].get('avg_pinball_loss')})")
    save_rules(path, rules)
    return 0


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(prog="fpl_copilot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("recommend")
    p.add_argument("--if-due", action="store_true")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("grade")
    p.add_argument("--if-ready", action="store_true")
    p.add_argument("--force", action="store_true")

    sub.add_parser("status")

    p = sub.add_parser("set-squad")
    p.add_argument("players", help="15 comma-separated player ids")
    p.add_argument("--bank", type=int, default=0, help="bank in tenths of a million")

    p = sub.add_parser("set-state")
    p.add_argument("key")
    p.add_argument("value")

    p = sub.add_parser("backtest")
    p.add_argument("--season", default="2025-26")

    p = sub.add_parser("validate-rules")
    p.add_argument("--season", default="2025-26")
    p.add_argument("--event", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = load_config()
    handlers = {
        "recommend": cmd_recommend, "grade": cmd_grade, "status": cmd_status,
        "set-squad": cmd_set_squad, "set-state": cmd_set_state,
        "backtest": cmd_backtest, "validate-rules": cmd_validate_rules,
    }
    return handlers[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
