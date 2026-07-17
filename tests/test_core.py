"""Unit tests on synthetic data (no network)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from fpl_copilot.db import Db, ImmutabilityError
from fpl_copilot.decide import decide, pick_xi, transfer_options
from fpl_copilot.grade import grade_event, pinball_loss, season_scoreboard
from fpl_copilot.model import Predictor
from fpl_copilot.rules import apply_rules, condition_holds


# ---------------------------------------------------------------- fixtures --
def make_element(pid, pos, team, cost=50, points=60, minutes=900, form="4.0",
                 status="a", chance=None, selected="25.0"):
    return {
        "id": pid, "element_type": pos, "team": team, "now_cost": cost,
        "total_points": points, "minutes": minutes, "form": form,
        "status": status, "chance_of_playing_next_round": chance,
        "ep_next": "4.0", "selected_by_percent": selected,
        "web_name": f"P{pid}",
    }


@pytest.fixture
def world():
    """20 teams-lite world: 2 GK, 5 DEF, 5 MID, 3 FWD squad + transfer candidates."""
    elements = []
    pid = 1
    squad = []
    shapes = [(1, 2), (2, 5), (3, 5), (4, 3)]
    for pos, count in shapes:
        for i in range(count):
            elements.append(make_element(pid, pos, team=pid % 10 + 1,
                                         points=40 + pid, minutes=900))
            squad.append(pid)
            pid += 1
    # candidates: better players not in squad
    for pos in (2, 3, 4):
        for i in range(3):
            elements.append(make_element(pid, pos, team=pid % 10 + 1, cost=55,
                                         points=120 + pid, minutes=1000, form="7.5"))
            pid += 1
    bootstrap = {
        "elements": elements,
        "teams": [{"id": t} for t in range(1, 21)],
        "events": [],
    }
    fixtures = []
    # every team plays every event 1..5, alternating home/away, FDR 3
    for event in range(1, 6):
        for t in range(1, 21, 2):
            fixtures.append({
                "event": event, "team_h": t, "team_a": t + 1,
                "team_h_difficulty": 3, "team_a_difficulty": 3,
            })
    return bootstrap, fixtures, squad


# ------------------------------------------------------------------- model --
def test_predictor_blank_gw_scores_zero(world):
    bootstrap, fixtures, squad = world
    pred = Predictor(bootstrap, [f for f in fixtures if f["event"] != 2], rules=[])
    p = pred.predict_event(squad[0], 2)
    assert p["ev"] == 0.0 and p["p90"] == 0.0
    assert p["detail"]["note"] == "blank gameweek"


def test_predictor_quantiles_ordered(world):
    bootstrap, fixtures, squad = world
    pred = Predictor(bootstrap, fixtures, rules=[])
    for pid in squad:
        p = pred.predict_event(pid, 1)
        assert p["p10"] <= p["ev"] <= p["p90"]
        assert p["p10"] >= 0


def test_injured_player_ev_drops(world):
    bootstrap, fixtures, squad = world
    pred = Predictor(bootstrap, fixtures, rules=[])
    healthy = pred.predict_event(squad[3], 1)["ev"]
    bootstrap["elements"][3]["status"] = "i"
    bootstrap["elements"][3]["chance_of_playing_next_round"] = 25
    pred2 = Predictor(bootstrap, fixtures, rules=[])
    injured = pred2.predict_event(squad[3], 1)["ev"]
    assert injured < healthy * 0.5


def test_horizon_ev_positive_and_weighted(world):
    bootstrap, fixtures, squad = world
    pred = Predictor(bootstrap, fixtures, rules=[])
    one = pred.predict_event(squad[5], 1)["ev"]
    horizon = pred.horizon_ev(squad[5], 1, 3)
    assert horizon > one  # more gameweeks add EV
    assert horizon < one * 3  # but with decaying weights


# ------------------------------------------------------------------- rules --
def test_condition_any_all():
    ctx = {"recent_minutes": 50, "fdr": 4}
    assert condition_holds(
        {"any": [{"feature": "recent_minutes", "op": "<", "value": 65}]}, ctx)
    assert not condition_holds(
        {"all": [{"feature": "recent_minutes", "op": "<", "value": 65},
                 {"feature": "fdr", "op": ">", "value": 4}]}, ctx)


def test_apply_rules_only_active_fire():
    rules = [
        {"id": "a", "status": "active",
         "condition": {"feature": "fdr", "op": ">=", "value": 4},
         "adjustment": {"type": "ev_delta", "value": -1.0}},
        {"id": "h", "status": "hypothesis",
         "condition": {"feature": "fdr", "op": ">=", "value": 4},
         "adjustment": {"type": "ev_delta", "value": -5.0}},
    ]
    ev, hit = apply_rules(rules, 5.0, {"fdr": 5})
    assert ev == 4.0 and hit == ["a"]


def test_rule_never_negative():
    rules = [{"id": "big", "status": "active",
              "condition": {"feature": "fdr", "op": ">", "value": 0},
              "adjustment": {"type": "ev_delta", "value": -99}}]
    ev, _ = apply_rules(rules, 2.0, {"fdr": 3})
    assert ev == 0.0


# ------------------------------------------------------------------ decide --
def test_pick_xi_valid_formation(world):
    bootstrap, fixtures, squad = world
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pred = Predictor(bootstrap, fixtures, rules=[])
    squad_preds = [pred.predict_event(pid, 1) for pid in squad]
    xi, bench = pick_xi(squad_preds, elements)
    assert len(xi) == 11 and len(bench) == 4
    pos = [elements[p["player_id"]]["element_type"] for p in xi]
    assert pos.count(1) == 1
    assert pos.count(2) >= 3
    assert pos.count(4) >= 1
    assert bench[0] is not None  # bench GK first


def test_transfer_respects_club_limit_and_budget(world):
    bootstrap, fixtures, squad = world
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pred = Predictor(bootstrap, fixtures, rules=[])
    options = transfer_options(squad, bank=0, predictor=pred, elements=elements,
                               event=1, horizon=3)
    for o in options:
        in_el, out_el = elements[o["in"]], elements[o["out"]]
        assert in_el["element_type"] == out_el["element_type"]
        assert in_el["now_cost"] <= out_el["now_cost"] + 0
        squad_after = set(squad) - {o["out"]} | {o["in"]}
        clubs = {}
        for pid in squad_after:
            clubs[elements[pid]["team"]] = clubs.get(elements[pid]["team"], 0) + 1
        assert max(clubs.values()) <= 3


def test_decide_roll_when_gain_below_threshold(world):
    bootstrap, fixtures, squad = world
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pred = Predictor(bootstrap, fixtures, rules=[])
    squad_info = {"players": squad, "bank": 0}
    d = decide(squad_info, pred, elements, 1, 3, "safe", 1)
    assert d["action"] in ("transfer", "roll")
    if d["action"] == "roll":
        assert d["transfer"] is None and d["roll_rationale"]
    assert d["captain"] != d["vice_captain"]
    assert len(d["xi"]) == 11
    assert d["predicted_points"]["p10"] <= d["predicted_points"]["ev"] \
        <= d["predicted_points"]["p90"]


# ---------------------------------------------------------------------- db --
def test_recommendation_immutability(tmp_path):
    db = Db(tmp_path / "t.db")
    run = db.new_run("recommend", 1)
    db.save_recommendation(7, 1, run, "2026-08-15T17:30:00Z", {"action": "roll"})
    with pytest.raises(ImmutabilityError):
        db.save_recommendation(7, 1, run, "2026-08-15T17:30:00Z", {"action": "transfer"})
    assert db.get_recommendation(7, 1)["payload"]["action"] == "roll"


# ------------------------------------------------------------------- grade --
def test_pinball_loss_basics():
    # actual above the quantile: penalised by tau
    assert pinball_loss(10, 5, 0.9) == pytest.approx(4.5)
    # actual below: penalised by (1 - tau)
    assert pinball_loss(2, 5, 0.9) == pytest.approx(0.3)


def test_grade_event_and_scoreboard(world, tmp_path):
    bootstrap, fixtures, squad = world
    elements = {e["id"]: e for e in bootstrap["elements"]}
    pred = Predictor(bootstrap, fixtures, rules=[])
    squad_info = {"players": squad, "bank": 0}
    decision = decide(squad_info, pred, elements, 1, 3, "balanced", 1)

    rec = {"event": 1, "payload": decision}
    actual_points = {pid: 4 for pid in squad}
    actual_points.update({pid: 4 for pid in decision["xi"]})
    bootstrap_event = {"most_captained": decision["captain"],
                       "most_transferred_in": None}
    g = grade_event(rec, actual_points, None, bootstrap_event, elements)

    assert g["calibration"]["actual_xi_points"] == 4 * 11 + 4  # captain doubled
    assert g["captain"]["beat_crowd_captain"] is True  # same player, ties count
    assert isinstance(g["calibration"]["range_hit"], bool)

    db = Db(tmp_path / "g.db")
    db.save_grading(7, 1, "run1", g)
    sb = season_scoreboard(db.all_gradings(7))
    assert sb["graded_events"] == 1
    assert 0 <= sb["range_coverage"] <= 1
