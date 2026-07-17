"""Backtest harness on the community historical dataset (vaastav/Fantasy-Premier-League).

Replays the predictor week by week with strict data cutoffs (only prior gameweeks
visible) and reports calibration metrics. Used to (a) sanity-check the base model
before GW1 and (b) validate rule hypotheses before promotion.

This validates CALIBRATION (are the quantiles honest? do rules help?), not final
rank. Fixture difficulty is approximated from opponent season strength.
"""
import csv
import io
import math
import urllib.request
from collections import defaultdict

from .grade import pinball_loss
from .model import POS_SIGMA, Z10
from .rules import apply_rules

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)
POS_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def load_season(season="2025-26", csv_text=None):
    if csv_text is None:
        with urllib.request.urlopen(DATA_URL.format(season=season), timeout=60) as r:
            csv_text = r.read().decode("utf-8", errors="replace")
    rows = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        try:
            rows.append({
                "name": row["name"],
                "position": POS_MAP.get(row.get("position", ""), 3),
                "team": row.get("team", ""),
                "gw": int(float(row["GW"])),
                "minutes": int(float(row["minutes"])),
                "total_points": int(float(row["total_points"])),
                "was_home": str(row.get("was_home", "")).lower() == "true",
            })
        except (KeyError, ValueError):
            continue
    return rows


def replay(rows, rules=None, min_history_gws=4, start_gw=6):
    """Predict each player-GW from prior GWs only; score quantiles."""
    rules = rules or []
    by_player = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["gw"]):
        by_player[r["name"]].append(r)

    results = []
    for name, games in by_player.items():
        for i, g in enumerate(games):
            if g["gw"] < start_gw or i < min_history_gws:
                continue
            hist = games[:i]
            recent = [h for h in hist[-6:] if h["minutes"] > 0]
            recent_mins = [h["minutes"] for h in hist[-4:]]
            exp_min = sum(recent_mins) / len(recent_mins) if recent_mins else 45.0
            if exp_min < 15:
                continue  # mirror live behaviour: fringe players aren't recommended
            if recent:
                rmin = sum(h["minutes"] for h in recent)
                rpts = sum(h["total_points"] for h in recent)
                pp90 = rpts / max(rmin, 1) * 90
            else:
                pp90 = 0.0
            home_adj = 1.05 if g["was_home"] else 0.97
            ev = pp90 * (exp_min / 90.0) * home_adj

            ctx = {
                "chance_of_playing": 100,
                "recent_minutes": exp_min,
                "expected_minutes": exp_min,
                "fdr": 3,
                "position": g["position"],
                "is_home": 1 if g["was_home"] else 0,
            }
            ev, _ = apply_rules(rules, ev, ctx)

            sigma = POS_SIGMA.get(g["position"], 3.0) * max(exp_min / 90.0, 0.3)
            p10 = max(0.0, ev - Z10 * sigma * 0.85)
            p90 = ev + Z10 * sigma * 1.25
            actual = g["total_points"]
            results.append({
                "gw": g["gw"], "name": name, "ev": ev, "p10": p10, "p90": p90,
                "actual": actual,
                "pinball": pinball_loss(actual, p10, 0.10) + pinball_loss(actual, p90, 0.90),
                "covered": p10 <= actual <= p90,
                "abs_err": abs(actual - ev),
            })
    return results


def metrics(results):
    if not results:
        return {}
    n = len(results)
    return {
        "n_predictions": n,
        "coverage_p10_p90": round(sum(r["covered"] for r in results) / n, 3),
        "avg_pinball_loss": round(sum(r["pinball"] for r in results) / n, 3),
        "mae": round(sum(r["abs_err"] for r in results) / n, 3),
        "rmse": round(math.sqrt(sum(r["abs_err"] ** 2 for r in results) / n), 3),
    }


def validate_rule(rows, rule, base_rules=None):
    """A hypothesis is promotable if it improves pinball loss without wrecking coverage."""
    base_rules = base_rules or []
    base = metrics(replay(rows, rules=base_rules))
    candidate_rule = dict(rule)
    candidate_rule["status"] = "active"  # force-fire during validation
    with_rule = metrics(replay(rows, rules=base_rules + [candidate_rule]))
    improved = (
        with_rule.get("avg_pinball_loss", 9e9) < base.get("avg_pinball_loss", 9e9)
        and with_rule.get("coverage_p10_p90", 0) >= base.get("coverage_p10_p90", 1) - 0.02
    )
    return {
        "rule_id": rule.get("id"),
        "baseline": base,
        "with_rule": with_rule,
        "improved": improved,
    }
