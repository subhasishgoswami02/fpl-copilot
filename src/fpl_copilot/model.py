"""Heuristic quantile predictor.

Deterministic and interpretable by design: the LLM never touches these numbers.
Outputs a (p10, ev, p90) distribution per player per gameweek, plus a weighted
EV across the transfer horizon. Calibration rules from the registry are applied
as additive EV adjustments on top of the base model.

v2 swaps this module for gradient-boosted quantile models behind the same interface.
"""
import math

# fixture difficulty multiplier (FPL FDR 1..5)
FDR_MULT = {1: 1.20, 2: 1.12, 3: 1.00, 4: 0.88, 5: 0.78}
HOME_ADJ = 1.05
AWAY_ADJ = 0.97

# per-position point volatility (element_type: 1 GK, 2 DEF, 3 MID, 4 FWD)
POS_SIGMA = {1: 2.0, 2: 2.8, 3: 3.2, 4: 3.4}

# weights for horizon EV (this GW, next, next+1, ...)
HORIZON_WEIGHTS = [1.0, 0.8, 0.6, 0.45, 0.35, 0.28]

Z10 = 1.2816  # z for 10th/90th percentile


class Predictor:
    def __init__(self, bootstrap, fixtures, rules=None):
        self.elements = {e["id"]: e for e in bootstrap["elements"]}
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.fixtures = [f for f in fixtures if f.get("event")]
        self.rules = rules or []
        self.histories = {}  # player_id -> list of past-GW dicts (this season)

    def add_history(self, player_id, summary):
        self.histories[player_id] = summary.get("history", [])

    # -- feature helpers -----------------------------------------------------
    def team_fixtures(self, team_id, from_event, n):
        fx = [
            f for f in self.fixtures
            if f["event"] >= from_event and team_id in (f["team_h"], f["team_a"])
        ]
        fx.sort(key=lambda f: f["event"])
        return fx[:n]

    def expected_minutes(self, el):
        chance = el.get("chance_of_playing_next_round")
        status = el.get("status", "a")
        if status in ("i", "s", "u", "n"):
            avail = (chance or 0) / 100.0
        elif status == "d":
            avail = (chance if chance is not None else 75) / 100.0
        else:
            avail = 1.0
        hist = self.histories.get(el["id"], [])
        recent = [h["minutes"] for h in hist[-4:]]
        if recent:
            base = sum(recent) / len(recent)
        elif el.get("minutes", 0) > 0:
            base = 70.0
        else:
            base = 45.0  # unknown/new player with no record yet
        return avail * base

    def points_per_90(self, el):
        season_min = el.get("minutes", 0)
        season_pp90 = (el.get("total_points", 0) / season_min * 90) if season_min else 0.0
        hist = self.histories.get(el["id"], [])
        recent = [h for h in hist[-6:] if h["minutes"] > 0]
        if recent:
            rmin = sum(h["minutes"] for h in recent)
            rpts = sum(h["total_points"] for h in recent)
            recent_pp90 = rpts / max(rmin, 1) * 90
            return 0.6 * recent_pp90 + 0.4 * season_pp90
        return season_pp90

    def _fdr_for(self, fixture, team_id):
        if fixture["team_h"] == team_id:
            return fixture.get("team_h_difficulty", 3), True
        return fixture.get("team_a_difficulty", 3), False

    def _rule_context(self, el, exp_min, fdr, is_home):
        hist = self.histories.get(el["id"], [])
        recent_mins = [h["minutes"] for h in hist[-4:]]
        return {
            "chance_of_playing": el.get("chance_of_playing_next_round")
            if el.get("chance_of_playing_next_round") is not None else 100,
            "recent_minutes": (sum(recent_mins) / len(recent_mins)) if recent_mins else exp_min,
            "expected_minutes": exp_min,
            "fdr": fdr,
            "position": el["element_type"],
            "is_home": 1 if is_home else 0,
        }

    def _apply_rules(self, ev, context):
        from .rules import apply_rules  # local import to avoid cycle
        return apply_rules(self.rules, ev, context)

    # -- prediction ------------------------------------------------------------
    def predict_event(self, player_id, event):
        """One player, one gameweek -> {ev, p10, p90, detail}. Handles blanks/doubles."""
        el = self.elements[player_id]
        gw_fixtures = [
            f for f in self.fixtures
            if f["event"] == event and el["team"] in (f["team_h"], f["team_a"])
        ]
        if not gw_fixtures:
            return {"player_id": player_id, "ev": 0.0, "p10": 0.0, "p90": 0.0,
                    "detail": {"note": "blank gameweek"}}

        exp_min = self.expected_minutes(el)
        pp90 = self.points_per_90(el)

        # cold start (early season): lean on FPL's own ep_next when we lack minutes
        if el.get("minutes", 0) < 180:
            ep_next = float(el.get("ep_next") or 0.0)
            model_ev_1fx = pp90 * (exp_min / 90.0)
            if ep_next > model_ev_1fx:
                pp90 = ep_next / max(exp_min / 90.0, 0.3)

        ev = 0.0
        contexts = []
        for fx in gw_fixtures:
            fdr, is_home = self._fdr_for(fx, el["team"])
            mult = FDR_MULT.get(fdr, 1.0) * (HOME_ADJ if is_home else AWAY_ADJ)
            ev += pp90 * (exp_min / 90.0) * mult
            contexts.append(self._rule_context(el, exp_min, fdr, is_home))

        # rules are applied once per gameweek: average the per-fixture-context deltas
        deltas, rules_hit = [], []
        for ctx in contexts:
            adjusted, hit = self._apply_rules(ev, ctx)
            deltas.append(adjusted - ev)
            rules_hit.extend(hit)
        ev = max(0.0, ev + sum(deltas) / len(deltas))

        sigma = POS_SIGMA.get(el["element_type"], 3.0) * math.sqrt(len(gw_fixtures))
        sigma *= max(exp_min / 90.0, 0.3)
        p10 = max(0.0, ev - Z10 * sigma * 0.85)   # floor: scores are bounded below
        p90 = ev + Z10 * sigma * 1.25             # right skew: hauls happen

        return {
            "player_id": player_id,
            "ev": round(ev, 2),
            "p10": round(p10, 2),
            "p90": round(p90, 2),
            "detail": {
                "exp_minutes": round(exp_min, 1),
                "pp90": round(pp90, 2),
                "fixtures": [
                    {"fdr": self._fdr_for(f, el["team"])[0],
                     "home": self._fdr_for(f, el["team"])[1]} for f in gw_fixtures
                ],
                "rules_applied": list(set(rules_hit)),
            },
        }

    def horizon_ev(self, player_id, from_event, n_gws):
        """Weighted EV across the next n gameweeks."""
        total = 0.0
        for i in range(n_gws):
            w = HORIZON_WEIGHTS[i] if i < len(HORIZON_WEIGHTS) else 0.25
            total += w * self.predict_event(player_id, from_event + i)["ev"]
        return round(total, 2)
