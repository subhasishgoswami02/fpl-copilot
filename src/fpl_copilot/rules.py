"""Calibration rule registry.

Rules are structured hypotheses, never prose. Lifecycle:

    hypothesis (proposed by LLM postmortem or by hand)
        -> active   (only after backtest validation improves calibration)
        -> retired  (manually, on decay, or when re-validation fails)

Rule shape:
{
  "id": "minutes_risk_penalty",
  "description": "...",
  "condition": {"any": [{"feature": "recent_minutes", "op": "<", "value": 65}]},
  "adjustment": {"type": "ev_delta", "value": -0.5},
  "status": "active",
  "confidence": 0.7,
  "created_from": "seed",
  "created_utc": "...",
  "validation": {...}          # backtest evidence, filled by validate-rules
}
"""
import datetime
import json
import pathlib

OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def load_rules(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def save_rules(path, rules):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rules, indent=2))


def active_rules(rules):
    return [r for r in rules if r.get("status") == "active"]


def _check_clause(clause, context):
    feature = clause.get("feature")
    if feature not in context:
        return False
    op = OPS.get(clause.get("op"))
    if op is None:
        return False
    try:
        return op(context[feature], clause.get("value"))
    except TypeError:
        return False


def condition_holds(condition, context):
    if "any" in condition:
        return any(_check_clause(c, context) for c in condition["any"])
    if "all" in condition:
        return all(_check_clause(c, context) for c in condition["all"])
    return _check_clause(condition, context)


def apply_rules(rules, ev, context):
    """Return (adjusted_ev, [rule ids that fired]). Only active rules fire."""
    hit = []
    adjusted = ev
    for rule in active_rules(rules):
        if condition_holds(rule.get("condition", {}), context):
            adj = rule.get("adjustment", {})
            if adj.get("type") == "ev_delta":
                adjusted += float(adj.get("value", 0.0))
            elif adj.get("type") == "ev_mult":
                adjusted *= float(adj.get("value", 1.0))
            hit.append(rule["id"])
    return max(0.0, adjusted), hit


def add_hypotheses(path, hypotheses, created_from):
    """Append LLM-proposed hypotheses to the registry with status=hypothesis."""
    rules = load_rules(path)
    existing = {r["id"] for r in rules}
    added = []
    for h in hypotheses:
        if not isinstance(h, dict) or "condition" not in h or "adjustment" not in h:
            continue
        rid = h.get("id") or f"hyp_{len(rules) + len(added) + 1}"
        if rid in existing:
            continue
        rules.append({
            "id": rid,
            "description": h.get("description", ""),
            "condition": h["condition"],
            "adjustment": h["adjustment"],
            "status": "hypothesis",
            "confidence": float(h.get("confidence", 0.5)),
            "created_from": created_from,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        })
        added.append(rid)
    save_rules(path, rules)
    return added


def decay_rules(path, max_age_events=10, current_event=None):
    """Retire active rules not re-validated within max_age_events."""
    rules = load_rules(path)
    changed = False
    for r in rules:
        if r.get("status") != "active" or current_event is None:
            continue
        validated_at = (r.get("validation") or {}).get("event", 0)
        if current_event - validated_at > max_age_events:
            r["status"] = "retired"
            r["retired_reason"] = "decay"
            changed = True
    if changed:
        save_rules(path, rules)
    return rules
