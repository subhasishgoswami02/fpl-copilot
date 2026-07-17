"""LLM layer: the analyst, never the oracle.

Two jobs only:
1. write_memo   — turn the deterministic decision JSON into a readable memo
2. postmortem   — read a grading report, write an honest narrative, and propose
                  rule HYPOTHESES as structured JSON (validated later, never trusted)

Falls back to plain templates when ANTHROPIC_API_KEY is absent, so the pipeline
never blocks on the LLM.
"""
import json
import os
import re

MEMO_PROMPT = """You are the written-reasoning layer of an FPL decision agent. The numbers
below come from a deterministic model; you must not change or invent any number.
Write a concise pre-deadline memo (250-400 words) covering: the recommended action
(transfer or roll) and why, captain and vice with the case for each, one risk that
could make this recommendation look bad, and the predicted points range stated as a
falsifiable claim. Plain language, no hype, cite the numbers you were given.

DECISION JSON:
{decision}

PLAYER NAMES:
{names}"""

POSTMORTEM_PROMPT = """You are the postmortem layer of a self-grading FPL decision agent.
Below is this gameweek's grading report and the season scoreboard so far.

Write two things:
1. A short honest narrative (150-250 words): what the process got right and wrong.
   Distinguish process error from variance; a good decision can blank.
2. Zero to two rule hypotheses in a fenced json block: a JSON array where each item is
   {{"id": snake_case_string, "description": string,
     "condition": {{"any"|"all": [{{"feature": one of
       [chance_of_playing, recent_minutes, expected_minutes, fdr, position, is_home],
       "op": one of ["<","<=",">",">=","==","!="], "value": number}}]}},
     "adjustment": {{"type": "ev_delta"|"ev_mult", "value": number}},
     "confidence": 0.0-1.0}}
   Propose a hypothesis only if the evidence suggests a systematic bias, not one bad
   week. An empty array is a perfectly good answer.

GRADING REPORT:
{grading}

SEASON SCOREBOARD:
{scoreboard}"""


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def _call(prompt, model, max_tokens=1500):
    client = _client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception:
        return None


def write_memo(decision, names, model):
    slim = {k: v for k, v in decision.items()
            if k not in ("squad_predictions", "pred_by_id")}
    text = _call(
        MEMO_PROMPT.format(decision=json.dumps(slim, indent=1),
                           names=json.dumps(names)),
        model,
    )
    if text:
        return text.strip()
    # deterministic fallback
    cap = names.get(str(decision["captain"]), decision["captain"])
    rng = decision["predicted_points"]
    if decision["action"] == "transfer" and decision["transfer"]:
        t = decision["transfer"]
        move = (f"Transfer: {names.get(str(t['out']), t['out'])} -> "
                f"{names.get(str(t['in']), t['in'])} "
                f"(+{t['delta_horizon_ev']} horizon EV).")
    else:
        move = f"Roll the transfer. {decision.get('roll_rationale', '')}"
    return (
        f"{move}\nCaptain: {cap}. Vice: "
        f"{names.get(str(decision['vice_captain']), decision['vice_captain'])}.\n"
        f"Predicted XI points (captain doubled): {rng['ev']} "
        f"(80% range {rng['p10']}-{rng['p90']}).\n"
        "(Template memo: no LLM key configured.)"
    )


def postmortem(grading, scoreboard, model):
    text = _call(
        POSTMORTEM_PROMPT.format(grading=json.dumps(grading, indent=1),
                                 scoreboard=json.dumps(scoreboard, indent=1)),
        model, max_tokens=2000,
    )
    if not text:
        hit = grading["calibration"]["range_hit"]
        return {
            "narrative": (
                f"GW{grading['event']}: predicted range "
                f"{'contained' if hit else 'missed'} the actual XI score "
                f"({grading['calibration']['actual_xi_points']} vs "
                f"{grading['calibration']['predicted']['p10']}-"
                f"{grading['calibration']['predicted']['p90']}). "
                "(Template postmortem: no LLM key configured.)"
            ),
            "hypotheses": [],
        }
    hypotheses = []
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                hypotheses = parsed
        except json.JSONDecodeError:
            pass
    narrative = re.sub(r"```json.*?```", "", text, flags=re.DOTALL).strip()
    return {"narrative": narrative, "hypotheses": hypotheses}
