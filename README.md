# FPL Copilot

A self-calibrating decision agent for Fantasy Premier League. It commits to falsifiable
predictions in public before every deadline, grades its own decision process after the
gameweek, and only adopts rule changes that survive backtest validation.

Not "an FPL recommender." The domain is football; the product is the loop:
**predict → commit → grade → calibrate**.

## How the proof works

Every recommendation is committed to this public repository *before* the FPL deadline.
Git commit timestamps are third-party verified, so predictions cannot be retrofitted.
After each gameweek, a grading report is committed alongside it. The season-long record
lives in `reports/` and is auditable by anyone.

## The loop

1. **Recommend (pre-deadline).** A scheduled job wakes up, checks the next deadline via
   the FPL API (deadlines float: Friday nights, Saturday mornings, midweek rounds), and
   if within the window: snapshots all public data, reconstructs the current squad,
   predicts a points distribution (p10/EV/p90) for every relevant player, evaluates
   transfer options *including rolling the transfer*, picks captain/vice/bench order,
   and writes a reasoned recommendation. The LLM writes the memo; it never produces the
   numbers.
2. **Grade (post-gameweek).** Once bonus points are confirmed (via `event-status`), the
   agent grades itself: prediction range coverage, pinball loss on quantiles, captain
   percentile, transfer delta, bench-order errors, and, critically, performance against
   three naive baselines (do nothing, most-transferred-in move, most-captained pick).
   Process quality and outcome are scored separately: a good decision can blank, a bad
   one can haul.
3. **Calibrate.** The LLM reads the grading report and proposes rule hypotheses as
   structured JSON (condition + adjustment), never prose. A hypothesis only becomes an
   active rule if it improves quantile calibration in backtest. Rules carry confidence
   and decay; stale rules retire.

## Architecture

Deliberately boring: one Python package, SQLite, GitHub Actions cron, markdown reports
committed to the repo. No servers. The scheduler, the proof mechanism, and the hosting
are all the same free tool.

```
FPL public API -> snapshot (immutable, committed) -> features -> quantile predictor
    -> decision engine (transfers/roll, captain, bench) -> LLM memo -> report committed pre-deadline
actuals -> grader (vs baselines, proper scoring rules) -> LLM postmortem -> rule hypotheses
    -> backtest validation -> rule registry -> feeds predictor
```

## Repo layout

```
src/fpl_copilot/    the package (api, db, squad, model, decide, grade, rules, llm, backtest, cli)
reports/            committed recommendations, predictions, and grading reports per GW
data/snapshots/     immutable pre-deadline raw API snapshots
rules/rules.json    the calibration rule registry (hypotheses + active + retired)
.github/workflows/  recommend (every 6h, acts only inside deadline window) and grade (daily)
tests/              unit tests on synthetic data
```

## Setup

1. Create a repo and push this code (public, for the proof mechanism to mean anything).
2. Set `manager_id` in `config.yaml` (re-check after the new season's game opens; IDs can change).
3. Add `ANTHROPIC_API_KEY` as a GitHub Actions secret (Settings → Secrets → Actions).
4. Workflows are on by default. Manual run: Actions tab → recommend → Run workflow.

Local usage:

```
pip install -r requirements.txt && pip install -e .
python -m fpl_copilot.cli status              # next deadline, squad state, last grade
python -m fpl_copilot.cli recommend --force   # produce a recommendation now
python -m fpl_copilot.cli grade --force       # grade the last finished gameweek
python -m fpl_copilot.cli set-squad 1,15,...  # 15 player IDs; needed before GW1 only
python -m fpl_copilot.cli backtest            # calibration metrics on last season's data
python -m fpl_copilot.cli validate-rules      # promote/reject rule hypotheses via backtest
```

## Honest limitations (v1)

- Selling prices are approximated by current price; FPL's 50% sell-on rule on price
  rises is not yet modelled. Affects affordability at the margin.
- Before the GW1 deadline the public API exposes no squad; use `set-squad` once.
- Free-transfer count is tracked internally and can drift; `set-state free_transfers N` corrects it.
- The backtest harness uses the community historical dataset (vaastav/Fantasy-Premier-League)
  and simplifies fixture difficulty; it validates calibration, not final rank.
- One-gameweek chip logic (wildcard/bench boost/etc.) is out of scope for v1; the
  horizon EV (3 GWs) is the only forward-looking element.

## Roadmap

v1.1: selling-price tracking, effective-ownership-aware captaincy, chip windows.
v2: per-position gradient boosted quantile models trained on the historical dataset,
replacing the heuristic predictor behind the same interface.

## Disclaimer

Recommendations only. This tool never logs into or acts on an FPL account.
