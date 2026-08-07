# KB Extraction Defense — Query Diversity Throttle (`kb_extraction_defense`)

Countermeasure for `attack/kb_extraction`'s topic-balanced dataset-question
extraction strategy.

## Why not confidence-score noise injection?

A commonly-referenced design for this attack family proposes injecting
calibrated (Laplace) noise into a per-peer similarity/confidence score to
weaken extraction. That mechanism does not apply to this repo's actual
system: `drag_data_source`'s `/query` returns raw retrieved document text,
and `drag_llm_service`'s `/query` returns only `{"response": text}` — there
is no numeric confidence score anywhere in the client-visible response to
add noise to. `defense/mia_defense/mia_defense.py`'s docstring establishes
this same fact for the sibling membership-inference attack; it applies
identically here.

## What this defends against instead

`drag_data_source/app/server.py` already rate-limits raw request *volume*
via Flask-Limiter (60/min per IP). The extraction attack's recommended
strategy is specifically **topic-balanced** — spreading queries across many
subjects to maximize coverage per query budget (see
`attack/kb_extraction/run_attack.py`'s `topic_coverage` metric) — so an
attacker can stay comfortably under a per-minute cap while still touching
an anomalous number of distinct topics. `QueryDiversityThrottle` adds the
missing signal: not "how many requests," but "how many distinct subjects
has this client asked about recently."

"Topic" here is the real SQuAD article title backing each loaded document
(`context_to_title`, built in `attack/kb_extraction/run_attack.py`) — the
actual corpus on disk carries no separate topic/category field, so this is
the same real signal the attack script itself uses for `topic_coverage`,
not an invented one.

## Mechanism

Per-client sliding window (`window_size`, default 30 most-recent queries).
Once a client has sent at least `min_queries_before_check` (default 15)
queries, if the number of *distinct* topics touched in that window exceeds
`max_topics_per_window` (default 12), the client is flagged and blocked for
`cooldown_queries` (default 20) subsequent queries — a bounded cooldown,
not a permanent ban, so a legitimate client whose early queries happened to
be unusually varied isn't locked out forever.

## Running

```bash
docker compose up -d
python defense/kb_extraction_defense/run_defense.py
python defense/kb_extraction_defense/run_defense.py --source_index 1 --max_topics_per_window 8
```

Runs the same real probe set through `attack.kb_extraction.run_attack.probe_source()`
twice against the real live data source — once ungated (`attack_only`),
once gated through the throttle (`attack_plus_defense`, via `probe_source`'s
`query_gate` hook) — and reports the reduction in `extraction_rate` and
`topic_coverage`, plus how many queries were blocked. Logs to
`defense_logs/kb_extraction_defense/`.

## Tuning

| Parameter | Too low | Too high |
|---|---|---|
| `max_topics_per_window` | flags legitimate clients with normally varied interests | never catches a topic-balanced sweep before it completes |
| `min_queries_before_check` | flags a client on too small a sample | slow to react to a real sweep |
| `cooldown_queries` | attacker re-evaluated (and can resume) almost immediately | a flagged legitimate client is denied service for a long time |
| `window_size` | short memory — an attacker can pace queries just below the threshold, let the window slide past old topics, and keep going indefinitely | slow to forget a legitimate client's genuinely broad history, more false positives over a long session |

## Honesty note

`min_queries_before_check`/`window_size`/`cooldown_queries` are reasonable
defaults, not empirically tuned against a measured "legitimate client"
baseline the way `defense/sfa_sim_defense`'s thresholds were calibrated
against a measured honest response rate. Treat this as a working starting
point, not a validated production configuration — measure your own
deployment's real per-client topic-diversity distribution before relying on
these numbers to distinguish attacker from legitimate user with confidence.
