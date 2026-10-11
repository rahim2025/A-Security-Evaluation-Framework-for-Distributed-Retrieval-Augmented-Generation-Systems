# defense/

One folder per defended attack. Deployed code sits at the top of each folder; everything that only evaluates it goes in subfolders.

```
defense/<name>/
├── README.md        single reference for that defense (design, results, limits, how to run)
├── <module>.py      deployed / importable code (what the Docker services import)
├── evaluation/      validation and before/after scripts (not deployed)   [larger defenses only]
└── legacy_v1/       superseded code kept as a baseline                  [only where it exists]
```

Run output goes to `defense_logs/<name>/` (git-ignored).

| folder | defends against | deployed module(s) | README |
|---|---|---|---|
| `kb_extraction_defense/` | KB extraction (confidentiality) | `extraction_guard.py` | [README](kb_extraction_defense/README.md) |
| `ssm_defense/` | SSM score manipulation (integrity) | `ssm_score_defense.py`, `content_shape_detector.py`, `sentence_length_detector.py`, `grounding_verifier.py` (disabled) | [README](ssm_defense/README.md) |
| `datapoisoning_defense/` | data poisoning (integrity) | `poison_filter.py` | [README](datapoisoning_defense/README.md) |

Imports used by the Docker images (do not move without rebuilding): `defense.datapoisoning_defense.poison_filter`, `defense.ssm_defense.{content_shape_detector,sentence_length_detector,grounding_verifier}`, `defense.kb_extraction_defense.extraction_guard`.
