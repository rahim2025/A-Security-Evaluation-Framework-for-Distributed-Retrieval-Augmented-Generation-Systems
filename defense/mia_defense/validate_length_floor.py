"""
defense/mia_defense/validate_length_floor.py

Answers problems/mia_attack_gaps.md #4's length-normalization-floor item:
calibrate_target_length()'s `floor=100` was added after a live run measured
the deployed LLM returning extremely terse responses (as short as 10 chars,
e.g. a bare "yes"/"no" with a leaked chat-template role token) that collapsed
the *measured* median to a degenerate ~10 chars across all three tested
seeds -- see that function's own docstring. That leaked-role-token bug is
now fixed and verified (see problems/fixed/ddos_fixed.md sec 5 /
mia_fixed.md sec 3): 3 independent llm-service process restarts, same
questions, byte-identical clean output. This asks the natural follow-up: now
that the underlying corruption is fixed, does calibrate_target_length()
still need the floor to avoid collapsing, or does the *measured* median
naturally land in a reasonable range on its own?

Collects real response lengths from a live member+non-member probe sample
and reports calibrate_target_length()'s output with the floor at its
current value (100), at 0 (effectively disabled), and the raw measured
median -- so the floor's continued necessity is measured, not assumed.
"""
from __future__ import annotations

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.Mia_attack.mia_attack import (  # noqa: E402
    LLM_SERVICE_URL, CORPUS_JSONL, load_membership_documents, _query_llm,
)
from defense.mia_defense.mia_defense import calibrate_target_length  # noqa: E402

N_MEMBERS = 25
N_NONMEMBERS = 25
SEED = 5001  # fresh, not previously used in this project's dev/test seed history


def main() -> None:
    members, non_members = load_membership_documents(N_MEMBERS, N_NONMEMBERS, SEED, CORPUS_JSONL, probes_per_doc=1)
    responses = []
    for label, docs in (("MEMBER", members), ("NON-MEMBER", non_members)):
        for i, qa_list in enumerate(docs, 1):
            q = qa_list[0]["question"]
            resp = _query_llm(q, LLM_SERVICE_URL, api_key="")
            responses.append(resp)
            print(f"  [{label}] doc {i:>2}/{len(docs)}  len={len(resp):>4}  resp={resp[:50]!r}")

    lengths = [len(r) for r in responses if r and r.strip()]
    print(f"\nn_responses_nonempty = {len(lengths)} / {len(responses)}")
    print(f"min={min(lengths)}  median={statistics.median(lengths)}  "
          f"mean={statistics.mean(lengths):.1f}  max={max(lengths)}")

    target_with_floor = calibrate_target_length(responses, fallback=150, floor=100)
    target_no_floor = calibrate_target_length(responses, fallback=150, floor=0)
    print(f"\ncalibrate_target_length(floor=100) -> {target_with_floor}")
    print(f"calibrate_target_length(floor=0)   -> {target_no_floor}  (floor disabled)")
    if target_with_floor == target_no_floor:
        print("\n-> Floor made NO difference on this sample: the measured median already "
              "clears 100 chars on its own now that the leaked-role-token bug is fixed. "
              "The floor is a no-op safety net here, not a load-bearing correction.")
    else:
        print(f"\n-> Floor changed the result ({target_no_floor} -> {target_with_floor}): "
              "still measurably necessary even post-fix.")


if __name__ == "__main__":
    main()
