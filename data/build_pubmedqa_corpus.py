"""
data/build_pubmedqa_corpus.py

One-off generator: builds data/polluted_token/sources_0.jsonl from
qiaojin/PubMedQA (pqa_labeled config) instead of SQuAD.

Why: attack/Mia_attack's MIA against a SQuAD-derived corpus measured AUC-ROC
consistently BELOW 0.50 across 4 seeds -- not "no signal" but a systematic
inversion, most plausibly because the deployed LLM (Qwen2.5-1.5B-Instruct)
was already pretrained on Wikipedia (SQuAD's source), so it answers
non-member SQuAD questions from memory almost as well as member ones. A
biomedical-abstract corpus the model is far less likely to have memorized
verbatim gives the MIA a fairer test of retrieval-grounding leakage,
decoupled from pretraining-knowledge overlap.

Member/non-member split (same-domain, not cross-domain):
    pqa_labeled has 1,000 rows total, single "train" split.
    First N_CORPUS rows  -> written to sources_0.jsonl (the "loaded" corpus).
    Remaining rows       -> never written anywhere; genuinely disjoint
                             held-out non-members, sampled later by
                             attack/Mia_attack/mia_attack.py directly from
                             HF, exactly mirroring how SQuAD non-members
                             were fixed to be genuinely-unseen-but-same-
                             distribution rather than a different dataset
                             entirely (avoids a cross-domain topic confound
                             that would make any AUC lift ambiguous).

Schema written (matches drag_data_source's expected format exactly):
    {"htmlid": <int>, "html": "<joined context text>"}

The "html" field is the joined PubMedQA context passages
(" ".join(row["context"]["contexts"])) -- attack/Mia_attack/mia_attack.py
must join PubMedQA contexts with this exact same logic so its membership
lookup (string equality against this file's "html" values) matches.

Usage:
    python data/build_pubmedqa_corpus.py
    python data/build_pubmedqa_corpus.py --n_corpus 500 --out data/polluted_token/sources_0.jsonl

This script does NOT touch sources_20.jsonl / sources_100.jsonl -- those
remain SQuAD-derived, token-polluted variants supporting the unrelated
data-poisoning attack (attack/datapoisoning), which is out of scope here.
"""
from __future__ import annotations

import argparse
import json
import os

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
PUBMEDQA_SPLIT = "train"


def join_context(contexts) -> str:
    """Single shared join function -- mia_attack.py must use this exact
    logic when reconstructing PubMedQA context text for membership lookup."""
    return " ".join(c.strip() for c in contexts if c and c.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_corpus", type=int, default=500,
                   help="Number of PubMedQA rows to load into the corpus (default: 500)")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "polluted_token", "sources_0.jsonl"),
                   help="Output JSONL path (default: data/polluted_token/sources_0.jsonl)")
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading {PUBMEDQA_DATASET} ({PUBMEDQA_CONFIG}, {PUBMEDQA_SPLIT} split) ...")
    ds = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split=PUBMEDQA_SPLIT)
    print(f"  {len(ds)} total rows available")

    if args.n_corpus >= len(ds):
        raise ValueError(
            f"--n_corpus={args.n_corpus} must be less than the dataset size ({len(ds)}) "
            "so a genuinely disjoint held-out non-member pool remains."
        )

    seen_contexts = set()
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n_corpus):
            row = ds[i]
            context_text = join_context(row["context"]["contexts"])
            if not context_text or context_text in seen_contexts:
                continue
            seen_contexts.add(context_text)
            f.write(json.dumps({"htmlid": written, "html": context_text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} documents to {args.out}")
    print(f"Rows [0, {args.n_corpus}) reserved as the loaded corpus (members).")
    print(f"Rows [{args.n_corpus}, {len(ds)}) remain as the genuinely disjoint, "
          "same-distribution held-out non-member pool -- sampled directly from "
          "HF by attack/Mia_attack/mia_attack.py, not written anywhere on disk.")


if __name__ == "__main__":
    main()
