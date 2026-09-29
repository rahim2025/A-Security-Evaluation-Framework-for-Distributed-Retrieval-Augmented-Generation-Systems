"""
data/build_healthcaremagic_corpus.py

One-off generator: builds a corpus JSONL from
lavita/ChatDoctor-HealthCareMagic-100k, in the same shape and with the same
member/non-member split logic as data/build_pubmedqa_corpus.py.

Why: the MIA gap analysis (problems/, reports/MIA_Attack_Fix_Guide.md) flags
PubMedQA's yes/no/maybe label as the structural ceiling on MIA signal --
the fix requires a RAGLeak-style continuation/perplexity probe over a
free-text corpus instead. HealthCareMagic-100k is doctor-authored free-text
answers to real patient questions: same medical domain as PubMedQA (so the
"model already knows this from pretraining" confound stays comparable), but
long-form answer text instead of a closed yes/no/maybe label.

Field mapping (mirrors PubMedQA's question/context split):
    patient "input"  -> probe question  (like PubMedQA's "question")
    doctor  "output" -> document text   (like PubMedQA's "context")

Member/non-member split (identical structure to build_pubmedqa_corpus.py):
    First N_CORPUS rows  -> written to the corpus file (the "loaded" corpus,
                            i.e. members).
    Remaining rows       -> never written anywhere; genuinely disjoint
                             held-out non-members, to be sampled later
                             directly from HF by whatever MIA script consumes
                             this corpus (mirror _load_pubmedqa_pairs's
                             pattern: member iff document text is in the
                             corpus file's "html" set).

Schema written (matches drag_data_source's expected format exactly):
    {"htmlid": <int>, "html": "<doctor's answer text>"}

This script does NOT touch data/polluted_token/sources_0.jsonl (the current
live PubMedQA corpus) or any other existing corpus file. Output defaults to
data/healthcaremagic/sources_0.jsonl, a separate, inactive path -- this
dataset is an option to switch to later, not a replacement made now.

Usage:
    python data/build_healthcaremagic_corpus.py
    python data/build_healthcaremagic_corpus.py --n_corpus 5000 --n_nonmembers 5000

Outputs (all under data/healthcaremagic/ by default):
    sources_0.jsonl     the loaded corpus (members), {"htmlid", "html"}
    members_qa.jsonl    {"question", "context", "answer"} per member document
                        -- the patient question the MIA probes with
    nonmembers_qa.jsonl same shape, for held-out rows NEVER loaded into the
                        corpus; deduped against every member document so no
                        non-member text can match a loaded document.
    Both QA files are written locally so the attack needs no HF download at
    run time and membership is fixed (member iff in sources_0.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os

HEALTHCAREMAGIC_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
HEALTHCAREMAGIC_SPLIT = "train"

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "healthcaremagic", "sources_0.jsonl")

# Raised from 500: the previous pool gave <=500 members/non-members total, too
# few for a stable AUC (CI half-width ~0.1 at n=25+25). 5000 gives the MIA a
# large sampling pool per seed; the corpus stays small enough for one source.
DEFAULT_N_CORPUS = 5000
DEFAULT_N_NONMEMBERS = 5000
# Doctor answers shorter than this are boilerplate ("Hi, thanks for the query")
# and cannot carry any membership signal; drop them from BOTH sides.
MIN_DOC_CHARS = 200


def clean(text: str) -> str:
    """Single shared cleaning function -- any MIA script reconstructing
    HealthCareMagic document text for membership lookup must use this
    exact same logic."""
    return " ".join((text or "").split()).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_corpus", type=int, default=DEFAULT_N_CORPUS,
                   help=f"Number of HealthCareMagic rows to load into the corpus (default: {DEFAULT_N_CORPUS})")
    p.add_argument("--n_nonmembers", type=int, default=DEFAULT_N_NONMEMBERS,
                   help=f"Number of held-out non-member rows to export (default: {DEFAULT_N_NONMEMBERS})")
    p.add_argument("--min_doc_chars", type=int, default=MIN_DOC_CHARS,
                   help=f"Drop answers shorter than this (default: {MIN_DOC_CHARS})")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output JSONL path (default: {DEFAULT_OUT})")
    args = p.parse_args()

    from datasets import load_dataset

    print(f"Loading {HEALTHCAREMAGIC_DATASET} ({HEALTHCAREMAGIC_SPLIT} split) ...")
    ds = load_dataset(HEALTHCAREMAGIC_DATASET, split=HEALTHCAREMAGIC_SPLIT)
    print(f"  {len(ds)} total rows available")

    if args.n_corpus + args.n_nonmembers > len(ds):
        raise ValueError(
            f"--n_corpus ({args.n_corpus}) + --n_nonmembers ({args.n_nonmembers}) exceeds the "
            f"dataset size ({len(ds)}); the member and non-member row ranges must be disjoint."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    out_dir = os.path.dirname(args.out)
    members_qa = os.path.join(out_dir, "members_qa.jsonl")
    nonmembers_qa = os.path.join(out_dir, "nonmembers_qa.jsonl")

    def usable(row):
        doc, q = clean(row["output"]), clean(row["input"])
        return (doc, q) if len(doc) >= args.min_doc_chars and q else None

    seen_docs = set()
    written = 0
    with open(args.out, "w", encoding="utf-8") as f, open(members_qa, "w", encoding="utf-8") as fq:
        for i in range(args.n_corpus):
            pair = usable(ds[i])
            if pair is None or pair[0] in seen_docs:
                continue
            doc_text, question_text = pair
            seen_docs.add(doc_text)
            f.write(json.dumps({"htmlid": written, "html": doc_text}, ensure_ascii=False) + "\n")
            fq.write(json.dumps({"question": question_text, "context": doc_text, "answer": doc_text},
                                ensure_ascii=False) + "\n")
            written += 1

    # Non-members: rows strictly after the member range, deduped against every
    # member document AND each other (HCM has many near-verbatim template answers).
    nm_written = 0
    with open(nonmembers_qa, "w", encoding="utf-8") as fn:
        for i in range(args.n_corpus, args.n_corpus + args.n_nonmembers):
            pair = usable(ds[i])
            if pair is None or pair[0] in seen_docs:
                continue
            doc_text, question_text = pair
            seen_docs.add(doc_text)
            fn.write(json.dumps({"question": question_text, "context": doc_text, "answer": doc_text},
                                ensure_ascii=False) + "\n")
            nm_written += 1

    print(f"Wrote {written} member documents to {args.out} (+ {members_qa})")
    print(f"Wrote {nm_written} held-out non-member documents to {nonmembers_qa}")
    print("Non-members were never loaded into the corpus and share no text with any member.")
    print()
    print("This corpus is NOT active yet -- data/polluted_token/sources_0.jsonl "
          "(PubMedQA) remains the live source. Point the data sources at "
          f"{args.out}, then run the MIA with --dataset healthcaremagic.")

if __name__ == "__main__":
    main()
