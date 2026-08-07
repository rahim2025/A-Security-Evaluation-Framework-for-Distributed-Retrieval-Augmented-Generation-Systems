"""
SSM-Score attack via grounding-heuristic exploitation (Grounding-Farming).

Attack Principle
----------------
The Reliable-dRAG orchestrator grants a source "grounded" credit -- and the
resulting positive reliability_delta/usefulness_delta -- using a naive
literal-substring check (drag_llm_service/app/server.py:856-858):

    norm_response = _norm_join(_normalize_for_match(response_text))
    selected_contexts_norm = [_norm_join(_normalize_for_match(c["text"])) for c in selected]
    grounded_by_sources = [norm_response in ctx for ctx in selected_contexts_norm]

This is a relevance/provenance-free check: it only asks whether the LLM's
normalized answer appears as a contiguous token run somewhere inside a
source's returned chunk, regardless of whether that chunk had anything to
do with producing the answer.

A source can exploit this without touching the blockchain and without any
key beyond its own (which it already legitimately uses to sign every
/query response): return broad, high-coverage filler content designed to
contain likely short-answer phrasings, so it gets "grounded" credit -- and
the resulting on-chain score increase -- on queries it did not meaningfully
help answer. Every score-update transaction produced this way is submitted
by the real LLM orchestrator, correctly authorized, individually within the
on-chain rate/delta bounds, and honestly signed; only the *premise* (that
this source's content actually grounded the answer) is false. See
.claude/ssm_grounding_farming_plan.md §3a for why this is not blocked by
the existing on-chain SSM defense (drag_contract/contracts/drag_scores.sol).

Content design (.claude/ssm_grounding_farming_plan.md §4/§4a, empirically
validated): topic/domain alignment is the dominant factor in whether filler
content survives retrieval, far more than answer-type variety or phrasing
polish. Generic "broad across all topics" filler significantly underperforms
filler clustered around the corpus's actual article-topic distribution.

A first version built one generic "background information about X" filler
document per article title. A real attack run (plan §8b) showed this is
unstable: it competes well against broad/introductory-sounding questions but
collapses against the specific-paragraph-level facts that make up most of a
real, diverse question population (95% engagement on one 20-question sample,
2% on a different 50-question sample from the same corpus). This module now
builds several *sub-topic* filler documents per title, each anchored on a
cluster of real keywords drawn from the actual distribution of questions
asked about that title (aggregate word frequency only -- never literal
question or answer text reused as filler content), so a given title's
coverage isn't staked on one generic blurb. Every document still stays
under the retriever's 256-token embedding budget
(sentence-transformers/all-MiniLM-L6-v2) so nothing is silently truncated.
"""
import os
import re
from typing import Dict, List, Optional

import requests

DEFAULT_DATA_SOURCES = [
    {"name": "sources_0",   "url": "http://localhost:8001"},
    {"name": "sources_20",  "url": "http://localhost:8002"},
    {"name": "sources_100", "url": "http://localhost:8003"},
]

# Data-source endpoints (except /health) require this header whenever the
# service has API_KEY set in its environment -- see
# drag_data_source/app/server.py:46-58. docker-compose.yml sets the same
# value for every data source container.
DS_API_KEY = os.getenv("API_KEY", "reliable-derag-secret-2026")

MAX_TOKENS_PER_DOC = 220  # safety margin under the retriever's 256-token limit

# Fallback for titles with no harvested keyword pool (plan §4/§4a): one
# generic, topic-anchored blurb. Kept only as a fallback -- §8b showed this
# alone is unstable across different question samples.
DOMAIN_TEMPLATES = [
    "This passage provides background information related to {t}.",
    "{t} has been discussed extensively in reference works and news coverage.",
]

# Sub-topic anchors (plan §8b): each sentence anchors on a keyword cluster
# drawn from the real distribution of questions asked about a title (never
# literal question/answer text -- aggregate word frequency only), so a
# title's coverage is spread across several distinct facets instead of
# staking everything on one generic "background information" framing.
SUBTOPIC_TEMPLATES = [
    "One aspect of {t} that receives frequent attention involves {k}.",
    "Coverage of {t} often touches on {k}, among other details.",
    "Discussions of {t} have referenced {k} in various contexts.",
    "A recurring theme connected to {t} relates to {k}.",
    "Analyses of {t} sometimes focus on {k}.",
    "Accounts of {t} occasionally center on {k}.",
]

# Generic answer-shaped connector sentences, reused inside every filler doc
# (both the sub-topic and fallback-generic variants) so each one still
# carries literal short-answer substrings for the naive grounding check to
# trip on. Deliberately type-agnostic w.r.t. {f} (§8c bug #3):
# short_answer_fragments mixes every SQuAD answer type (years, names,
# numbers, phrases) indiscriminately, and earlier wording like "the year
# {f}" produced incoherent sentences whenever {f} wasn't actually a year.
# Each still repeats {t} (§8c follow-up fix): a first fix that dropped the
# title from every fragment sentence measured 0/50 -- worse than the 2/50
# baseline -- because mean-pooling then diluted the one on-topic opener
# sentence against several fragment sentences carrying no topical anchor
# at all. Keeping {t} in every sentence restores that anchor.
FRAGMENT_TEMPLATES = [
    "Related information about {t} sometimes cited in this context includes {f}.",
    "Some records mention {f} in connection with {t}.",
    "This aspect of {t} has occasionally been associated with {f} in other sources.",
    "Other accounts reference {f} alongside these details about {t}.",
]


class GroundingFarmingAttack:
    """
    SSM-Score attack via grounding-heuristic exploitation.

    A source injects broad, topic-aligned filler content (via the same
    /poison instrumentation Data Poisoning uses) designed to trip the
    orchestrator's naive substring-match grounding check on many queries,
    farming reliability/usefulness credit it did not earn -- without ever
    touching the blockchain directly or using any key but its own.
    """

    def __init__(
        self,
        target_source: Dict,
        data_sources: Optional[List[Dict]] = None,
        num_titles: int = 10,
        docs_per_title: int = 4,
        keywords_per_doc: int = 4,
        max_tokens_per_doc: int = MAX_TOKENS_PER_DOC,
        llm_service_url: str = "http://localhost:9000",
        blockchain_url: str = "http://localhost:8545",
    ):
        """
        Parameters
        ----------
        target_source      : {"name": ..., "url": ...} of the source to farm.
        data_sources        : registry of all data sources (defaults to the
                              standard 3-source Reliable-dRAG topology).
        num_titles          : number of distinct article titles to cover
                              (bounded above by how many the target's
                              ground-truth corpus actually contains -- §4a).
        docs_per_title      : sub-topic filler documents to build per title
                              (§8b) -- each anchored on a different keyword
                              cluster instead of one generic blurb per title.
        keywords_per_doc    : how many keywords from a title's harvested pool
                              anchor each sub-topic document.
        max_tokens_per_doc  : token budget per filler doc, kept under the
                              retriever's 256-token truncation limit.
        llm_service_url     : URL of the LLM orchestrator.
        blockchain_url      : URL of the Hardhat RPC endpoint (read-only use
                              here -- this attack never signs or submits
                              contract transactions itself).
        """
        self.target_source = target_source
        self.data_sources = data_sources or DEFAULT_DATA_SOURCES
        self.num_titles = num_titles
        self.docs_per_title = docs_per_title
        self.keywords_per_doc = keywords_per_doc
        self.max_tokens_per_doc = max_tokens_per_doc
        self.llm_service_url = llm_service_url
        self.blockchain_url = blockchain_url

        self.filler_docs: List[Dict] = []
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Content construction (plan §4 / §4a)
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from sentence_transformers import SentenceTransformer
            self._tokenizer = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).tokenizer
        return self._tokenizer

    @staticmethod
    def _safe_id(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

    def _pack_sentences(self, tok, title: str, opener: Optional[str], fragment_source) -> Optional[str]:
        """Pack opener + as many fragment sentences as fit under the token budget."""
        sentences: List[str] = []
        if opener is not None:
            if len(tok.encode(opener, add_special_tokens=True)) <= self.max_tokens_per_doc:
                sentences.append(opener)
        for ftpl in FRAGMENT_TEMPLATES:
            sentence = ftpl.format(t=title, f=fragment_source())
            candidate = " ".join(sentences + [sentence])
            if len(tok.encode(candidate, add_special_tokens=True)) > self.max_tokens_per_doc:
                break
            sentences.append(sentence)
        return " ".join(sentences) if sentences else None

    def build_filler_documents(
        self,
        title_counts: Dict[str, int],
        short_answer_fragments: List[str],
        title_keywords: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict]:
        """
        Several sub-topic filler documents per real article title actually
        present in the target's ground-truth corpus (most-represented titles
        first, capped at num_titles). Each document anchors on a cluster of
        `keywords_per_doc` real keywords harvested from the actual question
        distribution for that title (title_keywords, from
        run_grounding_farming.py -- aggregate word frequency only, no
        literal question/answer text reused), so a title's coverage is
        spread across several distinct facets (plan §8b) instead of one
        generic blurb. Titles with no keyword pool available fall back to
        the single generic blurb from the original design (§4/§4a).
        """
        tok = self._get_tokenizer()
        frag_pool = list(dict.fromkeys(short_answer_fragments))
        title_keywords = title_keywords or {}
        top_titles = sorted(title_counts.items(), key=lambda kv: -kv[1])[: self.num_titles]

        frag_i = [0]

        def next_frag():
            if not frag_pool:
                return ""
            f = frag_pool[frag_i[0] % len(frag_pool)]
            frag_i[0] += 1
            return f

        docs: List[Dict] = []
        for title, _count in top_titles:
            keywords = title_keywords.get(title, [])
            if not keywords:
                text = self._pack_sentences(
                    tok,
                    title,
                    DOMAIN_TEMPLATES[0].format(t=title),
                    next_frag,
                )
                if text:
                    docs.append({
                        "id": f"grounding_farm_{self._safe_id(title)}_generic",
                        "text": text,
                        "meta": {"filler": True, "domain_title": title, "subtopic_keywords": []},
                    })
                continue

            chunk = max(1, self.keywords_per_doc)
            buckets = [keywords[i:i + chunk] for i in range(0, len(keywords), chunk)][: self.docs_per_title]
            for b_idx, bucket in enumerate(buckets):
                if not bucket:
                    continue
                if len(bucket) == 1:
                    k_phrase = bucket[0]
                else:
                    k_phrase = ", ".join(bucket[:-1]) + " and " + bucket[-1]
                opener = SUBTOPIC_TEMPLATES[b_idx % len(SUBTOPIC_TEMPLATES)].format(t=title, k=k_phrase)
                text = self._pack_sentences(tok, title, opener, next_frag)
                if not text:
                    continue
                docs.append({
                    "id": f"grounding_farm_{self._safe_id(title)}_{b_idx}",
                    "text": text,
                    "meta": {"filler": True, "domain_title": title, "subtopic_keywords": bucket},
                })

        self.filler_docs = docs
        return docs

    # ------------------------------------------------------------------
    # Injection lifecycle (same /poison, /reset primitive as Data Poisoning)
    # ------------------------------------------------------------------

    def inject(self) -> Dict:
        """POST the built filler documents to the target source's /poison endpoint."""
        if not self.filler_docs:
            raise ValueError("No filler documents built -- call build_filler_documents() first.")
        r = requests.post(
            f"{self.target_source['url']}/poison",
            json={"documents": self.filler_docs},
            headers={"X-API-Key": DS_API_KEY},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def reset(self) -> Dict:
        """
        POST /reset on the target source to remove injected filler and
        restore its clean, on-disk-config document set. This reverses the
        retriever's in-memory state only -- it has no effect on any
        blockchain score changes the farming queries already produced (see
        run_grounding_farming.py and .claude/ssm_grounding_farming_plan.md
        §3a for why those are a separate, not-automatically-reversible
        concern).
        """
        r = requests.post(
            f"{self.target_source['url']}/reset",
            headers={"X-API-Key": DS_API_KEY},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def get_info(self) -> Dict:
        r = requests.get(
            f"{self.target_source['url']}/info",
            headers={"X-API-Key": DS_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
