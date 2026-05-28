"""Data poisoning utilities for FedRAG."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Sequence

from fed_rag.exceptions import MissingExtraError


@dataclass(frozen=True)
class DataPoisoningResult:
    """Results from applying a data poisoning attack."""

    poisoned_examples: list[dict[str, Any]]
    num_poisoned: int
    num_original: int
    poisoning_ratio: float
    poisoned_indices: list[int]
    mode: str
    poison_type: str
    amplification_factor: int
    question_variants: int


class DataPoisoningAttack:
    """Simple data poisoning attack for RAG fine-tuning datasets.

    This mirrors the DRAG data poisoning idea, but operates on a single client
    dataset by injecting poisoned (query, response) examples.
    """

    def __init__(
        self,
        poisoning_ratio: float = 0.1,
        poison_type: str = "wrong_answer",
        mode: str = "append",
        amplification_factor: int = 1,
        question_variants: int = 1,
        seed: int | None = None,
        query_key: str = "query",
        response_key: str = "response",
        topic_key: str = "topic",
        fallback_query_keys: Iterable[str] = ("question",),
        fallback_response_keys: Iterable[str] = ("answer",),
        wrong_answer_text: str = "POISONED: This is incorrect information.",
    ) -> None:
        self.poisoning_ratio = poisoning_ratio
        self.poison_type = poison_type
        self.mode = mode
        self.amplification_factor = max(1, int(amplification_factor))
        self.question_variants = max(1, int(question_variants))
        self.seed = seed
        self.query_key = query_key
        self.response_key = response_key
        self.topic_key = topic_key
        self.fallback_query_keys = tuple(fallback_query_keys)
        self.fallback_response_keys = tuple(fallback_response_keys)
        self.wrong_answer_text = wrong_answer_text

    def execute(
        self, examples: Sequence[dict[str, Any]]
    ) -> DataPoisoningResult:
        """Apply poisoning to a list of example dicts.

        Returns a DataPoisoningResult containing the poisoned examples and
        summary metadata.
        """

        rng = random.Random(self.seed)
        ratio = min(max(self.poisoning_ratio, 0.0), 1.0)
        base_examples = [dict(example) for example in examples]

        if not base_examples or ratio == 0.0:
            return DataPoisoningResult(
                poisoned_examples=base_examples,
                num_poisoned=0,
                num_original=len(base_examples),
                poisoning_ratio=ratio,
                poisoned_indices=[],
                mode=self.mode,
                poison_type=self.poison_type,
                amplification_factor=self.amplification_factor,
                question_variants=self.question_variants,
            )

        num_to_poison = max(1, int(len(base_examples) * ratio))
        num_to_poison = min(num_to_poison, len(base_examples))
        poisoned_indices = rng.sample(range(len(base_examples)), num_to_poison)

        answers_by_topic, all_answers = self._index_answers(base_examples)

        poisoned_records: list[dict[str, Any]] = []
        for idx in poisoned_indices:
            original = base_examples[idx]
            poisoned_variants = self._build_poisoned_variants(
                original,
                answers_by_topic=answers_by_topic,
                all_answers=all_answers,
                rng=rng,
            )

            if self.mode == "replace":
                base_examples[idx] = poisoned_variants[0]
            else:
                poisoned_records.extend(poisoned_variants)

        if self.mode == "append":
            base_examples.extend(poisoned_records)
        elif self.mode != "replace":
            raise ValueError(
                "Invalid mode. Expected 'append' or 'replace'."
            )

        return DataPoisoningResult(
            poisoned_examples=base_examples,
            num_poisoned=len(poisoned_records)
            if self.mode == "append"
            else len(poisoned_indices),
            num_original=len(examples),
            poisoning_ratio=ratio,
            poisoned_indices=poisoned_indices,
            mode=self.mode,
            poison_type=self.poison_type,
            amplification_factor=self.amplification_factor,
            question_variants=self.question_variants,
        )

    def execute_hf_dataset(self, dataset: Any) -> tuple[Any, DataPoisoningResult]:
        """Apply poisoning to a HuggingFace Dataset.

        Returns (poisoned_dataset, result).
        """

        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            msg = (
                "`execute_hf_dataset` requires the `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a datasets.Dataset instance")

        examples = dataset.to_list()
        result = self.execute(examples)
        poisoned_dataset = Dataset.from_list(result.poisoned_examples)
        return poisoned_dataset, result

    def _build_poisoned_variants(
        self,
        original: dict[str, Any],
        *,
        answers_by_topic: dict[str, list[str]],
        all_answers: list[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        query_key = self._resolve_key(
            original, self.query_key, self.fallback_query_keys
        )
        response_key = self._resolve_key(
            original, self.response_key, self.fallback_response_keys
        )

        question = original[query_key]
        poisoned_answer = self._poison_answer(
            original,
            response_key=response_key,
            answers_by_topic=answers_by_topic,
            all_answers=all_answers,
            rng=rng,
        )

        variants: list[dict[str, Any]] = []
        for variant_question in self._question_variants(
            question, self.question_variants
        ):
            poisoned = dict(original)
            poisoned[query_key] = variant_question
            poisoned[response_key] = poisoned_answer
            for _ in range(self.amplification_factor):
                variants.append(dict(poisoned))

        return variants

    def _question_variants(self, question: str, count: int) -> list[str]:
        base_variants = [
            question,
            f"{question} ?",
            question.replace("?", ""),
            question.replace("What", "what").replace("Which", "which"),
        ]
        if count <= len(base_variants):
            return base_variants[:count]

        variants = list(base_variants)
        while len(variants) < count:
            variants.append(base_variants[len(variants) % len(base_variants)])
        return variants

    def _poison_answer(
        self,
        original: dict[str, Any],
        *,
        response_key: str,
        answers_by_topic: dict[str, list[str]],
        all_answers: list[str],
        rng: random.Random,
    ) -> str:
        if self.poison_type == "wrong_answer":
            return self.wrong_answer_text

        if self.poison_type == "misleading":
            return (
                "The correct answer is the opposite of "
                f"{original[response_key]}"
            )

        if self.poison_type == "noise":
            noise = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10))
            return f"{original[response_key]} {noise}"

        if self.poison_type == "answer_swap":
            topic = original.get(self.topic_key)
            if topic and topic in answers_by_topic:
                candidates = [
                    a
                    for a in answers_by_topic[topic]
                    if a != original[response_key]
                ]
                if candidates:
                    return rng.choice(candidates)
            fallback = [a for a in all_answers if a != original[response_key]]
            return rng.choice(fallback) if fallback else self.wrong_answer_text

        return self.wrong_answer_text

    def _index_answers(
        self, examples: Sequence[dict[str, Any]]
    ) -> tuple[dict[str, list[str]], list[str]]:
        answers_by_topic: dict[str, list[str]] = {}
        all_answers: list[str] = []

        for example in examples:
            response_key = self._resolve_key(
                example, self.response_key, self.fallback_response_keys
            )
            answer = example.get(response_key)
            if answer is None:
                continue
            all_answers.append(answer)
            topic = example.get(self.topic_key)
            if topic is None:
                continue
            answers_by_topic.setdefault(topic, []).append(answer)

        return answers_by_topic, all_answers

    def _resolve_key(
        self,
        example: dict[str, Any],
        primary_key: str,
        fallbacks: Iterable[str],
    ) -> str:
        if primary_key in example:
            return primary_key
        for key in fallbacks:
            if key in example:
                return key
        raise KeyError(
            f"Missing expected key '{primary_key}' (fallbacks: {list(fallbacks)})"
        )
