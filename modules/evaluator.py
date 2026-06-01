from collections import Counter
import re
from typing import List, Dict

import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from modules.data_types import Testcase


class QAEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary models."""
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        # Initialize semantic similarity model
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Initialize containers for each metric
        self.metrics = {
            'exact_match': [], 'precision': [], 'recall': [], 'f1': [],
            'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [],
            'semantic_similarity': [], 'length_ratio': [], 'length_difference': [],
            'edit_distance': [], 'normalized_edit_distance': [],
            'bigram_overlap': [], 'trigram_overlap': [], 'avg_num_hops': [],
            'avg_num_messages': [], 'avg_query_hit': []
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing articles, punctuation, and extra whitespace."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'[\[\]]', '', text)
        text = ' '.join(text.split())
        return text

    def get_tokens(self, text: str) -> List[str]:
        """Get normalized tokens from text."""
        return word_tokenize(self.normalize_text(text))

    def calculate_basic_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate basic metrics (EM, F1, precision, recall)."""
        pred_tokens = self.get_tokens(prediction)
        ref_tokens = self.get_tokens(reference)

        # Exact match
        em = int(self.normalize_text(prediction) == self.normalize_text(reference))

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {
                "exact_match": em,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        # Token overlap
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(ref_tokens) if ref_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "exact_match": em,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score with smoothing."""
        pred_tokens = self.get_tokens(prediction)
        ref_tokens = [self.get_tokens(reference)]

        smoothing = SmoothingFunction().method1
        try:
            return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        except:
            return 0.0

    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Calculate semantic similarity using sentence embeddings."""
        if not prediction or not reference:
            return 0.0  # Or handle it differently, e.g., log a warning
        # Encode sentences
        try:
            pred_embedding = self.semantic_model.encode(prediction)
            ref_embedding = self.semantic_model.encode(reference)
        except TypeError as e:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(pred_embedding, ref_embedding) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(ref_embedding)
        )
        return float(similarity)

    def calculate_length_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate length-based metrics."""
        pred_tokens = self.get_tokens(prediction)
        ref_tokens = self.get_tokens(reference)

        length_ratio = len(pred_tokens) / len(ref_tokens) if ref_tokens else 0
        length_difference = abs(len(pred_tokens) - len(ref_tokens))

        return {
            "length_ratio": length_ratio,
            "length_difference": length_difference
        }

    def calculate_edit_distance(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate Levenshtein distance and normalized score."""
        distance = Levenshtein.distance(prediction, reference)
        max_len = max(len(prediction), len(reference))
        normalized_distance = 1 - (distance / max_len) if max_len > 0 else 0

        return {
            "edit_distance": distance,
            "normalized_edit_distance": normalized_distance
        }

    def calculate_ngram_overlap(self, prediction: str, reference: str, n: int = 2) -> float:
        """Calculate n-gram overlap between prediction and reference."""

        def get_ngrams(tokens: List[str], n: int) -> set:
            return set(' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

        pred_tokens = self.get_tokens(prediction)
        ref_tokens = self.get_tokens(reference)

        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0

        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        overlap = len(pred_ngrams & ref_ngrams)
        total = len(pred_ngrams | ref_ngrams)

        return overlap / total if total > 0 else 0.0
    
    def add(self, test_case: Testcase):
        # Basic metrics
        basic = self.calculate_basic_metrics(test_case.actual_output, test_case.expected_output)
        for k, v in basic.items():
            self.metrics[k].append(v)

        # BLEU
        self.metrics['bleu'].append(self.calculate_bleu(test_case.actual_output, test_case.expected_output))

        # ROUGE scores
        rouge = self.calculate_rouge(test_case.actual_output, test_case.expected_output)
        for k, v in rouge.items():
            self.metrics[k].append(v)

        # Semantic similarity
        self.metrics['semantic_similarity'].append(
            self.calculate_semantic_similarity(test_case.actual_output, test_case.expected_output)
        )

        # Length metrics
        length = self.calculate_length_metrics(test_case.actual_output, test_case.expected_output)
        for k, v in length.items():
            self.metrics[k].append(v)

        # Edit distance
        edit = self.calculate_edit_distance(test_case.actual_output, test_case.expected_output)
        for k, v in edit.items():
            self.metrics[k].append(v)

        # N-gram overlap
        self.metrics['bigram_overlap'].append(
            self.calculate_ngram_overlap(test_case.actual_output, test_case.expected_output, n=2)
        )
        self.metrics['trigram_overlap'].append(
            self.calculate_ngram_overlap(test_case.actual_output, test_case.expected_output, n=3)
        )

        # Average number of hops
        self.metrics['avg_num_hops'].append(test_case.num_hops)

        # Average number of messages
        self.metrics['avg_num_messages'].append(test_case.num_messages)

        # Average query hit rate
        self.metrics['avg_query_hit'].append(test_case.is_query_hit * 1)


    def get_results(self) -> Dict[str, float]:
        # Calculate means and convert to percentages where appropriate
        results = {}
        for metric, values in self.metrics.items():
            results[metric] = np.mean(values)
        return results
