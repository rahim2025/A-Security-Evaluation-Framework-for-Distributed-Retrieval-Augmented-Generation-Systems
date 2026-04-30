from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import torch

from modules.data_types import Datapoint

class KnowledgeBase:
    """
    Knowledge Base with Integrated Semantic Search and Embedding Caching (without Faiss):
        Combines data storage, embedding generation, and cosine similarity search using PyTorch for efficient vector operations.

        > Reimers, N. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
        arXiv preprint arXiv:1908.10084 (2019).
    """

    def __init__(self, text_embedding_model: SentenceTransformer):
        """
        Initializes the KnowledgeBase.

        Args:
            text_embedding_model: The Sentence Transformer model to use for generating embeddings.
        """
        self.data_points: List[Datapoint] = []
        self.text_embedding_model = text_embedding_model
        self.device = text_embedding_model.device
        self.embeddings = None  # Cache for storing embeddings

    def add(self, data_point: Datapoint):
        """
        Adds a new data point to the knowledge base and generates its embedding.

        Args:
            data_point: The Datapoint object to add.
        """
        self.data_points.append(data_point)

        # Format the data point for embedding generation.
        formatted_entry = f"{data_point.question}"

        # Generate embedding for the new data point.
        new_embedding = self.text_embedding_model.encode(
            [formatted_entry],
            convert_to_tensor=True,
            device=self.device
        )

        if self.embeddings is None:
            # Initialize embedding cache
            self.embeddings = new_embedding
        else:
            # Concatenate new embedding to the existing cache
            self.embeddings = torch.cat([self.embeddings, new_embedding], dim=0)

    def semantic_search(self, query: str, top_k: int = 1) -> List[Tuple[Datapoint, float]]:
        """
        Performs a semantic search to find relevant knowledge within the knowledge base.

        Args:
            query: The question or search query.
            top_k: The number of top results to return (defaults to 1).

        Returns:
            A list of tuples containing:
            - The most relevant knowledge found (as a formatted string).
            - The relevance score (a float between 0.0 and 1.0).
        """
        if not self.data_points:
            return []

        # Generate embedding for the query.
        query_embedding = self.text_embedding_model.encode(
            [query],
            convert_to_tensor=True,
            device=self.device
        )

        # Calculate cosine similarity between the query and all data point embeddings.
        similarities = util.cos_sim(query_embedding, self.embeddings)
        similarities = similarities.squeeze()  # Remove extra dimension

        # Get top-k results using torch operations
        top_k = min(top_k, len(self.data_points))  # Ensure top_k doesn't exceed list length
        top_scores, top_indices = similarities.topk(top_k)

        # Ensure top_scores and top_indices are lists even when top_k=1
        if top_k == 1:
            top_scores = [top_scores.item()]
            top_indices = [top_indices.item()]
        else:
            top_scores = top_scores.tolist()
            top_indices = top_indices.tolist()

        # Create a list of (sentence, score) tuples
        top_results = []
        for idx, score in zip(top_indices, top_scores):
            relevant_data_point = self.data_points[idx]
            relevance_score = float(score)
            top_results.append((relevant_data_point, relevance_score))

        return top_results
