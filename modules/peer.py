from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer

from modules.data_types import Datapoint
from modules.knowledge_base import KnowledgeBase
from modules.llm import LLM


class Peer:
    def __init__(
            self, 
            peer_id: int, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            text_embedding_model: SentenceTransformer = None
        ):
        """
        Initializes a Peer object.

        Args:
            peer_id: A unique identifier for the peer.
            llm_url: The URL of the Large Language Model (LLM) service.
            llm_name: The name/model identifier of the LLM.
            llm_seed: The seed value for the LLM to ensure reproducibility.
            text_embedding_model: The Sentence Transformer model to use for generating embeddings.
        """
        self.peer_id = peer_id
        self.llm = LLM(llm_url, llm_name, llm_num_ctx, llm_seed)
        if text_embedding_model:
            self.knowledge_base = KnowledgeBase(text_embedding_model)
    
    def add_knowledge(self, data_point: Datapoint):
        """
        Adds a new data point to the peer's knowledge base.

        Args:
            data_point: The Datapoint object to be added.
        """
        self.knowledge_base.add(data_point)

    def parse_topic(self, question: str, pre_defined_topics: List[str]) -> Optional[str]:
        """
        Parses the topic of a given question using an LLM.

        Args:
            question: The question to analyze.
            pre_defined_topics: A list of predefined topics to choose from.

        Returns:
            The identified topic from the predefined list, or None if no topic is identified.
        """
        # Construct the prompt for the LLM using a Jinja2 template.
        template_environment = Environment(loader=FileSystemLoader(searchpath="./templates"))
        prompt_template = template_environment.get_template("parse_topic.tmpl")
        llm_prompt = prompt_template.render(question=question, topics=pre_defined_topics)

        # Invoke the LLM to generate a response.
        llm_response = self.llm.generate(llm_prompt)

        # Extract the identified topic from the LLM's response.
        extracted_topic = llm_response.get("topic", None)
        return extracted_topic

    def query(
            self, 
            question: str, 
            query_confidence_threshold: float
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        """
        Queries the peer's knowledge base for an answer to the given question.

        Args:
            question: The question to answer.
            query_confidence_threshold: The minimum confidence score for a relevant knowledge to be considered a hit.

        Returns:
            A tuple containing:
            - The LLM's generated answer (or None if no answer is found).
            - The most relevant knowledge found in the knowledge base (or None if nothing relevant is found).
            - The relevance score of the retrieved knowledge.
            - A boolean indicating whether the query was a hit (True) or a miss (False) based on the confidence 
                threshold.
        """
        # Look up relevant knowledge in the local knowledge base using semantic search.
        top_results = self.knowledge_base.semantic_search(question, 1)
        
        if len(top_results) == 0:
            return None, None, 0.0, False
        
        # Only pick the top-1 data point
        relevant_data_point, relevance_score = top_results[0]

        # Check if the relevance score meets the confidence threshold.
        if relevance_score > query_confidence_threshold:
            # Construct a prompt for the LLM to generate an answer based on the relevant knowledge.
            template_environment = Environment(loader=FileSystemLoader(searchpath="./templates"))
            answer_template = template_environment.get_template("generate_answer.tmpl")
            llm_prompt = answer_template.render(
                question=question, 
                ref_question=relevant_data_point.question, 
                ref_answer=relevant_data_point.answer
            )

            # Invoke the LLM to generate an answer.
            llm_response = self.llm.generate(llm_prompt)

            # Extract the generated answer from the LLM's response.
            generated_answer = llm_response.get("answer", None)
            return generated_answer, relevant_data_point.model_dump_json(), relevance_score, True

        # If the confidence threshold is not met, return None for the answer and indicate a query miss.
        return None, relevant_data_point.model_dump_json(), relevance_score, False

    def query_no_rag(
            self, 
            question: str,
    ) -> Tuple[Optional[str], Optional[str], float, bool]:
        """
        Queries the peer's knowledge base for an answer to the given question.

        Args:
            question: The question to answer.
            query_confidence_threshold: The minimum confidence score for a relevant knowledge to be considered a hit.

        Returns:
            A tuple containing:
            - The LLM's generated answer (or None if no answer is found).
            - The most relevant knowledge found in the knowledge base (or None if nothing relevant is found).
            - The relevance score of the retrieved knowledge.
            - A boolean indicating whether the query was a hit (True) or a miss (False) based on the confidence 
                threshold.
        """

        # Construct a prompt for the LLM to generate an answer based on the relevant knowledge.
        template_environment = Environment(loader=FileSystemLoader(searchpath="./templates"))
        answer_template = template_environment.get_template("generate_answer_no_rag.tmpl")
        llm_prompt = answer_template.render(
            question=question
        )

        # Invoke the LLM to generate an answer.
        llm_response = self.llm.generate(llm_prompt)

        # Extract the generated answer from the LLM's response.
        generated_answer = llm_response.get("answer", None)

        return generated_answer, "", 0.0, False
