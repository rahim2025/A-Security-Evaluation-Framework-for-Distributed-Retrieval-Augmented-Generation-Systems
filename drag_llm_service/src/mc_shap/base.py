# base.py

from typing import Optional, List, Dict, Tuple, Union, Any, Callable, Set
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import random
from itertools import combinations
import torch
from queue import Queue

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import VLLMModel for LocalModel
try:
    from src.models.open_model import VLLMModel
except ImportError:
    # Handle case where import fails
    VLLMModel = None

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

class TextVectorizer:
    """Base class for text vectorization"""
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ModelBase(ABC):
    """Base class for all models (text and vision)"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.client = None
        
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from model"""
        pass

class HuggingFaceEmbeddings(TextVectorizer):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace sentence embeddings vectorizer - much simpler implementation
        
        Args:
            model_name: Name of the sentence-transformer model from HuggingFace
            device: Device to run model on ('cpu' or 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load model - SentenceTransformer handles all the complexity
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers - much simpler"""
        if not self.model:
            self._initialize_model()
            
        # SentenceTransformer handles batching, padding, etc. automatically
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Sentence-transformers models already return normalized vectors
        return np.dot(comparison_vectors, base_vector)

class OpenAIEmbeddings(TextVectorizer):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings vectorizer
        
        Args:
            api_key: OpenAI API key
            model: Embeddings model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        if not self.client:
            self._initialize_client()
            
        # Process texts in batches to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                # Extract embeddings from response
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise Exception(f"Error getting embeddings from OpenAI: {str(e)}")
                
        return np.array(all_embeddings)
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Normalize vectors
        base_norm = np.linalg.norm(base_vector)
        comparison_norms = np.linalg.norm(comparison_vectors, axis=1)
        
        # Avoid division by zero
        if base_norm == 0 or np.any(comparison_norms == 0):
            return np.zeros(len(comparison_vectors))
            
        normalized_base = base_vector / base_norm
        normalized_comparisons = comparison_vectors / comparison_norms[:, np.newaxis]
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_comparisons, normalized_base)
        return similarities

class TfidfTextVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = None
        
    def vectorize(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer().fit(texts) 
        return self.vectorizer.transform(texts).toarray()
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            base_vector.reshape(1, -1), comparison_vectors
        ).flatten()

class OpenAIModel(ModelBase):
    """Generic AI Model API wrapper supporting multiple providers"""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        :param model_name: Name of the model (e.g., "gpt-4-turbo", "gemini-2.0-flash", "claude-3").
        :param api_key: API key for the respective provider.
        :param base_url: Optional base URL (e.g., Gemini, Claude, OpenAI custom endpoint).
        """
        super().__init__(model_name, api_key=api_key)
        self.base_url = base_url
        self.system_prompt = system_prompt
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the API client with the given base_url."""
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url else OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")

    def generate(self, prompt: str) -> str:
        """Generates text based on a prompt with optional vision support."""
        if not self.client:
            self._initialize_client()

        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

class LocalModel(ModelBase):
    """Local model implementation supporting text using HuggingFace models"""
    
    def __init__(self, 
                model_name: str,
                system_setting: str=None,
                temperature: str=0.0,
                seed: int=42):
        """
        Initialize local model
        
        Args:
            model_name: HuggingFace model name/path
            model_type: "text"
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            device: Device to run model on ("auto", "cuda", "cpu")
            torch_dtype: Torch data type for model
            **model_kwargs: Additional kwargs for model initialization
        """
        super().__init__(model_name)
        self.temperature = temperature
        self.system_setting = system_setting
        self.seed = seed

        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate model and tokenizer/processor"""
        if VLLMModel is None:
            raise ImportError(
                "vllm is not installed. Rebuild the Docker image with INSTALL_VLLM=true "
                "or install vllm locally to use LocalModel."
            )

        self.model = VLLMModel(
            model_handle=self.model_name,
            system_setting=self.system_setting,
            temperature=self.temperature,
            seed=self.seed
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Text prompt

        Returns:
            Generated text response
        """
        try:
            response = self.model(prompt)
            return response
        finally:
            self.model.restart()

class BaseSHAP(ABC):
    """Base class for SHAP implementations"""
    
    def __init__(self, 
                 model: ModelBase,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False,
                 progress_queue: Optional[Queue] = None):
        self.model = model
        self.vectorizer = vectorizer
        self.debug = debug
        self.results_df = None
        self.shapley_values = None
        self.progress_queue = progress_queue

    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(message)

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """Calculate baseline model response"""
        return self.model.generate(**self._prepare_generate_args(content, **kwargs))

    @abstractmethod
    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        pass

    def _generate_random_combinations(self, 
                                    samples: List[Any], 
                                    k: int, 
                                    exclude_combinations_set: Set[Tuple[int, ...]]) -> List[Tuple[List, Tuple[int, ...]]]:
        """
        Generate random combinations efficiently using binary representation
        """
        n = len(samples)
        sampled_combinations_set = set()
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))

        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_combination(self, 
                                content: Any, 
                                sampling_ratio: float,
                                max_combinations: Optional[int] = 1000) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        """
        Get model responses for combinations
        
        Args:
            content: Content to analyze
            sampling_ratio: Ratio of non-essential combinations to sample (0-1)
            max_combinations: Maximum number of combinations (must be >= n for n tokens)
        """
        samples = self._get_samples(content)
        n = len(samples)
        self._debug_print(f"Number of samples: {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        # Always start with essential combinations (each missing one sample)
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)
        
        num_essential = len(essential_combinations)
        self._debug_print(f"Number of essential combinations: {num_essential}")
        if max_combinations is not None and max_combinations < num_essential:
            print(f"Warning: max_combinations ({max_combinations}) is less than the number of essential combinations "
                  f"({num_essential}). Will use all essential combinations despite the limit.")
            self._debug_print("No additional combinations will be added.")
            max_combinations = num_essential
        # Calculate how many additional combinations we can/should generate
        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)
            self._debug_print(f"Remaining combinations budget after essentials: {remaining_budget}")

        # If using sampling ratio, calculate possible additional combinations without generating them
        if sampling_ratio < 1.0:
            # Get theoretical number of total combinations
            theoretical_total = 2 ** n - 1
            theoretical_additional = theoretical_total - num_essential
            # Calculate desired number based on ratio
            desired_additional = int(theoretical_additional * sampling_ratio)
            # Take minimum of sampling ratio and max_combinations limits
            num_additional = min(desired_additional, remaining_budget)
        else:
            num_additional = remaining_budget

        num_additional = int(num_additional)  # Ensure integer
        self._debug_print(f"Number of additional combinations to sample: {num_additional}")

        # Generate additional random combinations if needed
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(additional_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Process all combinations
        all_combinations = essential_combinations + additional_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations)}")

        responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations, desc="Processing combinations")):
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations)}:")
            self._debug_print(f"Combination: {combination}")
            self._debug_print(f"Indexes: {indexes}")

            args = self._prepare_combination_args(combination, content)
            response = self.model.generate(**args)
            self._debug_print(f"Received response for combination {idx + 1}")

            if self.progress_queue:
                # count and percentage
                count = idx + 1
                progress_message = {
                    'message': 'Processing',
                    'count': count,
                    'total': len(all_combinations),
                }
                self.progress_queue.put(progress_message)

            key = self._get_combination_key(combination, indexes)
            responses[key] = (response, indexes)

        return responses

    def _get_df_per_combination(self, responses: Dict[str, Tuple[str, Tuple[int, ...]]], baseline_text: str) -> pd.DataFrame:
        """Create DataFrame with combination results"""
        df = pd.DataFrame(
            [(key.split('_')[0], response[0], response[1])
             for key, response in responses.items()],
            columns=['Content', 'Response', 'Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        try:
            vectors = self.vectorizer.vectorize(all_texts)
        except ValueError:  # perhaps the documents only contain stop words
            print("Warning: All generated responses contain stop words. Returning 0 similarity.")
            df['Similarity'] = np.zeros(len(df))
            return df
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["Similarity"] = similarities

        return df

    def _calculate_shapley_values(self, df: pd.DataFrame, content: Any) -> Dict[str, float]:
        """Calculate Shapley values"""
        samples = self._get_samples(content)
        shapley_values = {}

        def normalize_shapley_values(values: Dict[str, float], power: float = 1) -> Dict[str, float]:
            min_value = min(values.values())
            shifted_values = {k: v - min_value for k, v in values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            return {k: v / total for k, v in powered_values.items()}

        for i, sample in tqdm(enumerate(samples, start=1), desc="Calculating Shapley values"):
            with_sample = np.average(
                df[df["Indexes"].apply(lambda x: i in x)]["Similarity"].values
            )
            without_sample = np.average(
                df[df["Indexes"].apply(lambda x: i not in x)]["Similarity"].values
            )

            shapley_values[f"{sample}_{i}"] = with_sample - without_sample

        return normalize_shapley_values(shapley_values)

    @abstractmethod
    def _get_samples(self, content: Any) -> List[Any]:
        """Get samples from content for analysis"""
        pass

    @abstractmethod
    def _prepare_combination_args(self, combination: List[Any], original_content: Any) -> Dict:
        """Prepare model arguments for a combination"""
        pass

    @abstractmethod
    def _get_combination_key(self, combination: List[Any], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        pass

    def save_results(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results_df is not None:
            self.results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            
        if self.shapley_values is not None:
            with open(os.path.join(output_dir, "shapley_values.json"), 'w') as f:
                json.dump(self.shapley_values, f, indent=2)
                
        if metadata:
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
