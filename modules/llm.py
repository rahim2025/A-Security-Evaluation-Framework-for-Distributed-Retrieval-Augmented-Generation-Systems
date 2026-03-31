import json
from ollama import Client


class LLM:
    def __init__(self, llm_url: str, llm_name: str, llm_num_ctx: int, llm_seed: int = 0):
        """
        Initializes an LLM object to interact with an Ollama language model.

        Args:
            llm_url: The URL of the Ollama server (including protocol and port, e.g., 'http://localhost:11434').
            llm_name: The name of the language model to use (e.g., 'llama2').
            llm_seed: An optional seed for the random number generator to ensure reproducibility of results.
        """
        self.llm_name = llm_name
        self.llm_client = Client(host=llm_url)
        self.llm_num_ctx = llm_num_ctx
        self.llm_seed = llm_seed
    
    def generate(self, prompt: str) -> dict:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt: The input prompt for the language model.

        Returns:
            A dictionary containing the parsed JSON response from the LLM.
        """
        # Generate a response from the Ollama model.
        response = self.llm_client.generate(
            model=self.llm_name,
            prompt=prompt,
            format="json",  # Ensure the model returns a JSON formatted response.
            options={
                "seed": self.llm_seed,
                "num_ctx": self.llm_num_ctx,
                "num_predict": self.llm_num_ctx
            }
        )

        # Extract the response text (which should be a JSON string).
        response_text = response["response"]

        # Parse the JSON string into a dictionary.
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            response_json = {}

        return response_json
