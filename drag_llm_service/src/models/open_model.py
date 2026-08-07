import os
import re
from vllm import LLM, SamplingParams

# A chat-templated model occasionally echoes a leaked role-name token
# ("system\n", "user\n") at the start of its generation instead of only the
# answer text; the header-id markers are already stripped below, but the
# bare role words were not, which corrupted exact-match-style scoring
# downstream (see reports/ddos_attack.md). Strip any leading role token(s).
_ROLE_PREFIX = re.compile(r"^\s*(system|user|assistant)\s*\n+\s*", re.IGNORECASE)
_HEADER_ID_TOKENS = re.compile(r"<\|start_header_id\|>|<\|end_header_id\|>")


def _strip_leaked_role_tokens(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _ROLE_PREFIX.sub("", text)
    return text

class VLLMModel:
    def __init__(self,
                 model_handle: str,
                 system_setting: str=None,
                 temperature: str=0.0,
                 seed: int=42):
        
        self.model_handle = model_handle
        self.system_setting = system_setting
        self.temperature = temperature
        self.seed = seed
        self.load_model()
        self.restart()

    def load_model(self):
        self.model = LLM(
            model=self.model_handle,
            seed=self.seed,
            gpu_memory_utilization=0.80,  # Use 80% of GPU RAM (leaves headroom for Windows display driver)
        )
        self.tokenizer = self.model.get_tokenizer()

        self.config = SamplingParams(
            n=1,
            temperature=self.temperature,
            max_tokens=128,
            seed=self.seed,
            skip_special_tokens=True
        )
    
    def __call__(self, prompt) -> str:
        if "inst" in self.model_handle.lower():
            self.message.append({"role": "user", "content": prompt})
            self.message = self.tokenizer.apply_chat_template(self.message, tokenize=False)
        else:
            self.message = self.message + prompt
    
        result = self.model.generate([self.message], sampling_params=self.config, use_tqdm=False)
        result = result[0].outputs[0].text
        result = _HEADER_ID_TOKENS.sub("", result)
        result = _strip_leaked_role_tokens(result)
        return result

    def restart(self):
        if "inst" in self.model_handle.lower():
            self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []
        else:
            self.message = self.system_setting if self.system_setting else ""