import os
from vllm import LLM, SamplingParams

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
            formatted = self.tokenizer.apply_chat_template(self.message, tokenize=False)
        else:
            formatted = self.message + prompt

        result = self.model.generate([formatted], sampling_params=self.config, use_tqdm=False)
        result = result[0].outputs[0].text
        result = result.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").replace("assistant", "")
        return result

    def restart(self):
        if "inst" in self.model_handle.lower():
            self.message = [{"role": "system", "content": self.system_setting}] if self.system_setting else []
        else:
            self.message = self.system_setting if self.system_setting else ""