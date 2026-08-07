"""Evaluate RORA with a local fine-tuned model.
"""
import torch
import os
import transformers
from ..models.huggingface_wrapper_module import HuggingfaceWrapperModule
from ..collate_fns.rora_collate_fn import RORAGenerationCollateFn
from ..trainer.rora_trainer import RORATrainer
from ..metrics.loss import AvgLoss
from ..metrics.accuracy import GenerationAccuracyMetric

class RORAModel:
    def __init__(self,
                 model_dir: str,
                 rationale_format: str,
                 device: str = "cuda:0"):
        self.model_dir = model_dir
        self.rationale_format = rationale_format
        self.device = device
        
    def load_model(self):
        model_dir = os.path.join(self.model_dir, "best_1")
        model = HuggingfaceWrapperModule.load_from_dir(model_dir)
        model.eval()
        model.to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_handle)
    
        self.collate_fn = RORAGenerationCollateFn(
            rationale_format=self.rationale_format,
            max_input_length=2048,
            max_output_length=32,
            tokenizer=self.tokenizer
        )
    
        self.trainer = RORATrainer(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0),
            device=self.device,
            metrics={
                "loss": AvgLoss(),
            },
            eval_metrics={
                "loss": AvgLoss(),
                "acc": GenerationAccuracyMetric(tokenizer=self.tokenizer),
            },
            main_metric="loss",
            save_dir=None,
        )

        self.metric = AvgLoss()

    def evaluate(self, rationale: str, question: str, answer: str):
        """
        Run self.trainer._eval_step on the given rationale and question.
        and compute the RORA score: the log probability difference of getting the 
        correct answer with and without the given rationale.
        """
        assert hasattr(self, 'collate_fn') and hasattr(self, 'trainer'), "Call load_model() first."

        # Build with- and without-rationale items
        item_with = {
            'question': question,
            'rationale': rationale,
            'answer': answer,
        }
        item_without = {
            'question': question,
            'rationale': "",
            'answer': answer,
        }

        batch_with = self.collate_fn.collate([item_with])
        batch_without = self.collate_fn.collate([item_without])

        for k, v in list(batch_with.items()):
            if torch.is_tensor(v):
                batch_with[k] = v.to(self.device)
        for k, v in list(batch_without.items()):
            if torch.is_tensor(v):
                batch_without[k] = v.to(self.device)

        # Prepare single-compare batch for trainer._eval_step
        logits_with = self.trainer._eval_step(batch_with)
        logits_without = self.trainer._eval_step(batch_without)
        
        # Compute RORA score
        score = self.metric._detach_tensors(logits_with)["loss"] - self.metric._detach_tensors(logits_without)["loss"]

        return score