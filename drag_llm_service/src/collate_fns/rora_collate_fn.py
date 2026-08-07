import re
from typing import Dict, Any, Text, Optional, List, Tuple, Union
from transformers import PreTrainedTokenizer
from overrides import overrides

from .collate_fn import CollateFn

__TEMPLATES__ = {
    "g": "{gold_rationale}",
    "s": "{base_rationale}",
    "l": "{leaky_rationale}",
    "gs": "{gold_rationale} {base_rationale}",
    "ls": "{leaky_rationale} {base_rationale}"
}


__LABEL_TO_ANSWER__ = {
    True: "yes",
    False: "no"
}


__LABEL_TO_LEAKY_RATIONALE__ = {
    True: f"The answer is {__LABEL_TO_ANSWER__[True]}",
    False: f"The answer is {__LABEL_TO_ANSWER__[False]}"
}

def generate_no_more_than_ngrams(
    x: List[Text],
    n: int
) -> List[Text]:
    """Given a list of text,
    generate all ngrams from 1 to n.
    """
    
    # i-gram
    ngram_set = set(x)
    
    if n > 1:
        for i in range(2, n+1):
            ngram_set = ngram_set.union(set([' '.join(t) for t in zip(*[x[ii:] for ii in range(i)])]))
            
    return list(ngram_set)


class RORACollateFn(CollateFn):
    
    __TEMPLATES__ = __TEMPLATES__
    __LABEL_TO_ANSWER__ = __LABEL_TO_ANSWER__
    __LABEL_TO_LEAKY_RATIONALE__ = __LABEL_TO_LEAKY_RATIONALE__
    
    def __int__(
        rationale_format: Text,
    ):
        super().__init__(rationale_format=rationale_format)
        
    def rationale_templating(self, item: Dict[Text, Any]) -> Text:
        """Given an item, return the template filled with respective fields.
        """
        # Allow bypassing templates for on-the-fly evaluation
        if 'rationale_text' in item:
            return item['rationale_text']
        
        template = self.__TEMPLATES__[self.rationale_format]
        return template.format(
            gold_rationale=item['rationale'],
            leaky_rationale=self.__LABEL_TO_LEAKY_RATIONALE__[item['answer']]
        )
        
    def templating(self, item: Dict[Text, Any]) -> Text:
        """
        """
        return f"question: {item['question']} rationale: {self.rationale_templating(item)}"
    

class RORAGenerationCollateFn(RORACollateFn):
    def __init__(
        self,
        rationale_format: Text,
        tokenizer: PreTrainedTokenizer,
        max_input_length: Optional[int] = 256,
        max_output_length: Optional[int] = 32,
        removal_threshold: Optional[float] = None,
        mask_by_delete: Optional[bool] = False,
        rationale_only: Optional[bool] = False,
    ):
        """
        """
        super().__init__(rationale_format=rationale_format)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.removal_threshold = removal_threshold
        self.mask_by_delete = mask_by_delete
        self.rationale_only = rationale_only
        
    @overrides
    def templating(self, item: Dict[Text, Any]) -> Text:
        """Now there's possibility of removing spurious
        for rationale_template, we need to do it here.
        """
        
        if self.removal_threshold is not None:
            assert "attributions" in item, f"One or more items do not have attributions but we need to perform attribution-based removal."
            
            return "question: {question} rationale: {rationale}".format(
                question=item['question'],
                rationale=self.remove_spurious(self.rationale_templating(item), attributions=item['attributions'])
            )
            
        else:
            if self.rationale_only:
                return "rationale: {rationale}".format(
                    rationale=self.rationale_templating(item)
                )
            else:
                return "question: {question} rationale: {rationale}".format(
                    question=item['question'],
                    rationale=self.rationale_templating(item)
                )
        
    @overrides
    def collate(
        self,
        x: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        """Take the input and construct it into a format
        that will become the input of the model.
        """
        
        # construct prompt and target
        input_strs: List[Text] = [
            self.templating(item) for item in x
        ]
        
        input_outputs = self.tokenizer(
            input_strs,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = input_outputs.input_ids
        attention_mask = input_outputs.attention_mask
        
        labels = self.tokenizer(
            [
                self.__LABEL_TO_ANSWER__[item['answer']] for item in x
            ],
            max_length=self.max_output_length,
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        neg_labels = self.tokenizer(
            [
                self.__LABEL_TO_ANSWER__[not item['answer']] for item in x
            ],
            max_length=self.max_output_length,
            padding="longest",
            truncation=True,
            return_tensors='pt'
        ).input_ids
        
        neg_labels[neg_labels == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            "attention_mask": attention_mask,
            'labels': labels,
            "neg_labels": neg_labels
        }
        
    def remove_spurious(
        self,
        input_str: Text,
        attributions: List[Dict[Text, Any]],
        removal_threshold: Optional[float] = None,
        mask_by_delete: Optional[bool] = None
    ) -> Text:
        """We take input sentences and remove the spurious correlation
        and replace them with rationales.
        """

        if removal_threshold is None:
            removal_threshold = self.removal_threshold
            
        if mask_by_delete is None: 
            mask_by_delete = self.mask_by_delete
        
        # TODO: check whether T-5 does this with similar ratio.
        # TODO: make this more general for other models and tokenizers
        index_to_special_token = lambda x: "" if mask_by_delete else f"<extra_id_{x}>"
        
        spans: List[Tuple[int, int]] = []

        def _join(nspan: Tuple[int, int], banks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Integrate a new span into the existing span bank.
            """
            
            # Notice that is_join operate on [) left closed right open interval.
            _is_join = lambda x, y: x[1] > y[0] and x[0] < y[1]
            
            # TODO: check if we need to sort the banks every time
            banks = sorted([tuple(nspan)] + banks, key=lambda x: x[0], reverse=False)
            new_banks = []
            
            for ospan in banks:
                if new_banks and _is_join(new_banks[-1], ospan):
                    new_banks[-1] = (new_banks[-1][0], max(ospan[1], new_banks[-1][1]))
                else:
                    new_banks.append(ospan)
                    
            return new_banks
                
        
        for attr in filter(lambda x: x['score'] > removal_threshold, attributions):
            for attr_span in attr['in_rationale_ids']:
                spans = _join(attr_span, spans)
            
        # now fix the spans by joining spans separated by space tokens
        fixed_spans = []
        
        for span in spans:
            if fixed_spans and re.fullmatch(r"\s*", input_str[fixed_spans[-1][1]:span[0]]) is not None:
                fixed_spans[-1] = (fixed_spans[-1][0], span[1])
            else:
                fixed_spans.append(span)
                
        concatenated_inputs = []
        last_time_idx = 0
        for span_idx, span in enumerate(fixed_spans):
            concatenated_inputs.append(input_str[last_time_idx:span[0]].strip())
            concatenated_inputs.append(index_to_special_token(span_idx))
            last_time_idx = span[1]
            
        concatenated_inputs.append(input_str[last_time_idx:])
            
        return " ".join(concatenated_inputs)
    
    