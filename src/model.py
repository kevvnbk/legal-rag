import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 20
CONTEXT_MAX_TOKENS = {
    "meta-llama/Llama-2-7b-chat": 4096
}
def create_model(model_name, **kwargs):
    model_mapping = {
        "llama7b": "meta-llama/Llama-2-7b-chat"
    }
    if model_name in model_mapping:
        return HFModel(model_mapping[model_name], **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported. Available models: {', '.join(model_mapping.keys())}")

class BaseModel:
    def __init__(self):
        self.clean_str = []
        self.prompt_template = {}

    def query(self, prompt):
        return self._query(prompt)

    def _query(self, prompt):
        raise NotImplementedError
    
    def _clean_response(self, response):
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx != -1:
                response = response[:idx]
        
        return response.strip()
        
    def wrap_prompt(self, prompt):
        pass


class HFModel(BaseModel):
    def __init__(self, model_name, max_output_tokens=None, **kwargs):
        super().__init__()
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', **kwargs)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        self.clean_str = ['\n\n']

    def _query(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to("cuda")

        outputs = self.model.generate(
            **inputs, 
            do_sample=False, 
            top_p=None,
            temperature=None,
            max_new_tokens=20,
            pad_token_id=self.tokenizer.eos_token_id
        )
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result