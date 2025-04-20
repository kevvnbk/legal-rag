import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import logging
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 500
CONTEXT_MAX_TOKENS = {
    "meta-llama/Llama-2-7b-hf": 4096,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 2048,
}
def create_model(model_name, **kwargs):
    model_mapping = {
        "llama7b": "meta-llama/Llama-2-7b-hf",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deberta": "potsawee/deberta-v3-large-mnli",
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map='auto', 
            **kwargs
        )
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        model_max_length = CONTEXT_MAX_TOKENS.get(self.model_name, 2048)
        max_input_length = max(1, model_max_length - self.max_output_tokens)

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device) if "attention_mask" in tokenized else None

        if input_ids.shape[1] > max_input_length:
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length
            )
            input_ids = tokenized.input_ids.to(self.model.device)
            attention_mask = tokenized.attention_mask.to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False, 
                top_p=None,
                temperature=None,
                max_new_tokens=self.max_output_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
    
class HFModelBERT(BaseModel):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def query(self, textA, textB):
        inputs = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(textA, textB)],
            add_special_tokens=True, return_tensors="pt",
        )
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # probs = [0.7080, 0.2920], meaning that prob(entail) = 0.708, prob(contradict) = 0.292
        return probs
    