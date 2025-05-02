import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import logging
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 1024
CONTEXT_MAX_TOKENS = {
    "meta-llama/Llama-2-7b-hf": 4096,
    "meta-llama/Llama-3.2-1B-Instruct": 8000,
    "nvidia/Llama3-ChatQA-1.5-8B": 128_000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 8192,
    "Equall/Saul-7B-Instruct-v1": 4096,
}
def create_model(model_name, **kwargs):
    model_mapping = {
        "llama7b": "meta-llama/Llama-2-7b-hf",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3qa-8b": "nvidia/Llama3-ChatQA-1.5-8B",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deberta": "potsawee/deberta-v3-large-mnli",
        "saul7b": "Equall/Saul-7B-Instruct-v1",
        #"saul7b": "./Saul-7B-Instruct-v1",
        "bart-large": "facebook/bart-large-mnli",
    }
    if model_name in model_mapping:
        if model_name == "deberta":
            return HFModelBERT(model_mapping[model_name], **kwargs)
        else:
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
    
    # def _clean_response(self, response):
    #     for pattern in self.clean_str:
    #         idx = response.find(pattern)
    #         if idx != -1:
    #             response = response[:idx]
        
    #     return response.strip()

    def _clean_response(self, response: str) -> str:
        """
        Keep only the portion that follows the closing </think> tag,
        then run the other clean‑ups you already defined.
        """
        close_tag = "[/INST]"
        idx = response.find(close_tag)
        if idx != -1:
            response = response[idx + len(close_tag):]   # text *after* </think>
        # run your other string‑trims
        # for pattern in self.clean_str:
        #     cut = response.find(pattern)
        #     if cut != -1:
        #         response = response[:cut]
        return response.lstrip()        # strip leading spaces / newlines
        
    def wrap_prompt(self, prompt):
        pass


class HFModel(BaseModel):
    def __init__(self, model_name, max_output_tokens=None, **kwargs):
        super().__init__()
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        default_kw = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            force_download=True
        )
        default_kw.update(kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **default_kw)
        self.model.eval()
        # self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        model_max_length = CONTEXT_MAX_TOKENS.get(self.model_name, 2048)
        max_input_length = max(1, model_max_length - self.max_output_tokens)

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=False, return_attention_mask=True)
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device) if "attention_mask" in tokenized else None

        if input_ids.shape[1] > max_input_length:
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_attention_mask=True,
            )
            input_ids = tokenized.input_ids.to(self.model.device)
            attention_mask = tokenized.attention_mask.to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True, 
                top_p=0.95,
                temperature=1.3,
                max_new_tokens=self.max_output_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        result = self._clean_response(result)
        
        # --- free CUDA & CPU memory no longer needed ---
        del outputs, generated_tokens, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()   # release unreferenced CUDA memory
        gc.collect()              # encourage Python to free CPU objects

        return result
    
class HFModelBERT(BaseModel):
    def __init__(self, model_name, device=None, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def query(self, textA, textB):
        inputs = self.tokenizer.encode(textA, textB, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(inputs.to(self.device))[0]
            probs = logits.softmax(dim=-1)[0]
        
        prob_neutral = probs[1].item()
        return prob_neutral