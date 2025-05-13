import torch
import torch.nn as nn
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline

import re
import logging
logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 1024
CONTEXT_MAX_TOKENS = {
    "meta-llama/Llama-2-7b-hf": 4096,
    "meta-llama/Llama-3.2-1B-Instruct": 8000,
    "nvidia/Llama3-ChatQA-1.5-8B": 128_000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 8192,
    "Equall/Saul-7B-Instruct-v1": 8192,
    "meta-llama/Llama-3.1-8B-Instruct": 128_000,
}
def create_model(model_name, **kwargs):
    model_mapping = {
        "llama7b": "meta-llama/Llama-2-7b-hf",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3qa-8b": "nvidia/Llama3-ChatQA-1.5-8B",
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deberta": "potsawee/deberta-v3-large-mnli",
        "saul7b": "Equall/Saul-7B-Instruct-v1",
        #"saul7b": "./Saul-7B-Instruct-v1",
        "bart-large": "facebook/bart-large-mnli",
    }
    if model_name in model_mapping:
        if model_name == "deberta":
            return HFModelBERT(model_mapping[model_name], **kwargs)
        elif model_name == "saul7b":
            return HFModelPipeline(model_mapping[model_name], **kwargs)
        else:
            return HFModel(model_mapping[model_name], **kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported. Available models: {', '.join(model_mapping.keys())}")

class BaseModel:
    def __init__(self):
        self.clean_str = []
        self.prompt_template = {}

    def query(self, prompt):
        wrapped_prompt = self.wrap_prompt(prompt)
        return self._query(wrapped_prompt)

    def _query(self, prompt):
        raise NotImplementedError

    def extract_single_choice(self, response: str) -> str:
        match = re.search(r'\b([ABCD])\b', response.strip())
        return match.group(1) if match else None
        
        # def _clean_response(self, response):
        #     for pattern in self.clean_str:
        #         idx = response.find(pattern)
        #         if idx != -1:
        #             response = response[:idx]
            
        #     return response.strip().lstrip('\n')

    def _clean_response(self, response: str) -> str:
        """
        Keep only the portion that follows the closing </think> tag,
        then run the other clean‑ups you already defined.
        """
        # if self.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        #     close_tag = "</think>"
        # elif self.model_name == "Equall/Saul-7B-Instruct-v1":
        #     close_tag = "[/INST']"
        close_tag = "[/INST]"
        idx = response.find(close_tag)
        if idx != -1:
            response = response[idx + len(close_tag):]  
        # run your other string‑trims
        # for pattern in self.clean_str:
        #     cut = response.find(pattern)
        #     if cut != -1:
        #         response = response[:cut]
        return response.lstrip()        # strip leading spaces / newlines

    def wrap_prompt(self, prompt):
        """
        Wraps the user prompt in the system/user/assistant header format
        expected by the model.
        """
        system_header = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.\n"
            "**Rules you must follow for every reply**\n"
            "1. Think silently.\n"
            "2. In your visible reply, output only the answer.\n"
            "3. Output nothing else—no punctuation, no explanations, no extra words.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        )
        assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return f"{system_header}{prompt}{assistant_header}"

    # def wrap_prompt(self, prompt):
    #     header = (
    #         "You are a legal reasoning assistant.\n"
    #         "**Rules you must follow for every reply**\n"
    #         "1. Think between <think> and </think> tags.\n"
    #         "2. In your response, output **only one word** and the **answer only**.\n"
    #         "3. Output nothing else—no punctuation, no explanations, no extra words.\n"
    #         "<｜end▁of▁sentence｜><｜User｜>"
    #     )
    #     footer = (
    #         "<｜Assistant｜><think>"
    #     )
    #     return f"{header}{prompt}{footer}"

class HFModel(BaseModel):
    def __init__(self, model_name, max_output_tokens=None, **kwargs):
        super().__init__()
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        default_kw = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        default_kw.update(kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **default_kw)
        self.model.eval()

        # Enable multi-GPU inference if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        model_max_length = CONTEXT_MAX_TOKENS.get(self.model_name, 2048)
        device = next(self.model.parameters()).device
        max_input_length = max(1, model_max_length - self.max_output_tokens)

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=False, return_attention_mask=True)
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device) if "attention_mask" in tokenized else None

        if input_ids.shape[1] > max_input_length:
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_attention_mask=True,
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
        
        with torch.inference_mode():
            # Use the underlying model if wrapped in DataParallel
            model_for_generate = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            outputs = model_for_generate.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True, 
                # top_p=0.95,
                # temperature=1.3,
                temperature=1.0,
                max_new_tokens=self.max_output_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        result = self._clean_response(result)
        # result = self.extract_single_choice(result)
        
        # --- free CUDA & CPU memory no longer needed ---
        del outputs, generated_tokens, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()   # release unreferenced CUDA memory
        gc.collect()              # encourage Python to free CPU objects

        return result

class HFModelPipeline(BaseModel):
    def __init__(self, model_name, device=None, **kwargs):
        super().__init__()
        self.pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def query(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        result = outputs[0]["generated_text"]
        result = self._clean_response(result)

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
        prob_entailment = probs[0].item()
        return prob_entailment