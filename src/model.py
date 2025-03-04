import torch
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# from huggingface_hub import login

# login()

MAX_NEW_TOKENS = 20
CONTENT_MAX_TOKENS = {
    "meta-llama/Llama-2-7b-chat": 4096
}
def create_model(model_name, **kwargs):
    if model_name == "llama7b":
        return HFModel("meta-llama/Llama-2-7b-chat-hf",**kwargs)
    else:
        raise NotImplementedError

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
        
    def wrap_prompt(self, data):
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
        self.generation_kwargs = {
            "max_new_tokens": self.max_output_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False
        }
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_response(result)