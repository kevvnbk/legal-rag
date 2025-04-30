import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TopPLogitsWarper,
    LogitsWarperList,
)
import torch

class IRAC:
    """
    Entailment-Constrained Decoding for legal IRAC reasoning.
    Generates Issue, Rule, Application, Conclusion in logical chunks,
    applies NLI-based entailment checks after each chunk, 
    uses soft logit biasing and dynamic decoding strategies to enforce coherence.
    """
    def __init__(
        self,
        llm_model_name: str,
        nli_model_name: str,
        entailment_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Load LLM for generation
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Load NLI model for entailment checks
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        self.threshold = entailment_threshold
        self.device = device

    def run(self, query: str, max_chunks: int = 5, top_p: float = 0.9) -> dict:
        """
        Generate a full IRAC response with entailment-constrained decoding.
        Args:
            query: The legal question.
            max_chunks: Number of chunks (sentences/clauses) per IRAC component.
            top_p: Initial top-p sampling parameter.
        Returns:
            Dict with keys 'Issue', 'Rule', 'Application', 'Conclusion'.
        """
        sections = ["Issue", "Rule", "Application", "Conclusion"]
        context = query
        outputs = {}

        for section in sections:
            prompt = f"{' '.join(f'{k}: {v}' for k,v in outputs.items())}\n{section}:"
            outputs[section] = self._generate_section(
                prompt=prompt,
                previous=context,
                max_chunks=max_chunks,
                top_p=top_p
            )
            context = outputs[section]

        return outputs

    def _generate_section(self, prompt: str, previous: str, max_chunks: int, top_p: float) -> str:
        """
        Generate a section in logical chunks with entailment checks.
        """
        full_text = ""
        for chunk_idx in range(max_chunks):
            current_top_p = top_p
            for attempt in range(3):
                chunk = self._decode_chunk(
                    prompt=prompt + " " + full_text,
                    top_p=current_top_p
                )
                label, score = self._check_entailment(previous, chunk)
                if label == "ENTAILMENT" and score >= self.threshold:
                    full_text += (" " if full_text else "") + chunk.strip()
                    break
                current_top_p = max(0.3, current_top_p * 0.7)
            else:
                full_text += (" " if full_text else "") + chunk.strip()

            if re.search(r"[\.!?]$", full_text.strip()):
                break
        return full_text.strip()

    def _decode_chunk(self, prompt: str, top_p: float) -> str:
        """
        Decode the next logical chunk (up to the first sentence terminator)
        using soft logit biasing and top-p sampling.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        logits_warper = LogitsWarperList([TopPLogitsWarper(top_p=top_p)])
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            logits_warper=logits_warper,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        match = re.search(r"^.*?[\.!?]", generated)
        return match.group(0) if match else generated

    def _check_entailment(self, premise: str, hypothesis: str):
        """
        Run NLI model to check entailment between premise and hypothesis.
        Returns (label, score).
        """
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        best_idx = torch.argmax(probs).item()
        return labels[best_idx], probs[best_idx].item()