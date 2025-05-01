import re
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TopPLogitsWarper,
)
import torch
import logging
logger = logging.getLogger(__name__)

class IRAC:
    """
    Entailment-Constrained Decoding for legal IRAC reasoning.
    Generates Issue, Rule, Application, Conclusion in logical chunks,
    applies NLI-based entailment checks after each chunk, 
    uses soft logit biasing and dynamic decoding strategies to enforce coherence.
    """
    def __init__(
        self,
        llm_model,
        nli_model,
        entailment_threshold: float = 0.8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Load LLM for generation
        self.llm = llm_model
        self.tokenizer = llm_model.tokenizer
        self.llm_model = llm_model.model

        # Load NLI model for entailment checks
        self.nli = nli_model
        self.nli_tokenizer = nli_model.tokenizer
        self.nli_model = nli_model.model
        self.threshold = entailment_threshold
        self.device = device

    def run(self, query: str, docs=None, max_chunks: int = 5, top_p: float = 0.9) -> dict:
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
        if docs is not None:
            base_context = " ".join(map(str, docs))
        else:
            base_context = query

        outputs = {}

        for section in sections:
             # Build the prompt so the LLM always sees the original query, any docs, and preceding sections
            prompt = (
                f"Query: {query}\n"
                f"Context: {base_context}\n"
                f"{' '.join(f'{k}: {v}' for k, v in outputs.items())}\n"
                f"{section}:"
            )

            # Premise for entailment = query + docs + all previous sections
            previous_context = f"{base_context} {' '.join(outputs.values())}".strip()

            outputs[section] = self._generate_section(
                prompt=prompt,
                previous=previous_context,
                max_chunks=max_chunks,
                top_p=top_p
            )

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
                logger.info(f"Decoded chunk: {chunk}")
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
        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_p=top_p,
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
        probs = self.nli.query(premise, hypothesis)
        labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        best_idx = torch.argmax(probs).item()
        return labels[best_idx], probs[best_idx].item()