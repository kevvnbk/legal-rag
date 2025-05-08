"""
irac_batch.py
One-shot IRAC generation + entailment-repair loop
"""

import re
import logging
import torch
from typing import Dict, List, Tuple

from src.defense import extract_label

logger = logging.getLogger(__name__)


class IRACBatch:
    """
    Generate Issue–Rule–Application–Conclusion in one pass,
    then repair any part that is not entailed by the provided context.
    """

    SECTIONS = ["Issue", "Rule", "Application", "Conclusion"]

    def __init__(
        self,
        llm_model,
        nli_model,
        entailment_threshold: float = 0.8,
        max_retries: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # ── LLM setup ───────────────────────────────────────────────────────────
        self.llm = llm_model
        self.tokenizer = llm_model.tokenizer
        self.llm_model = llm_model.model

        # ── NLI setup ───────────────────────────────────────────────────────────
        self.nli = nli_model
        self.nli_tokenizer = nli_model.tokenizer
        self.nli_model = nli_model.model

        self.threshold = entailment_threshold
        self.max_retries = max_retries
        self.device = device

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │ Public API                                                          │
    # ╰──────────────────────────────────────────────────────────────────────╯
    def run(
        self,
        query: str,
        docs: List[str] | None = None,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        labels: list[str] | None = None,
    ) -> str:
        """
        Returns a label from the provided labels list based on IRAC analysis.
        """
        if labels is None:
            labels = ["Yes", "No"]

        context = " ".join(map(str, docs)) if docs else query

        # 1. One-shot draft ------------------------------------------------------
        raw_irac = self._generate_irac(query, context, top_p, max_new_tokens)
        parts = self._split_irac(raw_irac)

        # 2. Entailment repair loop ---------------------------------------------
        for section in self.SECTIONS:
            parts[section] = self._ensure_entailment(
                section_name=section,
                premise=context,
                initial_text=parts[section],
                query=query,
            )

        # 3. Classification among provided labels
        irac_text = "\n".join(f"{k.upper()}: {v}" for k, v in parts.items())
        prompt = (
            f"Question: {query}\n"
            f"Context: {irac_text}\n"
            f"You are a legal reasoning assistant. Given the IRAC analysis below, "
            f"answer with exactly one of the following options: {', '.join(labels)}. No explanation.\n"
            "Do not include any explanation—answer with exactly just one label.\n"
            "Answer:"
        )
        raw = self._single_turn_completion(prompt).strip()
        label = extract_label(raw, labels)
        return label

    # ╭──────────────────────────────────────────────────────────────────────╮
    # │ Internal helpers                                                    │
    # ╰──────────────────────────────────────────────────────────────────────╯
    def _generate_irac(
        self, query: str, context: str, top_p: float, max_new_tokens: int
    ) -> str:
        """Prompt the LLM for a full IRAC answer in one go."""
        prompt = (
            "You are a legal reasoning assistant. Using the IRAC(Issue, Rule, Application, Conclusion) format, write each "
            "section on its own line starting with the section name in ALL CAPS "
            "followed by a colon. Example:\n"
            "ISSUE: ...\nRULE: ...\nAPPLICATION: ...\nCONCLUSION: ...\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n\n"
            "Begin your IRAC response with </think>:\n"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.llm_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True
        )
        logger.debug(f"RAW_IRAC\n{generated}")
        return generated

    def _split_irac(self, text: str) -> Dict[str, str]:
        """Parse the one-shot output back into a dict."""
        parts = {k: "" for k in self.SECTIONS}
        pattern = re.compile(r"^\s*(ISSUE|RULE|APPLICATION|CONCLUSION)\s*:\s*(.+)", re.I)
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                key, value = match.group(1).title(), match.group(2).strip()
                if key in parts:
                    parts[key] = value
        return parts

    def _ensure_entailment(
        self,
        section_name: str,
        premise: str,
        initial_text: str,
        query: str,
    ) -> str:
        """Iteratively repair a section until it meets the entailment threshold."""
        consecutive_failures = 0
        text = initial_text
        attempt = 0
        while True:
            neutral_prob = self._check_entailment(premise, text)
            logger.debug(
                f"{section_name} | attempt {attempt} | neutral:{neutral_prob:.3f}"
            )
            if neutral_prob >= 0.85:
                return text  # success
            consecutive_failures += 1
            if consecutive_failures > 3:
                return text  # give up after 3 consecutive failures

            # Ask LLM to rewrite only the problematic section
            fix_prompt = (
                f"The following {section_name} is not sufficiently entailed by the "
                f"context. Please rewrite it so that it logically follows with </think>. "
                f"Context: {premise}\n"
                f"Original {section_name}: {text}\n"
                f"Rewritten {section_name}:"
            )

            text = self._single_turn_completion(fix_prompt).strip()
            attempt += 1

    def _single_turn_completion(self, prompt: str) -> str:
        """Utility for short completions without extra parsing."""
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.llm_model.generate(
            ids,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][ids.shape[-1] :], skip_special_tokens=True)

    # ── NLI glue ───────────────────────────────────────────────────────────────
    def _check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float, float]:
        """
        Returns (predicted_label, entailment_score, neutral_score) where
        * `predicted_label` ∈ {CONTRADICTION, NEUTRAL, ENTAILMENT}
        * `entailment_score` is the model‑estimated probability of ENTAILMENT
        * `neutral_score`    is the model‑estimated probability of NEUTRAL
        """
        prob = self.nli.query(premise, hypothesis)

        return prob