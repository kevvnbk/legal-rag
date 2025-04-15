from typing import List, Tuple, Dict

class TradeOffAnalyzer:
    """
    A class to extract trade-off frames from a legal query and evaluate
    legal documents according to each trade-off frame.
    """
    def __init__(self, model):
        self.model = model

    def extract_tradeoff_frames(self, query: str, max_pairs: int = 3) -> List[Tuple[str, str]]:
        """
        Uses an LLM to extract up to 'max_pairs' trade-off frames from the legal query.
        Returns a list of tuples where each tuple is (Value_A, Value_B).
        """
        prompt = (
            f"You are a legal expert. Identify up to {max_pairs} value-based trade-offs "
            f"(e.g., 'Freedom of Expression vs. Institutional Order') relevant to the legal question below.\n\n"
            f"Question: \"{query}\"\n\n"
            "Respond in the following format (one trade-off per line):\n"
            "1. [Value A] vs. [Value B]\n"
            "2. [Value A] vs. [Value B]\n"
        )
        response = self.model.query(prompt)
        frames = response.strip()

        return frames

    def evaluate_document_all_frames(self, document: str, frames: List[Tuple[str, str]]) -> Dict:
        """
        Evaluates a legal document with respect to multiple trade-off frames at once.
        It asks the LLM to provide a score (1-10) for each value in every frame along with a brief justification,
        in one single prompt. The response is expected in a JSON format and parsed accordingly.
        
        :param document: The legal document to evaluate.
        :param frames: A list of tuples, where each tuple is (Value_A, Value_B) representing a trade-off frame.
        :return: A dictionary containing the raw LLM response and the parsed evaluations if JSON parsing succeeds.
        """
        prompt = (
            "You are a legal analyst. Evaluate how much the following legal document emphasizes each of the trade-off frames listed below. "
            "For each frame, provide a score from 1 to 10 for each value. Do not include any justifications.\n\n"
            "Output your response as a list of score pairs. For each frame, output a single line in the following format:\n"
            "[[Value A, Value B], [score_for_Value_A, score_for_Value_B]]\n\n"
            "Document:\n"
            f"\"{document.strip()}\"\n\n"
            "Frames:\n"
            f"\"{frames}\"\n"
        )
        
        response = self.model.query(prompt)
        return response