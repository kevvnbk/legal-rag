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
            f"You are a legal expert. Identify exactly 3 legal value-based trade-offs relevant to the question below."
            "For each trade-off, output a single line in the following format: Name of Value A vs. Name of Value B\n"
            "In total you should output ONLY 3 lines.\n"
            "DO NOT INCLUDE ANY JUSTIFICATIONS. ONLY OUTPUT THE TRADE-OFFS.\n\n"
            "Make sure to begin your response with \"<think>\\n\"."
            f"Question: \"{query}\"\n\n"
        )
        response = self.model.query(prompt)
        frames = response.split("</think>")[-1].strip()
 
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
            f"Frames: {frames}\n"
            "For each frame, provide a score from 1 to 10 for each value."
            "For each frame, output a single line in the following format:\n"
            "[[Name of Value A, Name of Value B], [Score for Value A, Score for Value B]]\n\n"
            "In total you should output ONLY 3 lines.\n"
            "DO NOT INCLUDE ANY JUSTIFICATIONS. ONLY OUTPUT THE VALUE PAIRS AND ITS SCORE PAIRS.\n\n"
            "Make sure to begin your response with \"<think>\\n\"."
            "Document:\n"
            f"\"{document}\""
        )
        
        response = self.model.query(prompt)
        response = response.split("</think>")[-1].strip()
        return response