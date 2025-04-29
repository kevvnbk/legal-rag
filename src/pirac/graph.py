from typing import Dict, Tuple

class ConsistencyGraph:
    def __init__(self, nli_model):
        self.nli = nli_model

    def score_edge(self, textA: str, textB: str) -> float:
        result = self.nli.query(textA, textB)
        return result[0]

    def compute_ics(self, irac: Dict[str, str], weights: Dict[Tuple[str, str], float] = None) -> Tuple[float, Dict]:
        # edges = [
        #     ('issue', 'rule'),
        #     ('rule', 'application'),
        #     ('application', 'conclusion')
        # ]
        edges = [
            ('issue', 'analysis'),
            ('analysis', 'conclusion')
        ]
        scores = {}
        for (src, tgt) in edges:
            score = self.score_edge(irac[src], irac[tgt])
            scores[(src, tgt)] = score
        ics = min(scores.values())
        return ics, scores