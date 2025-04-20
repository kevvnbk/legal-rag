from typing import List, Dict

from src.pirac.graph import ConsistencyGraph

class PIRAC:
    def __init__(self, llm_model, nli_model):
        self.llm = llm_model
        self.agents = {
            'issue': IRACAgent(self.llm, prompt_template=ISSUE_PROMPT),
            'rule': IRACAgent(self.llm, prompt_template=RULE_PROMPT),
            'application': IRACAgent(self.llm, prompt_template=APPLICATION_PROMPT),
            'conclusion': IRACAgent(self.llm, prompt_template=CONCLUSION_PROMPT)
        }
        self.cg = ConsistencyGraph(nli_model)

    def run(self, query: str, docs: List[str], threshold: float = 0.5) -> Dict:
        # 1. Retrieval
        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])

        # 2. Parallel IRAC Generation with dependencies
        irac_outputs = {}
        irac_outputs['issue'] = self.agents['issue'].generate(query=query, legal_materials=context)
        irac_outputs['rule'] = self.agents['rule'].generate(query=query, legal_materials=context)
        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])

        # 3. Consistency Graph / ICS
        ics, edge_scores = self.cg.compute_ics(irac_outputs)
        
        # 4. Regeneration if ICS < threshold
        count = 0
        while ics < threshold:
            regenerated = False
            for (src, tgt), score in edge_scores.items():
                if score < threshold:
                    regenerated = True
                    if src == 'issue' and tgt == 'rule':
                        # regenerate rule and downstream nodes
                        irac_outputs['rule'] = self.agents['rule'].generate(query=query, legal_materials=context)
                        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
                    elif src == 'rule' and tgt == 'application':
                        # regenerate application and downstream node
                        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
                    elif src == 'application' and tgt == 'conclusion':
                        # regenerate conclusion only
                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])

            # if any regeneration occurred, recompute ICS and edge scores
            if regenerated:
                ics, edge_scores = self.cg.compute_ics(irac_outputs)
                count += 1

                if count > 3:
                    break

        return irac_outputs
    
class IRACAgent:
    def __init__(self, llm_model: str, prompt_template: str):
        self.llm_model = llm_model
        self.prompt_template = prompt_template

    def generate(self, query: str, legal_materials: str = "", rules: str = "", application: str = "") -> str:
        if "{rules}" in self.prompt_template:
            prompt = self.prompt_template.format(query=query, rules=rules)
            response = self.llm_model.query(prompt)
        elif "{application}" in self.prompt_template:
            prompt = self.prompt_template.format(query=query, application=application)
            response = self.llm_model.query(prompt)
        else:
            prompt = self.prompt_template.format(query=query, legal_materials=legal_materials)
            response = self.llm_model.query(prompt)

        return response

ISSUE_PROMPT = (
    "You are a legal expert. Given the following query and legal materials, identify the central legal issue:\n\n"
    "Make sure to begin your response with <think>\\n.\n"
    "Query: {query}\n\n"
    "Legal Materials: {legal_materials}\n\n"
    )

RULE_PROMPT = (
    "You are a legal expert. Given the following query and legal materials, state the applicable legal rules:\n\n"\
    "Make sure to begin your response with <think>\\n.\n"
    "Query: {query}\n\n"
    "Legal Materials: {legal_materials}\n\n"
    )

APPLICATION_PROMPT = (
    "You are a legal expert. Given the rules and query, how do these rules apply to this specific situation?\n\n"
    "Make sure to begin your response with <think>\\n.\n"
    "Query: {query}\n\n"
    "Rules: {rules}\n\n"
    )

CONCLUSION_PROMPT = (
    "You are a legal expert. Based on the application of the rules to the facts, what is the likely legal outcome?\n\n"
    "Make sure to begin your response with <think>\\n.\n"
    "Query: {query}\n\n"
    "Application: {application}\n\n"
    )