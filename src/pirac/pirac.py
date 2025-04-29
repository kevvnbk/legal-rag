from typing import List, Dict
import torch
import logging
logger = logging.getLogger(__name__)

from src.pirac.graph import ConsistencyGraph
from src.retriever.retriever import clean_document

class PIRAC:
    def __init__(self, llm_model, nli_model):
        self.llm = llm_model
        self.agents = {
            'issue': IRACAgent(self.llm, prompt_template=ISSUE_PROMPT),
            'rule': IRACAgent(self.llm, prompt_template=RULE_PROMPT),
            'application': IRACAgent(self.llm, prompt_template=APPLICATION_PROMPT),
            'conclusion': IRACAgent(self.llm, prompt_template=CONCLUSION_PROMPT)
        }
        self.nli_model = nli_model

    def run(self, query: str, docs: List[str], threshold: float = 0.5) -> Dict:
        # 1. Retrieval
        context = "\n".join([f"Document {i+1}: {clean_document(doc)}" for i, (_, doc, _) in enumerate(docs)])

        # 2. Parallel IRAC Generation with dependencies
        irac_outputs = {}
        irac_outputs['issue'] = self.agents['issue'].generate(query=query, legal_materials=context)
        logger.debug(f"IRAC Issue: {irac_outputs['issue']}")
        torch.cuda.empty_cache()

        irac_outputs['rule'] = self.agents['rule'].generate(query=query, legal_materials=context)
        logger.debug(f"IRAC Rule: {irac_outputs['rule']}")
        torch.cuda.empty_cache()

        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
        logger.debug(f"IRAC Application: {irac_outputs['application']}")
        torch.cuda.empty_cache()

        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
        logger.debug(f"IRAC Conclusion: {irac_outputs['conclusion']}")
        torch.cuda.empty_cache()

        # 3. Consistency Graph / ICS
        with torch.inference_mode():
            cg = ConsistencyGraph(self.nli_model)
            ics, edge_scores = cg.compute_ics(irac_outputs)
        logger.debug(f"ICS: {ics}, Edge Scores: {edge_scores}")
        
        # 4. Regeneration if ICS < threshold
        count = 0
        while ics < threshold:
            logger.debug(f"ICS {ics} is below threshold {threshold}. Regenerating...")
            regenerated = False
            for (src, tgt), score in edge_scores.items():
                if score < threshold:
                    regenerated = True
                    if src == 'issue' and tgt == 'rule':
                        # regenerate rule and downstream nodes
                        irac_outputs['rule'] = self.agents['rule'].generate(query=query, legal_materials=context)
                        logger.debug(f"Regenerated Rule: {irac_outputs['rule']}")

                        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
                        logger.debug(f"Regenerated Application: {irac_outputs['application']}")

                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
                        logger.debug(f"Regenerated Conclusion: {irac_outputs['conclusion']}")
                    elif src == 'rule' and tgt == 'application':
                        # regenerate application and downstream node
                        irac_outputs['application'] = self.agents['application'].generate(query=query, rules=irac_outputs['rule'])
                        logger.debug(f"Regenerated Application: {irac_outputs['application']}")

                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
                        logger.debug(f"Regenerated Conclusion: {irac_outputs['conclusion']}")
                    elif src == 'application' and tgt == 'conclusion':
                        # regenerate conclusion only
                        irac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, application=irac_outputs['application'])
                        logger.debug(f"Regenerated Conclusion: {irac_outputs['conclusion']}")

            # if any regeneration occurred, recompute ICS and edge scores
            if regenerated:
                with torch.inference_mode():
                    ics, edge_scores = cg.compute_ics(irac_outputs)
                logger.debug(f"ICS: {ics}, Edge Scores: {edge_scores}")
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

class IAC:
    def __init__(self, llm_model, nli_model):
        self.llm = llm_model
        self.agents = {
            'issue': IACAgent(self.llm, prompt_template=IAC_ISSUE_PROMPT),
            'analysis': IACAgent(self.llm, prompt_template=IAC_ANALYSIS_PROMPT),
            'conclusion': IACAgent(self.llm, prompt_template=IAC_CONCLUSION_PROMPT)
        }
        self.nli_model = nli_model

    def run(self, query: str, docs: List[str], threshold: float = 0.5) -> Dict:
        # 1. Retrieval
        context = "\n".join([f"Document {i+1}: {clean_document(doc)}" for i, (_, doc, _) in enumerate(docs)])

        # 2. Parallel IRAC Generation with dependencies
        iac_outputs = {}
        iac_outputs['issue'] = self.agents['issue'].generate(query=query, legal_materials=context)
        logger.debug(f"IAC Issue: {iac_outputs['issue']}")
        torch.cuda.empty_cache()

        iac_outputs['analysis'] = self.agents['analysis'].generate(query=query, legal_materials=context)
        logger.debug(f"IAC Application: {iac_outputs['analysis']}")
        torch.cuda.empty_cache()

        iac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, legal_materials=context)
        logger.debug(f"IAC Conclusion: {iac_outputs['conclusion']}")
        torch.cuda.empty_cache()

        # 3. Consistency Graph / ICS
        with torch.inference_mode():
            cg = ConsistencyGraph(self.nli_model)
            ics, edge_scores = cg.compute_ics(iac_outputs)
        logger.info(f"ICS: {ics}, Edge Scores: {edge_scores}")
        
        # 4. Regeneration if ICS < threshold
        count = 0
        while ics < threshold:
            logger.info(f"ICS {ics} is below threshold {threshold}. Regenerating...")
            regenerated = False
            for (src, tgt), score in edge_scores.items():
                if score < threshold:
                    regenerated = True
                    if src == 'issue' and tgt == 'analysis':
                        # regenerate rule and downstream nodes
                        iac_outputs['analysis'] = self.agents['analysis'].generate(query=query, legal_materials=context)
                        logger.debug(f"Regenerated Analysis: {iac_outputs['analysis']}")

                        iac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, legal_materials=context)
                        logger.debug(f"Regenerated Conclusion: {iac_outputs['conclusion']}")

                    elif src == 'analysis' and tgt == 'conclusion':
                        # regenerate conclusion only
                        iac_outputs['conclusion'] = self.agents['conclusion'].generate(query=query, legal_materials=context)
                        logger.debug(f"Regenerated Conclusion: {iac_outputs['conclusion']}")

            # if any regeneration occurred, recompute ICS and edge scores
            if regenerated:
                with torch.inference_mode():
                    ics, edge_scores = cg.compute_ics(iac_outputs)
                logger.debug(f"ICS: {ics}, Edge Scores: {edge_scores}")
                count += 1

                if count > 3:
                    break

        return iac_outputs
    
class IACAgent:
    def __init__(self, llm_model: str, prompt_template: str):
        self.llm_model = llm_model
        self.prompt_template = prompt_template

    def generate(self, query: str, legal_materials: str = "", analysis: str = "") -> str:
        if "{analysis}" in self.prompt_template:
            prompt = self.prompt_template.format(query=query, application=analysis)
            response = self.llm_model.query(prompt)
        else:
            prompt = self.prompt_template.format(query=query, legal_materials=legal_materials)
            response = self.llm_model.query(prompt)

        return response

# ISSUE_PROMPT = (
#     "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
#     "Given the following query and legal materials, identify the central legal issue.\n\n"
#     "After </think> state the central legal issue.\n"
#     "Query:\n"
#     "{query}\n\n"
#     "Legal Materials:\n"
#     "\"{legal_materials}\"\n\n"
#     )

# RULE_PROMPT = (
#     "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
#     "Given the following query and legal materials, state the applicable legal rules.\n"
#     "After </think> state the applicable legal rules.\n\n"
#     "Query:\n"
#     "{query}\n\n"
#     "Legal Materials:\n"
#     "\"{legal_materials}\"\n\n"
#     )

# APPLICATION_PROMPT = (
#     "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
#     "Given the rules and query, how do these rules apply to this specific situation?\n"
#     "After </think> state how the rules apply to this specific situation.\n\n"
#     "Query:\n"
#     "{query}\n\n"
#     "Rules:\n"
#     "{rules}\n\n"
#     )

# CONCLUSION_PROMPT = (
#     "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
#     "Based on the application of the rules to the query, label the query either Yes or No.\n"
#     "After </think> only state the answer and DO NOT provide explanations.\n\n"
#     "Query:\n"
#     "{query}\n\n"
#     "Application:\n"
#     "{application}\n\n"
#     )

IAC_ISSUE_PROMPT = (
    "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
    "Given the following query and legal materials, identify the central legal issue.\n\n"
    "After </think> state the central legal issue.\n"
    "Query:\n"
    "{query}\n\n"
    "Legal Materials:\n"
    "\"{legal_materials}\"\n\n"
)

IAC_ANALYSIS_PROMPT = (
    "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
    "Given the following query and legal materials, state the applicable legal analysis.\n"
    "After </think> state the applicable legal analysis.\n\n"
    "Query:\n"
    "{query}\n\n"
    "Legal Materials:\n"
    "\"{legal_materials}\"\n\n"
)

IAC_CONCLUSION_PROMPT = (
    "You are a legal‑analysis assistant.  Think step‑by‑step between <think> and </think>.\n"
    "Given the following query and legal materials, label the query either Yes or No.\n"
    "After </think> only state the answer and DO NOT provide explanations.\n\n"
    "Query:\n"
    "{query}\n\n"
    "Legal Materials:\n"
    "{legal_materials}\n\n"
)