import logging
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm

from transformers import StoppingCriteriaList, MaxLengthCriteria 

logger = logging.getLogger(__name__)

INJECTION = True

class RRAG:
    def __init__(self, llm):
        self.llm = llm

    def query_undefended(self, data_item):
        query_prompt = self.llm.wrap_prompt()
        response = self.llm.query(query_prompt)
        logger.debug(f'Query prompt: {query_prompt}')
        logger.debug(f'Response: {response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

    def query(self, data_item):
        raise NotImplementedError
    
    def certify(self, data_item, corruption_size):
        raise NotImplementedError
    
    # def _eval_response(self, response, data_item):
    #     answer = data_item['answer']
    #     response = clean_str(response)
    #     for ans in answer:
    #         if clean_str(ans) in response:
    #             return True
    #     return False
    
class MajorityVoting(RRAG):
    def query(self, retrieved_docs, prompt, corruption_size):
        docs = [doc for (_, doc, _) in retrieved_docs]
        separate_responses = []
        for doc in docs:
            combined_prompt = f"Context:\n{doc}\n\nQuery:\n{prompt}"
            response = self.llm.query(combined_prompt)
            separate_responses.append(response)

        preds = []
        for response in separate_responses:
            lower_resp = response.lower().strip()
            if "yes" in lower_resp:
                preds.append("yes")
            elif "no" in lower_resp:
                preds.append("no")
            else:
                preds.append("no")
        logger.debug(f"Separate binary response: {preds}")

        cntr = Counter(preds)

        if not cntr:
            final_pred = "no"
            certificate = False
        else:
            common = cntr.most_common(2)
            final_pred = common[0][0]
            if len(common) == 1:
                delta = common[0][1]
            else:
                delta = common[0][1] - common[1][1]
            
            if INJECTION:
                delta -= sum(1 for x in preds[-corruption_size:] if x == final_pred)
                certificate = delta > corruption_size
            else:
                certificate = delta > 2 * corruption_size
        return final_pred, certificate