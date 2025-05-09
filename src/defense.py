import logging
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm
import random

from transformers import StoppingCriteriaList, MaxLengthCriteria 

logger = logging.getLogger(__name__)

INJECTION = True

import re

def extract_label(raw: str, labels: list[str]) -> str:
    first = raw.strip().split()[0]
    for label in labels:
        if first.lower().startswith(label.lower()):
            return label
    return first

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
        
class MajorityVoting3(RRAG):
    def query(self, retrieved_docs, prompt, labels, corruption_size, irac=None):
        separate_responses = []
        
        if labels is None:
            labels = ["Yes", "No"]
        
        if irac:
            for doc in retrieved_docs:
                raw_irac_outputs = irac.run(query=prompt , docs=doc)
                response = extract_label(raw_irac_outputs, labels)
                logger.info(f"Label: {response}")
                separate_responses.append(response)

            # # 2) 각 response를 그대로 예측 레이블로 사용
            # preds = [resp.strip() for resp in separate_responses]
            # logger.debug(f"Separate responses: {preds}")

            # 3) 다수결 투표
            preds = separate_responses
            cntr = Counter(preds)
            if not cntr:
                final_pred, certificate = "no", False
            else:
                common = cntr.most_common(2)
                final_pred = common[0][0]
                if len(common) == 1:
                    delta = common[0][1]
                else:
                    delta = common[0][1] - common[1][1]

                # 4) certificate 계산 (기존 로직 유지)
                if INJECTION:
                    delta -= sum(1 for x in preds[-corruption_size:] if x == final_pred)
                    certificate = delta > corruption_size
                else:
                    certificate = delta > 2 * corruption_size

            return final_pred, certificate

        # Not using IRAC
        else: 
            for doc in retrieved_docs:
                sample_docs = doc
                combined_prompt = f"Context:\n{sample_docs}\n\nQuery:\n{prompt}"
                
                # LLM에 한 번 요청하고, 결과를 리스트에 저장
                raw = self.llm.query(combined_prompt)
                label = extract_label(raw, labels)
                logger.info(f"Label: {label}")
                separate_responses.append(label)

            # 2) 각 response를 그대로 예측 레이블로 사용
            #preds = [resp.strip() for resp in separate_responses]
            preds = separate_responses
            logger.debug(f"Separate responses: {preds}")

            # 3) 다수결 투표
            cntr = Counter(preds)
            if not cntr:
                final_pred, certificate = "no", False
            else:
                common = cntr.most_common(2)
                final_pred = common[0][0]
                if len(common) == 1:
                    delta = common[0][1]
                else:
                    delta = common[0][1] - common[1][1]

                # 4) certificate 계산 (기존 로직 유지)
                if INJECTION:
                    delta -= sum(1 for x in preds[-corruption_size:] if x == final_pred)
                    certificate = delta > corruption_size
                else:
                    certificate = delta > 2 * corruption_size

            return final_pred, certificate
        
        