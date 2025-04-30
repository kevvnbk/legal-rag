import logging
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm
import random

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
        separate_responses = []
        for doc in retrieved_docs:
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
    
class MajorityVoting2(RRAG):
    def query(self, retrieved_docs, prompt, corruption_size, sample_size=3, num_rounds=6, pirac=None):
        separate_responses = []

        # PIRAC 사용
        if not pirac:
            # 1) num_rounds만큼 반복하면서, 매번 retrieved_docs에서 sample_size개를 랜덤 추출해 LLM에 쿼리
            for _ in range(num_rounds):
                # 실제 뽑을 개수는 retrieved_docs 길이에 맞춰 축소
                k = min(sample_size, len(retrieved_docs))
                sampled_docs = random.sample(retrieved_docs, k)

                # 샘플된 문서들을 하나로 묶어 Context 생성
                context = "\n".join([
                    f"Document {i+1}: {doc}" 
                    for i, doc in enumerate(sampled_docs)
                ])
                combined_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"

                # LLM에 한 번 요청하고, 결과를 리스트에 저장
                irac_outputs = pirac.run(query=prompt , docs=combined_prompt)
                response = irac_outputs['conclusion']
                logger.info(f"Response: {response}")
                
                separate_responses.append(response)

            # 2) binary 예측으로 변환
            preds = []
            for resp in separate_responses:
                lower = resp.lower().strip()
                if "yes" in lower:
                    preds.append("yes")
                elif "no" in lower:
                    preds.append("no")
                else:
                    preds.append("no")
            logger.debug(f"Separate binary responses: {preds}")

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

        else:
            # 1) num_rounds만큼 반복하면서, 매번 retrieved_docs에서 sample_size개를 랜덤 추출해 LLM에 쿼리
            for _ in range(num_rounds):
                # 실제 뽑을 개수는 retrieved_docs 길이에 맞춰 축소
                k = min(sample_size, len(retrieved_docs))
                sampled_docs = random.sample(retrieved_docs, k)

                # 샘플된 문서들을 하나로 묶어 Context 생성
                context = "\n".join([
                    f"Document {i+1}: {doc}" 
                    for i, doc in enumerate(sampled_docs)
                ])
                combined_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"

                # LLM에 한 번 요청하고, 결과를 리스트에 저장
                response = self.llm.query(combined_prompt)
                separate_responses.append(response)

            # 2) binary 예측으로 변환
            preds = []
            for resp in separate_responses:
                lower = resp.lower().strip()
                if "yes" in lower:
                    preds.append("yes")
                elif "no" in lower:
                    preds.append("no")
                else:
                    preds.append("no")
            logger.debug(f"Separate binary responses: {preds}")

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