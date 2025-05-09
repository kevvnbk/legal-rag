from tqdm.auto import tqdm
import argparse, logging, os, torch, json, random
from collections import Counter

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

from src.retriever.retriever import retrieve, setup_faiss_index
from src.defense import MajorityVoting3
from src.pirac.irac import IRAC#, PIRAC
from src.pirac.irac_batch import IRACBatch

from legalbench.tasks import TASKS

import random
import torch
import numpy as np  # 향후 사용 대비

# 랜덤 시드 고정
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU 연산의 결정론 보장
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Legal RAG hyperparam sweep & Testing')
    
    # LLM settings
    parser.add_argument('--model_name', type=str, default='deepseek-r1-1.5b', choices=['llama7b', 'llama3.2-1b', 'llama3qa-8b', 'deepseek-r1-1.5b', 'saul7b'], help='model to use')
    parser.add_argument('--dataset_name', type=str, default='barexam', choices=['legalbench', 'barexam'], help='dataset to use')

    # Defense
    parser.add_argument('--defense', type=str, default='none', choices=['none', 'voting'], help='defense method to use')

    # RAG settings
    parser.add_argument('--top_k', type=int, nargs='+', default=[3], help='Top K documents for retrieval')  # 제일 처음 retriveve할 document의 수
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

    # IRAC settings
    parser.add_argument('--use_irac', action='store_true', help='Enable IRAC')

    # other
    parser.add_argument('--debug', action='store_true', help='debug mode')

    return parser.parse_args()

def main():
    set_seed(42)  # 💡 여기서 시드 고정
    args = parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    top_k = 3

    if args.use_irac:
        LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.defense}-IRAC-k{top_k}"
    else:
        LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.defense}-k{top_k}"
    os.makedirs("log", exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(
        # level=logging_level,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        handlers=[logging.FileHandler(f"log/{LOG_NAME}.log"), logging.StreamHandler()],
        # force=True
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)
    logger.info(f"Full argument: {args}")
    logger.info(f"Using device: {device}")
    logger.info(f"Run start: top_k={top_k}")

    os.makedirs(f"results/{LOG_NAME}", exist_ok=True)

    if args.dataset_name == "barexam":
        split = "test"
        data_tool = dataset_utils.load_data(args.dataset_name, split=split)
        dataset = data_tool.get_data()

    if args.use_rag:
        json_files = ["corpus/state_code.jsonl", "corpus/uscode.jsonl", "corpus/canadian_decisions.jsonl", "corpus/cc_casebooks.jsonl",
                      "corpus/cfr.jsonl", "corpus/courtlisteneropinions_sampled.jsonl", "corpus/echr.jsonl", "corpus/eurlex.jsonl",
                      "corpus/taxrulings.jsonl"]
        faiss_index, retrieval_documents, retriever_model = setup_faiss_index(json_files)

    # Create LLM
    llm = create_model(args.model_name)

    if args.use_irac:
        nli_model = create_model("deberta")
        irac = IRACBatch(
            llm_model=llm,
            nli_model=nli_model
        )
    
    if args.defense == 'voting':
        defended_llm = MajorityVoting3(llm)

    # ─── 스윕 루프 ───
    for top_k in args.top_k:
        no_defense = args.defense == 'none' or top_k <= 0
        no_defense = args.defense == 'none' or top_k <= 0

        response_list = []
        labels = dataset["answer"].unique().tolist()
        for _, row in tqdm(dataset.iterrows(), desc="Processing dataset", unit="row"):    
            prompt = data_tool.create_prompt(row)
            if args.use_rag:
                logger.info(f"Retrieving documents for query: {prompt}")
                retrieved_docs = retrieve(prompt, faiss_index, retrieval_documents, retriever_model, top_k=top_k)
                    
                if args.use_irac:
                    logger.info(f"Using IRAC_Batch")

                    if not no_defense:
                        resp, cert = defended_llm.query(
                            retrieved_docs = retrieved_docs,
                            prompt = prompt,
                            labels=labels,
                            corruption_size=args.corruption_size,
                            irac=irac
                        )
                    else:
                        irac_outputs = irac.run(query=prompt , docs=retrieved_docs)
                        resp, cert = irac_outputs, None
                        logger.info(f"Response: {resp}")
                        
                    response_list.append({"query": prompt, "response": resp, "certificate": cert})
                    
                # Not Using IRAC
                else:
                    context = "\n".join([f"Document {i+1}: {doc}" for i, (doc, _) in enumerate(retrieved_docs)])
                        
                    rag_prompt = (
                                "You are a legal reasoning assistant. Using the legal materials below, "
                                # "answer the question with one word, either 'Yes' or 'No'.\n"
                                f"Answer with exactly one of the following options: {', '.join(labels)}."
                                "Query:\n"
                                f"{prompt}\n\n"
                                "Legal Materials:\n"
                                f"{context}\n\n"
                    )
                    
                    logger.debug(f"RAG prompt:\n{rag_prompt}")

                    if not no_defense:
                        resp, cert = defended_llm.query(
                            retrieved_docs,
                            prompt,
                            corruption_size=args.corruption_size,
                        )
                    else:
                        #context = "\n".join([f"Document {i+1}: {doc}" for i, (_,doc,_) in enumerate(retrieved_docs)])
                        #rag_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
                        resp, cert = llm.query(rag_prompt), None

                    response_list.append({"query": prompt, "response": resp, "certificate": cert})
                    
            else:
                new_prompt = (
                    "You are a legal reasoning assistant."
                    f"Answer with exactly one of the following options: {', '.join(labels)}."
                    # "Start your response with </think>\n"
                    f"{prompt}\n"
                )
                resp = llm.query(new_prompt)
                cert = None
            
                response_list.append({"query": prompt, "response": resp, "certificate": cert})
            logger.info(f"Response: {resp}")

        with open(f"results/{LOG_NAME}/answers.json", "w") as f:
            json.dump(response_list, f, indent=2)

        predictions = [entry["response"] for entry in response_list]

        gold_answers = dataset["answer"].tolist()
        correct_count = sum(1 for pred, gold in zip(predictions, gold_answers) if pred.strip() == gold.strip())

        logger.info(f"Correct Count: {correct_count}, Total: {correct_count}/{len(gold_answers)}")
        
        evaluation_score = {
            "correct_count": correct_count,
            "total_count": len(gold_answers),
            "accuracy": correct_count / len(gold_answers) * 100,
        }
        with open(f"results/{LOG_NAME}/eval.json", "w") as f:
            json.dump(evaluation_score, f, indent=4)
        logger.info(f"Run complete: {LOG_NAME}")

                
if __name__ == '__main__':
    main()
