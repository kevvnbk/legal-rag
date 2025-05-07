from tqdm.auto import tqdm
import argparse, logging, os, torch, json, random
from collections import Counter

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

from src.retriever.retriever import retrieve, setup_faiss_index, clean_document
from src.defense import MajorityVoting2
from src.attack import PIA, Poison
from src.pirac.irac import IRAC#, PIRAC
from src.pirac.irac_batch import IRACBatch

from tasks import CUAD_TASKS
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
    parser.add_argument('--dataset_name', type=str, default='legalbench', help='dataset to use')

    # Attack
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'Poison', 'PIA'], help='attack method to use')
    parser.add_argument('--corruption_size', type=int, default=1, help='number of documents to corrupt')

    # Defense
    parser.add_argument('--defense', type=str, default='voting', choices=['none', 'voting'], help='defense method to use')

    # RAG settings
    parser.add_argument('--top_k', type=int, nargs='+', default=[3], help='Top K documents for retrieval')  # 제일 처음 retriveve할 document의 수
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

    # PIRAC or IRAC settings
    #parser.add_argument('--use_pirac', action='store_true', help='Enable PIRAC')
    parser.add_argument('--use_pirac', action='store_true', help='Enable PIRAC')
    parser.add_argument('--use_irac', action='store_true', help='Enable IRAC')
    
    # Random sampling settings
    parser.add_argument('--sample_size_values', type=int, nargs='+', default=[3])    # k개 뽑은 document에서 random으로 선택할 document의 수
    parser.add_argument('--num_rounds_values', type=int, nargs='+', default=[3])     # majority voting에 참여할 투포자의 수

    # other
    parser.add_argument('--debug', action='store_true', help='debug mode')

    return parser.parse_args()

def main():
    set_seed(42)  # 💡 여기서 시드 고정
    args = parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    if args.dataset_name == "legalbench":
        tasks = TASKS
        split = "test"
        data_tool = dataset_utils.load_data(args.dataset_name, tasks=tasks, split=split)
        dataset = data_tool.get_data()
    else:
        print("There is no datasets")
        pass

    if args.use_rag:
        json_files = ["corpus/state_code.jsonl", "corpus/uscode.jsonl", "corpus/canadian_decisions.jsonl", "corpus/cc_casebooks.jsonl",
                      "corpus/cfr.jsonl", "corpus/courtlisteneropinions_sampled.jsonl", "corpus/echr.jsonl", "corpus/eurlex.jsonl",
                      "corpus/taxrulings.jsonl"]
        faiss_index, retrieval_documents, retriever_model = setup_faiss_index(json_files)

    # Create LLM
    llm = create_model(args.model_name)
    
    if args.use_pirac:
        nli_model = create_model("deberta", device="cpu")
        pirac = IRAC(llm, nli_model)
    elif args.use_irac:
        nli_model = create_model("deberta")
        irac = IRACBatch(
            llm_model=llm,
            nli_model=nli_model
        )
    
    if args.defense == 'voting':
        defended_llm = MajorityVoting2(llm)

    # ─── 스윕 루프 ───
    for top_k in args.top_k:
        no_defense = args.defense == 'none' or top_k <= 0
        no_attack = args.attack == 'none' or top_k <= 0

        if no_attack:
            pass
        elif args.attack == 'PIA':
            attacker = PIA(top_k=top_k, poison_num=args.corruption_size, repeat=5, poison_order="backward")
        elif args.attack == 'Poison':
            attacker = Poison(top_k=top_k, poison_num=args.corruption_size, repeat=5, poison_order="backward")
        else:
            raise NotImplementedError

        for sample_size in args.sample_size_values:
            for num_rounds in args.num_rounds_values:
                if args.use_pirac:
                    LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.attack}-{args.defense}-PIRAC-s{sample_size}-r{num_rounds}-k{top_k}"
                elif args.use_irac:
                    LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.attack}-{args.defense}-IRAC-s{sample_size}-r{num_rounds}-k{top_k}"
                else:
                    LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.attack}-{args.defense}-s{sample_size}-r{num_rounds}-k{top_k}"
                os.makedirs("log", exist_ok=True)
                # Clear existing logging handlers to avoid duplicate outputs
                # logging.getLogger().handler
                # s.clear()
                # Clear existing handlers before reconfiguring logging
                root_logger = logging.getLogger()
                if root_logger.hasHandlers():
                    root_logger.handlers.clear()
                logging.basicConfig(
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    handlers=[logging.FileHandler(f"log/{LOG_NAME}.log"), logging.StreamHandler()],
                )
                logger = logging.getLogger(__name__)
                logger.setLevel(logging_level)
                logger.info(f"Full argument: {args}")
                logger.info(f"Using device: {device}")
                logger.info(f"Run start: sample_size={sample_size}, num_rounds={num_rounds}, top_k={top_k}")

                os.makedirs(f"results/{LOG_NAME}", exist_ok=True)
                evaluation_score = []

                for task_name, df in dataset.items():
                    logger.info(f"Processing task: {task_name}, {len(df)} records")
                    
                    prompts = data_tool.create_prompts(task_name, df)
                    response_list = []
                    for prompt in tqdm(prompts, desc=f"Processing {task_name}", unit="query"):
                        logger.debug(f"Retrieving documents for query: {prompt}")
                        
                        if args.use_rag:
                            # Using PIRAC
                            retrieved_docs = retrieve(prompt, faiss_index, retrieval_documents, retriever_model, top_k=top_k)
                            if args.use_pirac:
                                logger.debug(f"Using PIRAC")
                                
                                if not no_attack:
                                    retrieved_docs = attacker.attack(retrieved_docs, task_name)

                                if not no_defense:
                                    resp, cert = defended_llm.query(
                                        retrieved_docs,
                                        prompt,
                                        corruption_size=args.corruption_size,
                                        sample_size=sample_size,
                                        num_rounds=num_rounds,
                                        pirac=pirac
                                    )
                                else:
                                    #context = "\n".join([f"Document {i+1}: {doc}" for i, (_,doc,_) in enumerate(retrieved_docs)])
                                    #rag_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
                                    #retrieved_docs = retrieve(prompt, faiss_index, retrieval_documents, retriever_model, top_k=top_k)
                                    irac_outputs = irac.run(query=prompt , docs=retrieved_docs)
                                    # logger.info(f"IRAC outputs: {irac_outputs}")
                                    resp = irac_outputs
                                    cert = irac_outputs['conclusion']
                                    logger.info(f"Response: {resp}")
                                    
                                #response_list.append({"query": prompt, "response": resp, "certificate": cert})
                                
                            elif args.use_irac:
                                logger.debug(f"Using IRAC_Batch")
                                
                                if not no_attack:
                                    retrieved_docs = attacker.attack(retrieved_docs, task_name)

                                if not no_defense:
                                    resp, cert = defended_llm.query(
                                        retrieved_docs,
                                        prompt,
                                        corruption_size=args.corruption_size,
                                        sample_size=sample_size,
                                        num_rounds=num_rounds,
                                        pirac=irac
                                    )
                                else:
                                    #context = "\n".join([f"Document {i+1}: {doc}" for i, (_,doc,_) in enumerate(retrieved_docs)])
                                    #rag_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
                                    #retrieved_docs = retrieve(prompt, faiss_index, retrieval_documents, retriever_model, top_k=top_k)
                                    irac_outputs = pirac.run(query=prompt , docs=retrieved_docs)
                                    resp, cert = irac_outputs, None
                                    logger.info(f"Response: {resp}")
                                    
                                #response_list.append({"query": prompt, "response": resp, "certificate": cert})
                                
                            # Not Using (P)IRAC
                            else:
                                if not no_attack:
                                    logger.debug(f"Attacking prompt...")
                                    retrieved_docs = attacker.attack(retrieved_docs, task_name)
                                    context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
                                    logger.debug(f"Attacked prompt")
                                else:
                                    context = "\n".join([f"Document {i+1}: {clean_document(doc)}" for i, (_, doc, _) in enumerate(retrieved_docs)])
                                    
                                rag_prompt = (
                                        "You are a legal reasoning assistant. Using the legal materials below, "
                                        "answer the question with one word, either 'Yes' or 'No'.\n"
                                        "Query:\n"
                                        f"{prompt}\n\n"
                                        "Legal Materials:\n"
                                        f"\"{context}\"\n\n"
                                        "Answer (Yes or No):"
                                )
                                
                                logger.debug(f"RAG prompt:\n{rag_prompt}")

                                if not no_defense:
                                    resp, cert = defended_llm.query(
                                        retrieved_docs,
                                        prompt,
                                        corruption_size=args.corruption_size,
                                        sample_size=sample_size,
                                        num_rounds=num_rounds
                                    )
                                else:
                                    #context = "\n".join([f"Document {i+1}: {doc}" for i, (_,doc,_) in enumerate(retrieved_docs)])
                                    #rag_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
                                    resp, cert = llm.query(rag_prompt), None

                                #response_list.append({"query": prompt, "response": resp, "certificate": cert})
                                
                        else:
                            new_prompt = (
                                "You are a legal reasoning assistant. Answer the question with one word, either 'Yes' or 'No'."
                                f"{prompt}\n"
                                "Answer (Yes or No):"
                            )
                            resp = llm.query(new_prompt)
                            cert = None
                        
                        response_list.append({"query": prompt, "response": resp, "certificate": cert})
                        logger.info(f"Response: {resp}")

                    with open(f"results/{LOG_NAME}/{task_name}.json", "w") as f:
                        json.dump(response_list, f, indent=2)

                    predictions = [entry["response"] for entry in response_list]
                    score = evaluate(task_name, predictions, df["answer"].tolist())
                    evaluation_score.append({"task": task_name, "score": score})
                    logger.info(f"Task: {task_name}, Score: {score}")

                with open(f"results/{LOG_NAME}/eval.json", "w") as f:
                    json.dump(evaluation_score, f, indent=4)
                logger.info(f"Run complete: {LOG_NAME}")

if __name__ == '__main__':
    main()
