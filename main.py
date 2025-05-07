from tqdm.auto import tqdm

import argparse
import logging
import os
import torch
import json

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

from src.retriever.retriever import retrieve, setup_faiss_index, clean_document
from src.defense import MajorityVoting
from src.attack import PIA, Poison
from src.pirac.irac import IRAC
from src.pirac.irac_batch import IRACBatch

from legalbench.tasks import TASKS

CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

def parse_args():
    parser = argparse.ArgumentParser(description='Legal RAG testing')

    # LLM settings
    parser.add_argument('--model_name', type=str, default='deepseek-r1-1.5b', choices=['llama7b', 'llama3.2-1b', 'llama3qa-8b', 'deepseek-r1-1.5b', 'saul7b'], help='model to use')
    parser.add_argument('--dataset_name', type=str, default='legalbench', help='dataset to use')

    # Attack
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'Poison', 'PIA'], help='attack method to use')
    parser.add_argument('--corruption_size', type=int, default=1, help='number of documents to corrupt')

    # Defense
    parser.add_argument('--defense', type=str, default='none', choices=['none', 'voting'], help='defense method to use')

    # RAG settings
    parser.add_argument('--top_k', type=int, default=3, help='Top K documents for retrieval')
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

    # IRAC settings
    parser.add_argument('--use_irac', action='store_true', help='Enable IRAC')

    # other
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    LOG_NAME = f'retriever-{args.dataset_name}-{args.model_name}-{args.use_rag}-{args.use_irac}'
    logging_level = logging.DEBUG if args.debug else logging.INFO

    os.makedirs(f'log', exist_ok=True)

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        handlers=[logging.FileHandler(f"log/{LOG_NAME}.log", encoding="utf-8"), logging.StreamHandler()],
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    logger.info(f"Starting Legal-RAG Experiment with settings: {args}")

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    if args.dataset_name == "legalbench":
        tasks = TASKS
        split = "train"
        data_tool = dataset_utils.load_data(args.dataset_name, tasks=tasks, split=split)
        dataset = data_tool.get_data()
    else:
        pass

    if args.use_rag:
        json_files = ["corpus/state_code.jsonl", "corpus/uscode.jsonl", "corpus/canadian_decisions.jsonl", "corpus/cc_casebooks.jsonl",
                      "corpus/cfr.jsonl", "corpus/courtlisteneropinons_sampled.jsonl", "corpus/echr.jsonl", "corpus/eurlex.jsonl",
                      "corpus/taxrulings.jsonl"]
        faiss_index, retrieval_documents, model = setup_faiss_index(json_files)

    # Create LLM
    llm = create_model(args.model_name)

    if args.use_irac:
        nli = create_model("deberta")
        irac = IRACBatch(
            llm_model=llm,
            nli_model=nli,
        )

    os.makedirs("results", exist_ok=True)

    evaluation_score = []

    no_defense = args.defense == 'none' or args.top_k<=0
    no_attack = args.attack == 'none' or args.top_k<=0

    if args.defense == 'voting':
        defended_llm = MajorityVoting(llm)

    if no_attack:
        pass
    elif args.attack == 'PIA':
        attacker = PIA(top_k=args.top_k, poison_num=args.corruption_size, repeat=5, poison_order="backward")
    elif args.attack == 'Poison':
        attacker = Poison(top_k=args.top_k, poison_num=args.corruption_size, repeat=5, poison_order="backward")
    else:
        NotImplementedError

    for task_name, df in dataset.items():
        logger.info(f"Processing task: {task_name}, {len(df)} records")

        prompts = data_tool.create_prompts(task_name, df)
        response_list = []
        for prompt in tqdm(prompts, desc=f"Processing {task_name}", unit="query"):
            logger.debug(f"Processing query: {prompt}")

            if args.use_rag:
                logger.debug(f"Retrieving documents for query: {prompt}")
                retrieved_docs = retrieve(prompt, faiss_index, retrieval_documents, model, top_k=args.top_k)

                if args.use_irac:
                    logger.debug(f"Using IRAC")
                    
                    irac_outputs = irac.run(query=prompt , docs=retrieved_docs)
                    # logger.info(f"IRAC outputs: {irac_outputs}")
                    response = irac_outputs
                    logger.info(f"Response: {response}")
                    
                else:
                    # attack
                    if not no_attack:
                        logger.debug(f"Attacking prompt...")
                        retrieved_docs = attacker.attack(retrieved_docs, task_name)
                        context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
                        logger.debug(f"Attacked prompt")
                    else:
                        context = "\n".join(
                            [
                                f"Document {i+1}: {doc.page_content}"
                                for i, (doc, _) in enumerate(retrieved_docs)
                            ]
                        )

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

                    # defense
                    if not no_defense:
                        response, certificate = defended_llm.query(retrieved_docs, prompt, corruption_size=1)
                    # no defense
                    else:
                        response = llm.query(rag_prompt)

            else:
                new_prompt = (
                    "You are a legal reasoning assistant. Answer the question with one word, either 'Yes' or 'No'."
                    f"{prompt}\n"
                    "Answer (Yes or No):"
                )
                response = llm.query(prompt)
            
            logger.debug(f"Model response: {response}")

            if not no_defense:
                response_list.append({"query": prompt, "response": response, "certificate": certificate})
            else:
                response_list.append({"query": prompt, "response": response})

        with open(f"results/{task_name}.json", "w") as f:
            json.dump(response_list, f, indent=4)

        predictions = [entry["response"] for entry in response_list]

        score = evaluate(task_name, predictions, df["answer"].tolist())
        evaluation_score.append({"task": task_name, "score": score})

        logger.info(f"Task: {task_name}, Score: {score}")

    # Save responses
    os.makedirs("evaluation", exist_ok=True)

    with open(f"evaluation/eval.json", "w") as f:
        json.dump(evaluation_score, f, indent=4)

    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main() 