from tqdm.auto import tqdm

import argparse
import logging
import os
import torch
import json

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

from src.retriever.retriever import retrieve, setup_bm25_index

def parse_args():
    parser = argparse.ArgumentParser(description='Legal RAG testing')

    # LLM settings
    parser.add_argument('--model_name', type=str, default='llama7b', help='model to use')
    parser.add_argument('--dataset_name', type=str, default='legalbench', help='dataset to use')

    # RAG settings
    parser.add_argument('--top_k', type=int, default=5, help='Top K documents for retrieval')
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

    # other
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    LOG_NAME = f'{args.dataset_name}-{args.model_name}'
    logging_level = logging.DEBUG if args.debug else logging.INFO

    os.makedirs(f'log', exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        handlers=[logging.FileHandler(f"log/{LOG_NAME}.log"), logging.StreamHandler()],
    )

    logger = logging.getLogger('legal-rag')
    logger.setLevel(logging_level)

    logger.info(f"Starting Legal-RAG Experiment with settings: {args}")

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    if args.dataset_name == "legalbench":
        tasks = ["cuad_affiliate_license-licensee",
                 "cuad_no-solicit_of_employees",
                 "cuad_price_restrictions",
                 "cuad_warranty_duration"]
        split = "train"
        data_tool = dataset_utils.load_data(args.dataset_name, tasks=tasks, split=split)
        dataset = data_tool.get_data()
    else:
        pass

    if args.use_rag:
        bm25, retrieval_documents = setup_bm25_index(
            retrieval_dataset="theatticusproject/cuad-qa",
            dataset_dir="bm25_data",
        )

    # Create LLM
    llm = create_model(args.model_name)

    os.makedirs("results/rag", exist_ok=True)

    evaluation_score = []

    for task_name, df in dataset.items():
        logger.info(f"Processing task: {task_name}, {len(df)} records")

        prompts = data_tool.create_prompts(task_name, df)
        response_list = []
        for prompt in tqdm(prompts, desc=f"Processing {task_name}", unit="query"):
            logger.debug(f"Processing query: {prompt}")

            if args.use_rag:
                logger.debug(f"Retrieving documents for query: {prompt}")
                results = retrieve(prompt, bm25, retrieval_documents, k=args.top_k)
                retrieved_docs = [doc for idx, doc, score in results]
                context = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
                logger.debug(f"Retrieved documents: {retrieved_docs}")

                rag_prompt = f"""
                Answer the query using the provided context.
                
                Context:
                {context}
                
                Query:
                {prompt}
                """
                logger.debug(f"RAG prompt:\n{rag_prompt}")
                response = llm.query(rag_prompt)
            else:
                response = llm.query(prompt)
            
            logger.debug(f"Model response: {response}")
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