from tqdm.auto import tqdm

import argparse
import logging
import os
import torch
import json

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Legal RAG testing')

    # LLM settings
    parser.add_argument('--model_name', type=str, default='llama7b', help='model to use')
    parser.add_argument('--dataset_name', type=str, default='legalbench', help='dataset to use')

    # RAG settings
    # parser.add_argument('--top_k', type=int, default=10, help='Top K documents for retrieval')
    # parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

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

    logger = logging.getLogger('legalrag-main')
    logger.setLevel(logging_level)

    logger.info(f"Starting Legal-RAG Experiment with settings: {args}")

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    if args.dataset_name == "legalbench":
        tasks = []
        split = "train"
        data_tool = dataset_utils.load_data(args.dataset_name, tasks=tasks, split=split)
        dataset = data_tool.get_data()
    else:
        pass

    # Create LLM
    llm = create_model(args.model_name)

    os.makedirs("results", exist_ok=True)

    evaluation_score = []

    for task_name, df in dataset.items():
        logger.info(f"Processing task: {task_name}, {len(df)} records")

        prompts = data_tool.create_prompts(task_name, df)
        response_list = []
        for prompt in prompts:
            response = llm.query(prompt)
            response_list.append({"query": prompt, "response": response})

        with open(f"results/{LOG_NAME}.json", "w") as f:
            json.dump(response_list, f, indent=4)

        predictions = [entry["response"] for entry in response_list]

        score = evaluate(task_name, predictions, df["answer"].tolist())
        evaluation_score.append({"task": task_name, "score": score})

    # Save responses
    os.makedirs("evaluation", exist_ok=True)

    with open(f"evaluation/eval.json", "w") as f:
        json.dump(evaluation_score, f, indent=4)


if __name__ == '__main__':
    main()