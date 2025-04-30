from tqdm.auto import tqdm
import argparse, logging, os, torch, json, random
from collections import Counter

from src.model import create_model
from src.evaluation import evaluate
from src import dataset_utils

from src.retriever.retriever import retrieve, setup_faiss_index, clean_document
from src.defense import MajorityVoting2
from src.attack import PIA, Poison
from src.pirac.irac import IRAC

from tasks import CUAD_TASKS

def parse_args():
    parser = argparse.ArgumentParser(description='Legal RAG hyperparam sweep & Testing')
    
    # LLM settings
    parser.add_argument('--model_name', type=str, default='llama7b', choices=['llama7b', 'llama3.2-1b', 'llama3qa-8b', 'deepseek-r1-1.5b'], help='model to use')
    parser.add_argument('--dataset_name', type=str, default='legalbench', help='dataset to use')

    # Attack
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'Poison', 'PIA'], help='attack method to use')
    parser.add_argument('--corruption_size', type=int, default=1, help='number of documents to corrupt')

    # Defense
    parser.add_argument('--defense', type=str, default='voting', choices=['none', 'voting'], help='defense method to use')

    # RAG settings
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 10], help='Top K documents for retrieval')
    parser.add_argument('--use_rag', action='store_true', help='Enable RAG')

    # PIRAC settings
    parser.add_argument('--use_pirac', action='store_true', help='Enable PIRAC')
    
    # Random sampling settings
    parser.add_argument('--sample_size_values', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--num_rounds_values', type=int, nargs='+', default=[1, 3])

    # other
    parser.add_argument('--debug', action='store_true', help='debug mode')

    return parser.parse_args()

def main():
    args = parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    if args.dataset_name == "legalbench":
        tasks = ["cuad_affiliate_license-licensee", "cuad_no-solicit_of_employees", "cuad_price_restrictions", "cuad_warranty_duration"]
        split = "test"
        data_tool = dataset_utils.load_data(args.dataset_name, tasks=tasks, split=split)
        dataset = data_tool.get_data()
    else:
        print("There is no datasets")
        pass

    if args.use_rag:
        faiss_index, retrieval_documents, retriever_model = setup_faiss_index("theatticusproject/cuad-qa")

    # Create LLM
    llm = create_model(args.model_name)
    
    if args.use_pirac:
        nli_model = create_model("deberta", device="cpu")
        pirac = IRAC(llm, nli_model)
    
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
                LOG_NAME = f"{args.dataset_name}-{args.model_name}-{args.attack}-{args.defense}-s{sample_size}-r{num_rounds}-k{top_k}"
                os.makedirs("log", exist_ok=True)
                # Clear existing logging handlers to avoid duplicate outputs
                logging.getLogger().handlers.clear()
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
                                    irac_outputs = pirac.run(query=prompt , docs=retrieved_docs)
                                    resp, cert = irac_outputs['conclusion'], None
                                    logger.info(f"Response: {resp}")
                                    
                                #response_list.append({"query": prompt, "response": resp, "certificate": cert})
                                
                            # Not Using PIRAC
                            else:
                                if not no_attack:
                                    retrieved_docs = attacker.attack(retrieved_docs, task_name)

                                if not no_defense:
                                    resp, cert = defended_llm.query(
                                        retrieved_docs,
                                        prompt,
                                        corruption_size=args.corruption_size,
                                        sample_size=sample_size,
                                        num_rounds=num_rounds
                                    )
                                else:
                                    context = "\n".join([f"Document {i+1}: {doc}" for i, (_,doc,_) in enumerate(retrieved_docs)])
                                    rag_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
                                    resp, cert = llm.query(rag_prompt), None

                                #response_list.append({"query": prompt, "response": resp, "certificate": cert})
                                
                        else:
                            resp = llm.query(prompt)
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
