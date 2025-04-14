from datasets import load_dataset
import json

def evaluate_correctness(response, gold_answer):
    """
    Evaluator M: returns 1 if the gold answer appears in the response, 0 otherwise.
    The check is done case-insensitively.
    """
    return 1 if gold_answer.lower() in response.lower() else 0

def calculate_cacc(response_list, gold_answers):
    """
    For each query, a prediction contributes to certifiable accuracy (cacc)
    if it is both correct (gold answer appears in the response) and the certificate is True.
    
    cacc is computed as:
       cacc = (number of queries with certificate True and correct response) / (total number of queries)
    """
    total = len(response_list)
    certifiable_count = 0
    
    # Replace zip-based iteration with index-based loop
    for i in range(total):
        entry = response_list[i]
        gold_answer = gold_answers[i]
        response = entry.get("response", "")
        certificate = entry.get("certificate", False)
        
        # Check if the response is correct (gold answer is found)
        correct = evaluate_correctness(response, gold_answer)
        
        # Only count if both the response is correct and certificate is True
        if certificate and correct:
            certifiable_count += 1
    
    cacc = certifiable_count / total
    return cacc

def main():
    tasks = [
        "cuad_no-solicit_of_employees",
        "cuad_price_restrictions",
        "cuad_warranty_duration"
    ]

    for task in tasks:
        # Load the dataset for the task (e.g., test split if available)
        dataset = load_dataset("nguha/legalbench", task)
        
        # Read the response list from the JSON file.
        # Note: Opening the file in read ("r") mode.
        with open(f"results/top3/defense-voting/{task}.json", "r") as f:
            response_list = json.load(f)
        
        # Assuming the gold answers are stored in a column "answer" in the test split.
        # Adjust according to your actual dataset structure.
        gold_answers = dataset['test']['answer'] if 'test' in dataset else dataset['train']['answer']
        
        cacc = calculate_cacc(response_list, gold_answers)
        print(f"Task: {task}, Certifiable Accuracy (cacc): {cacc:.4f}")

if __name__ == '__main__':
    main()