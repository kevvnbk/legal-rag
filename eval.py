import os
import json
from src.evaluation import evaluate
from src import dataset_utils

def evaluate_saved_results(results_dir, dataset_name="legalbench"):
    # 사용할 태스크 리스트
    tasks = ["cuad_affiliate_license-licensee", "cuad_no-solicit_of_employees", "cuad_price_restrictions", "cuad_warranty_duration"]

    # 원본 정답 데이터 불러오기
    data_tool = dataset_utils.load_data(dataset_name, tasks=tasks, split="test")
    dataset = data_tool.get_data()

    evaluation_score = []
    for task in tasks:
        pred_path = os.path.join(results_dir, f"{task}.json")
        if not os.path.exists(pred_path):
            print(f"[!] Prediction file not found: {pred_path}")
            continue

        with open(pred_path, "r") as f:
            predictions = json.load(f)

        predicted_labels = [item["response"] for item in predictions]
        gold_labels = dataset[task]["answer"].tolist()

        score = evaluate(task, predicted_labels, gold_labels)
        evaluation_score.append({"task": task, "score": score})
        print(f"[✓] {task}: {score}")

    # 결과 저장
    with open(os.path.join(results_dir, "eval.json"), "w") as f:
        json.dump(evaluation_score, f, indent=2)
    print(f"[✔] Evaluation completed and saved to {results_dir}/eval.json")

# 예시 실행
if __name__ == "__main__":
    # 측정하고 싶은 결과 폴더 이름만 바꾸면 됨
    LOG_NAME = "re_legalbench-deepseek-r1-1.5b-none-voting-s3-r3-k10"
    evaluate_saved_results(results_dir=f"results/{LOG_NAME}")