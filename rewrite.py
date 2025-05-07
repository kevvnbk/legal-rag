import os
import json
import re


input_dir = "results/legalbench-llama7b-none-voting-PIRAC-s1-r1-k1"
output_dir = "results/re_legalbench-llama7b-none-voting-PIRAC-s1-r1-k1"
os.makedirs(output_dir, exist_ok=True)

# response 필드 정제 함수
def extract_label(text):
    # 정규 표현식으로 Yes 또는 No 추출
    match = re.search(r'\b(Yes|No)\b', text)
    return match.group(1) if match else "No"  # 없으면 보수적으로 No 처리

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        if filename != "eval.json":
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                item["response"] = extract_label(item.get("response", ""))

            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Processed {filename}")