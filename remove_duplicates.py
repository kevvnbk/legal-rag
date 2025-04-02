import json
from collections import Counter

with open("faiss_data/chunked_dataset.json", "r") as f:
    data = json.load(f)

unique_entries = list(dict.fromkeys(data))

with open("faiss_data/chunked_dataset_no_duplicates.json", "w") as f:
    json.dump(unique_entries, f, indent=4)

print(f"Removed duplicates. Original count: {len(data)}, Unique count: {len(unique_entries)}")