import matplotlib.pyplot as plt
import numpy as np
import os
import json


results_dir = "llada/results/GSAI-ML__LLaDA-8B-Instruct"

all_jsons = [f for f in os.listdir(results_dir) if f.endswith(".json")]
results_to_configs = {}

BL_to_acc = {}
BL_to_time = {}
for json_file in all_jsons:
    with open(os.path.join(results_dir, json_file), "r") as f:
        result_data = json.load(f)
    metadata = result_data["configs"]["gsm8k"]["metadata"] 
    block_length = metadata["block_length"]
    gsm8k_acc = result_data["results"]["gsm8k"]["exact_match,flexible-extract"]
    BL_to_acc[block_length] = gsm8k_acc
    BL_to_time[block_length] = int(float(result_data["total_evaluation_time_seconds"]))
    
BL_to_acc = dict(sorted(BL_to_acc.items()))
BL_to_time = dict(sorted(BL_to_time.items()))
# plot accuracy vs. block length
plt.figure(figsize=(8,6))
plt.plot(BL_to_acc.keys(), BL_to_acc.values(), marker='o')
plt.title("GSM8K Accuracy vs. Block Length (LLaDA-8B-Instruct)")
plt.xlabel("Block Length")
plt.ylabel("GSM8K Accuracy (%)")
plt.xticks(list(BL_to_acc.keys()))
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("llada/vis_block_length_gsm8k_accuracy.png")
plt.show()
# plot evaluation time vs. block length
plt.figure(figsize=(8,6))
plt.plot(BL_to_time.keys(), BL_to_time.values(), marker='o', color='orange')
plt.title("Evaluation Time vs. Block Length (LLaDA-8B-Instruct)")
plt.xlabel("Block Length")
plt.ylabel("Total Evaluation Time (seconds)")
plt.xticks(list(BL_to_time.keys()))
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("llada/vis_block_length_evaluation_time.png")
plt.show()  