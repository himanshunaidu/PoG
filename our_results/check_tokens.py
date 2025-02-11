import json
import sys

JSONL_FILE_PATH = "/home/ubuntu/ClaimBenchKG_Baselines/PoG/our_results/PoG_cwq_gpt-3.5-turbo-0125.jsonl"

json_l_output = []
with open(JSONL_FILE_PATH, "r") as jsonl_file:
    for line in jsonl_file:
        json_line = json.loads(line)
        json_l_output.append(json_line)

input_tokens = 0.0
output_tokens = 0.0
for json_line in json_l_output:
    input_tokens += json_line["input_token"]
    output_tokens += json_line["output_token"]

avg_input_tokens = input_tokens/len(json_l_output)
avg_output_tokens = output_tokens/len(json_l_output)

print("Avg input tokens:", avg_input_tokens)
print("Avg output tokens:", avg_output_tokens)

print("Assuming 90k rows more required for processing (40k for ToG, 50k for PoG)")
print("Total input tokens:", avg_input_tokens*90000)
print("Total output tokens:", avg_output_tokens*90000)

print("Costs estimates: With 2.5 per 1m input tokens, 10 per 1m output tokens")
print("Total input costs:", avg_input_tokens*90000*2.5/1000000)
print("Total output costs:", avg_output_tokens*90000*10/1000000)