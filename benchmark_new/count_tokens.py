import os
import pandas as pd
from transformers import AutoTokenizer

# LLM_MODEL= "Qwen3-8B"
# MODE="explicit"
LLM_MODEL= "modular_pluralism"
MODE="raw"

CANDIDATE_DIR = f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL}/{MODE}"
# CANDIDATE_DIR ="/home/zczlyf7/Overton-PPO/op-grpo-results/qwen2.5-3b-im-rewarddown-unique"
# CANDIDATE_DIR ="/home/zczlyf7/Overton-PPO/steps_results/qwen1.5b/150steps"

CANDIDATE_TPL = "test_{label}_gen.csv"  
# CANDIDATE_TPL = "qwen2.5-1.5b-im-{label}.csv"


# LABELS = ["5p", "6p", "7p", "8p", "9p", "10p","gt10p"]
LABELS = ["5p","10p"]

CANDIDATE_COL = "answers"


# Load tokenizer (adjust to your model)
tokenizer = AutoTokenizer.from_pretrained("/scratch/zczlyf7/HF_models/Qwen2.5-1.5B-Instruct")

label_avg_tokens = {}
all_tokens = []

for label in LABELS:
    file_path = os.path.join(CANDIDATE_DIR, CANDIDATE_TPL.format(label=label))
    df = pd.read_csv(file_path)
    
    tokens_per_row = [
        len(tokenizer.tokenize(str(ans))) 
        for ans in df[CANDIDATE_COL].dropna()
    ]
    
    avg_tokens = sum(tokens_per_row) / len(tokens_per_row) if tokens_per_row else 0
    label_avg_tokens[label] = avg_tokens
    all_tokens.extend(tokens_per_row)

# Print results
for label, avg in label_avg_tokens.items():
    print(f"{label}: {avg:.2f} tokens on average")

overall_avg = sum(all_tokens) / len(all_tokens) if all_tokens else 0
print(f"Overall average tokens: {overall_avg:.2f}")



