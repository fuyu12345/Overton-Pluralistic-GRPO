"""
This script merges LLM-generated summaries with core answers into a
single training dataset. It:
  1. Loads summaries (situation + response) and training data (prompt + answer).
  2. Cleans and formats responses (capitalizes sentences, adds line breaks).
  3. Merges the two datasets on `prompt == situation`.
  4. Wraps each merged entry into a structured template with
     <core perspectives> and <summary> tags.
  5. Saves the final dataset for supervised fine-tuning (SFT).
"""


import re
import pandas as pd

#Config 
SUMMARIES_CSV   = "/p/project1/westai0052/self_rewarding/verl/summary/llm_summaries_tem0.1_topp0.6_full.csv"
FINAL_TRAIN_CSV = "/p/project1/westai0052/self_rewarding/verl/benchmark/dataset/final_train.csv"
OUTPUT_CSV      = "train_summary_sft_full.csv"

#Helpers 
def add_linebreaks(answer: str) -> str:
    """
    Insert a linebreak before each 'In the perspective of' except
    at the very start.
    """
    return re.sub(r'\s*(?<!^)(In the perspective of)', r'\n\1', answer).strip()

def capitalize_sentences(text: str) -> str:
    """
    Capitalize the first character of each sentence in `text`.
    We split on punctuation+space to detect sentence boundaries.
    """
    parts = re.split(r'([\.!\?]\s+)', text)
    out = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        sep  = parts[i+1] if i+1 < len(parts) else ""
        if sent:
            sent = sent[0].upper() + sent[1:]
        out.append(sent + sep)
    return "".join(out).strip()

# Load and preprocess
# Summaries CSV: has columns "situation", "response"
df_sum = pd.read_csv(SUMMARIES_CSV, usecols=["situation", "response"])
df_sum["response"] = df_sum["response"].astype(str).apply(capitalize_sentences)

# 2) Final train CSV: has columns "prompt", "answer"
df_train = pd.read_csv(FINAL_TRAIN_CSV, usecols=["prompt", "answer"])
df_train = df_train.rename(columns={"answer": "core_answer"})

# Merge on prompt == situation 
df = pd.merge(
    df_train,
    df_sum,
    left_on="prompt",
    right_on="situation",
    how="inner"
)

# Build the combined template in a new answer column
def build_template(row):
    core = add_linebreaks(row["core_answer"])
    summary = row["response"]
    return (
        "<core perspectives>\n"
        f"{core}\n"
        "</core perspectives>\n\n"
        "<summary>\n"
        f"{summary}\n"
        "</summary>"
    )

df["answer"] = df.apply(build_template, axis=1)

# Final output: keep prompt + answer 
out_df = df[["prompt", "answer"]]
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
