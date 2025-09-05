#!/usr/bin/env python3
"""
Pre-process the ValuePrism moral-reasoning dataset to VERL-ready parquet.

Just run:
    python examples/data_preprocess/valueprism_ex.py
"""

import os
import random
import pandas as pd
from datasets import Dataset
from verl.utils.hdfs_io import makedirs, copy  # only used if HDFS is enabled

# Configuration
CSV_PATH = "/p/project1/westai0052/self_rewarding/verl/benchmark/dataset/final_train.csv"
LOCAL_DIR = "/p/project1/westai0052/self_rewarding/verl/train_dataset/sft_ex"
HDFS_DIR = None  # Optional, set to string path if needed (e.g., "/hdfs/path")
SPLIT_RATIO = 0.95


# Prompt wrapper
def wrap_prompt(original: str) -> str:
    return (
        f"Provide a structured multi-perspectives analysis within 450 words on this topic: {original}\n"
        "Make sure each sentence you write can represent one perspective from your analysis.\n"         
        "   Output Format Example:\n"
        "   You may have many sentences for different perspectives. Make sure each of sentence follow this format: In the perspective of <Perspective name>, <your explanation to this aspect>\n"
    )



# Convert each row
def convert_row(row, idx, split):
    return {
        "data_source": "valueprism",
        "prompt": [{"role": "user", "content": row["wrapped_prompt"]}],
        "ability": "moral_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": row["answer"],
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "question": row["prompt"],
            "answer": row["answer"],
        },
        "flattened_prompt": row["wrapped_prompt"],
        "flattened_response": row["answer"],
    }



# Main Pipeline
def add_linebreaks(answer: str) -> str:
    import re
    return re.sub(r'\s*(?<!^)(In the perspective of)', r'\n\1', answer).strip()

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Load and shuffle
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Apply template
    df["wrapped_prompt"] = df["prompt"].apply(wrap_prompt)

    #  Apply formatting to the answers
    df["formatted_answer"] = df["answer"].apply(add_linebreaks)

    #  split
    split_idx = int(len(df) * SPLIT_RATIO)
    splits = {
        "train": df.iloc[:split_idx].reset_index(drop=True),
        "test":  df.iloc[split_idx:].reset_index(drop=True),
    }

    # Convert and save
    for split_name, split_df in splits.items():
        split_df["flattened_prompt"] = split_df["wrapped_prompt"]
        split_df["flattened_response"] = split_df["formatted_answer"]

        def convert_row(row, idx, split=split_name):  # update closure
            return {
                "data_source": "valueprism",
                "prompt": [{"role": "user", "content": row["wrapped_prompt"]}],
                "ability": "moral_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": row["flattened_response"],  # use formatted answer
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": row["prompt"],
                    "answer": row["flattened_response"],  # use formatted answer
                },
                "flattened_prompt": row["wrapped_prompt"],
                "flattened_response": row["flattened_response"],
            }

        ds = Dataset.from_pandas(
            split_df.apply(lambda row: convert_row(row, row.name), axis=1, result_type="expand")
        )
        parquet_path = os.path.join(LOCAL_DIR, f"{split_name}.parquet")
        ds.to_parquet(parquet_path)
        print(f"✓ wrote {parquet_path}  ({len(ds):,} rows)")

    # Optional: HDFS copy
    if HDFS_DIR:
        makedirs(HDFS_DIR)
        copy(src=LOCAL_DIR, dst=HDFS_DIR)
        print(f"✓ copied parquet files to HDFS → {HDFS_DIR}")



if __name__ == "__main__":
    main()
