#!/usr/bin/env python3
"""
Pre-process the ValuePrism moral-reasoning dataset to VERL-ready parquet.
"""

import os
import random
import pandas as pd
from datasets import Dataset
from verl.utils.hdfs_io import makedirs, copy  # only used if you set use_hdfs = True

# Configuration                                                              
CSV_PATH = "/p/project1/westai0052/self_rewarding/verl/benchmark/dataset/final_train.csv"
LOCAL_DIR = "/p/project1/westai0052/self_rewarding/verl/train_dataset/grpo_data_im_sum_12k"
HDFS_DIR = None  # Set this to a path if you want to copy to HDFS
SPLIT_RATIO = 0.9



# Chat-template                                                              
# def wrap_prompt(original: str) -> str:
#     return (
#         f"Provide a structured multi-perspectives analysis on this topic: {original}\n\n"
#         "### Output Format\n"
#         "<core perspectives>\n"
#         "In the perspective of <Perspective name>, <your explanation to this aspect>\n"
#         "… (repeat for each perspective)\n"
#         "</core perspectives>\n\n"
#         "<summary>\n"
#         "… (one coherent paragraph weaving together all perspectives in an engaging, conversational yet informative tone)\n"
#         "</summary>\n\n"
#         "### Task Instructions\n"
#         "1. Inside `<core perspectives>…</core perspectives>`, you may have multiple sentences for different perspectives; list each perspective on its own line, using:\n"
#         "   `In the perspective of {Perspective name}, {your explanation to this aspect}`\n"
#         "2. Inside `<summary>…</summary>`, write a single, natural paragraph that summarizes all of the above perspectives. Ensure that each sentence in your summary corresponds clearly to each of the original core perspective sentence above.\n"
#         "3. Only output the two tagged sections exactly as shown—do not add any extra commentary or text.\n"
#     )

def wrap_prompt(original: str) -> str:
    return (
        f"Provide a structured comprehensive analysis and your opinions on this topic: {original}\n\n"
    )



# Row conversion                                                             
def convert_row(row, idx, split):
    return {
        "data_source": "valueprism",
        "prompt": [
            {"role": "user", "content": row["wrapped_prompt"]},
        ],
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
    }


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Load and shuffle
    df = pd.read_csv(CSV_PATH)
    df = df[0:12000]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Subsample
    

    #  Apply template
    df["wrapped_prompt"] = df["prompt"].apply(wrap_prompt)

    # Split
    split_idx = int(len(df) * SPLIT_RATIO)
    splits = {
        "train": df.iloc[:split_idx].reset_index(drop=True),
        "test":  df.iloc[split_idx:].reset_index(drop=True),
    }

    # Convert and save
    for split_name, split_df in splits.items():
        ds = Dataset.from_pandas(
            split_df.apply(
                lambda row: convert_row(row, row.name, split_name), axis=1, result_type="expand"
            )
        )
        parquet_path = os.path.join(LOCAL_DIR, f"{split_name}.parquet")
        ds.to_parquet(parquet_path)
        print(f"✓ wrote {parquet_path}  ({len(ds):,} rows)")

    # Optional HDFS copy
    if HDFS_DIR:
        makedirs(HDFS_DIR)
        copy(src=LOCAL_DIR, dst=HDFS_DIR)
        print(f"✓ copied parquet files to HDFS → {HDFS_DIR}")


if __name__ == "__main__":
    main()
