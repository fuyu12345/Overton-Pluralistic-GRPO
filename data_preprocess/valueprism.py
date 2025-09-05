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
LOCAL_DIR = "/p/project1/westai0052/self_rewarding/verl/train_dataset/grpo_data_16k_implicit"
HDFS_DIR = None  # Set this to a path if you want to copy to HDFS
SPLIT_RATIO = 0.95



# Chat-template                                                              
def wrap_prompt(original: str) -> str:
    return (
     f"Provide a structured comprehensive analysis and your opinions on this topic: {original}\n"
        "Make sure your responses are in a paragraph, with No bullet points format\n"
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
    df = df[0:16000]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Subsample
    

    # Apply template
    df["wrapped_prompt"] = df["prompt"].apply(wrap_prompt)

    # Split
    split_idx = int(len(df) * SPLIT_RATIO)
    splits = {
        "train": df.iloc[:split_idx].reset_index(drop=True),
        "test":  df.iloc[split_idx:].reset_index(drop=True),
    }

    #  Convert and save
    for split_name, split_df in splits.items():
        ds = Dataset.from_pandas(
            split_df.apply(
                lambda row: convert_row(row, row.name, split_name), axis=1, result_type="expand"
            )
        )
        parquet_path = os.path.join(LOCAL_DIR, f"{split_name}.parquet")
        ds.to_parquet(parquet_path)
        print(f"✓ wrote {parquet_path}  ({len(ds):,} rows)")

    #  Optional HDFS copy
    if HDFS_DIR:
        makedirs(HDFS_DIR)
        copy(src=LOCAL_DIR, dst=HDFS_DIR)
        print(f"✓ copied parquet files to HDFS → {HDFS_DIR}")


if __name__ == "__main__":
    main()
