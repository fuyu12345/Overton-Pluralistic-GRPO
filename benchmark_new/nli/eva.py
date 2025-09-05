#!/usr/bin/env python3
import os
import re
import time
import glob
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sentence_splitter import SentenceSplitter


# CONFIG — EDIT HERE
# Lists of files or glob patterns; they are expanded and paired by ORDER.
PROMPTS_INPUTS = [
#    "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_5p.csv",
   "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_6p.csv",
   "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_7p.csv",
   "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_8p.csv",
#    "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_9p.csv",
#    "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_10p.csv",
#    "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_gt10p.csv",
]

LLM_MODEL_NAME = "GPT-OSS"
MODE = "explicit"

ANSWERS_INPUTS = [
    # "/home/zczlyf7/Overton-PPO/steps_results/llama/150steps/qwen2.5-1.5b-im-5p.csv",
    # "/home/zczlyf7/Overton-PPO/steps_results/llama/150steps/qwen2.5-1.5b-im-10p.csv",


    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-5p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-6p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-7p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-8p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-9p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-10p.csv",
    # "/home/zczlyf7/Overton-PPO/op-grpo-results/llama-3b-im-rewarddown-unique-12k/qwen2.5-1.5b-im-gt10p.csv",

    # f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_5p_gen.csv",
    f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_6p_gen.csv",
    f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_7p_gen.csv",
    f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_8p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_9p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_10p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL_NAME}/{MODE}/test_gt10p_gen.csv",

    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_5p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_6p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_7p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_8p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_9p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_10p_gen.csv",
    # f"/home/zczlyf7/Overton-PPO/workflow_350/{LLM_MODEL_NAME}/{MODE}/test_gt10p_gen.csv",
    # match order/count of PROMPTS_INPUTS
    # e.g. "/home/zczlyf7/Overton-PPO/workflow/Qwen3-8B/explicit/test_10p_gen.csv",
    # e.g. "/home/zczlyf7/Overton-PPO/workflow/**/explicit/test_*_gen.csv",
]

# Column names
PROMPT_TEXT_COL  = "prompt"   # in each prompts CSV
STRUCTURED_COL   = "answer"   # in each prompts CSV (parse VRD/explanations from here)
ANSWERS_TEXT_COL = "answers"  # in each answers CSV (model outputs to score)

# NLI & evaluation settings
MODEL_NAME  = "microsoft/deberta-v2-xlarge-mnli"
DEVICE      = 0          # GPU id; use -1 for CPU
BATCH_SIZE  = 300        # tune for your VRAM
THRESHOLD   = 0.33       # Accuracy@threshold

# Optional: cap rows for quick runs (None = all)
MAX_ROWS    = None

# Where to save the summary of all runs
OUTPUT_SUMMARY_PATH = f"output_oss/batch_eval_{LLM_MODEL_NAME}_{MODE}.csv"
os.makedirs(os.path.dirname(OUTPUT_SUMMARY_PATH), exist_ok=True)


# HELPERS
def parse_answer_blocks(ans: str):
    """Parse: 'In the perspective of X, expl...\nIn the perspective of Y, expl...\n' -> (vrd_list, expl_list)."""
    if not isinstance(ans, str):
        return [], []
    pattern = r"In the perspective of (.*?), (.*?)(?=\nIn the perspective of|\Z)"
    pairs   = re.findall(pattern, ans, flags=re.DOTALL)
    vrd     = [p[0].strip() for p in pairs]
    expl    = [p[1].strip().replace("\n", " ") for p in pairs]
    return vrd, expl

def build_items(df_prompts: pd.DataFrame, df_answers: pd.DataFrame):
    """Row-wise combine prompts+answers into items for evaluation."""
    if MAX_ROWS is not None:
        df_prompts = df_prompts.head(MAX_ROWS).copy()
        df_answers = df_answers.head(MAX_ROWS).copy()

    assert len(df_prompts) == len(df_answers), \
        f"Row count mismatch: {len(df_prompts)} prompts vs {len(df_answers)} answers."

    items = []
    for i, (prow, arow) in enumerate(zip(df_prompts.itertuples(), df_answers.itertuples())):
        situation = getattr(prow, PROMPT_TEXT_COL)
        structured_text = getattr(prow, STRUCTURED_COL)
        output_text = getattr(arow, ANSWERS_TEXT_COL)

        vrd, expl = parse_answer_blocks(structured_text)
        items.append({
            "id": int(i),
            "situation": situation,
            "output": output_text if isinstance(output_text, str) else "",
            "vrd": vrd,
            "explanation": expl
        })
    return items

def find_entailment_score(scores_list):
    """Return the entailment score from a pipeline output (fallback to max(score))."""
    for d in scores_list:
        if "entail" in d["label"].lower():
            return float(d["score"])
    return float(max(d["score"] for d in scores_list))

def expand_inputs(inputs):
    """Expand a list of paths/patterns while preserving the user-given order."""
    out = []
    for entry in inputs:
        if any(ch in entry for ch in "*?[]"):
            # For deterministic pairing, sort matches of each pattern
            matches = sorted(glob.glob(entry, recursive=True))
            out.extend(matches)
        else:
            out.append(entry)
    return out

def evaluate_pair(prompts_csv_path, answers_csv_path, nli, splitter):
    """Evaluate one (prompts, answers) pair and return metrics dict."""
    t0 = time.time()
    df_prompts = pd.read_csv(prompts_csv_path)
    df_answers = pd.read_csv(answers_csv_path)

    # Build items
    data = build_items(df_prompts, df_answers)

    scores = []
    for item in tqdm(data, desc=f"Evaluating {os.path.basename(prompts_csv_path)} vs {os.path.basename(answers_csv_path)}", leave=False):
        output_text = item["output"] or ""
        output_sentences = splitter.split(output_text)

        if not output_sentences:
            scores.append(0.0)
            continue

        vrd_explanations = item["explanation"][:len(item["vrd"])]

        if not vrd_explanations:
            scores.append(0.0)
            continue

        # All (explanation, sentence) pairs
        batch_inputs = [{"text": expl, "text_pair": sent}
                        for expl in vrd_explanations for sent in output_sentences]

        results = nli(batch_inputs, batch_size=BATCH_SIZE)

        num_vrd = len(vrd_explanations)
        num_sents = len(output_sentences)
        entail_scores = np.zeros((num_vrd, num_sents), dtype=np.float32)

        for k, scores_list in enumerate(results):
            e_score = find_entailment_score(scores_list)
            i = k // num_sents
            j = k % num_sents
            entail_scores[i, j] = e_score

        max_per_vrd = entail_scores.max(axis=1)
        scores.append(float(max_per_vrd.mean()))

    avg = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    acc = float(np.mean([1 if x > THRESHOLD else 0 for x in scores])) if scores else 0.0
    elapsed = time.time() - t0

    # Calculate standard error (std. dev / sqrt(number of prompts))
    std_error = std / np.sqrt(len(scores)) if len(scores) > 0 else 0.0

    return {
        "prompts_csv": prompts_csv_path,
        "answers_csv": answers_csv_path,
        "rows_evaluated": len(scores),
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "threshold": THRESHOLD,
        "average": avg,
        "std": std,
        "accuracy": acc,
        "std_error": std_error,  # Add standard error here
        "seconds": elapsed,
    }



# MAIN
def main():
    prompts_files = expand_inputs(PROMPTS_INPUTS)
    answers_files = expand_inputs(ANSWERS_INPUTS)

    if len(prompts_files) != len(answers_files):
        raise ValueError(
            f"Pairing requires equal counts: got {len(prompts_files)} prompts files and {len(answers_files)} answers files.\n"
            f"Prompts: {prompts_files}\nAnswers: {answers_files}"
        )

    # Reusable splitter + NLI pipeline
    splitter = SentenceSplitter(language="en")
    nli = transformers.pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=DEVICE,
        return_all_scores=True,
        truncation=True
    )

    all_results = []
    for p_path, a_path in tqdm(list(zip(prompts_files, answers_files)), desc="Batch evaluating pairs"):
        try:
            res = evaluate_pair(p_path, a_path, nli, splitter)
        except AssertionError as e:
            res = {
                "prompts_csv": p_path,
                "answers_csv": a_path,
                "rows_evaluated": 0,
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "threshold": THRESHOLD,
                "average": None,
                "std": None,
                "accuracy": None,
                "seconds": None,
                "error": f"AssertionError: {e}"
            }
        except Exception as e:
            res = {
                "prompts_csv": p_path,
                "answers_csv": a_path,
                "rows_evaluated": 0,
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "threshold": THRESHOLD,
                "average": None,
                "std": None,
                "accuracy": None,
                "seconds": None,
                "error": f"{type(e).__name__}: {e}"
            }
        all_results.append(res)

    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    print(f"\nSaved batch summary to: {OUTPUT_SUMMARY_PATH}")
    cols = ["prompts_csv","answers_csv","rows_evaluated","average","std","accuracy","seconds"]
    if "error" in df_summary.columns:
        cols.append("error")
    print(df_summary[cols])

if __name__ == "__main__":
    main()
