#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_multi_perspectives_no_llm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluate candidate multi‑perspective answers **without** performing any LLM
generation.  Candidates are loaded from a separate CSV and matched to the
reference answers by identical `prompt` strings.

Expected CSV schemas
--------------------
reference_df : prompt, answer
candidate_df : id, prompt, answers     (answers = candidate text)

The similarity logic (sentence splitting, SBERT embeddings, mutual‑best
matching, scoring) is unchanged from the original script.
"""
import os, re, sys, pandas as pd, numpy as np, torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# PARAMETERS 
REFERENCE_CSV = "/home/zczlyf7/Overton-PPO/benchmark_new/dataset/test_5p.csv"
CANDIDATE_CSV = "/home/zczlyf7/Overton-PPO/benchmark_new/extract/grpo_eval_summaries_clean_10p.csv"
SBERT_PATH     = "/scratch/zczlyf7/st_models/MultipleNegativesRankingLoss/hpo_scale_70/checkpoint-1186"

THRESHOLD      = 0.70                # cosine‑sim threshold for a match
MODE           = "explicit"          # only used for log readability
MAX_WORDS      = 450                 # just for log readability
LOGDIR         = "eval_logs_no_llm"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


#  Sentence splitting helpers 
_PERSPECTIVE_SPLIT = re.compile(r"(?=\b(?:In|From) the perspective of\b)",
                                flags=re.IGNORECASE)

def split_candidate(text: str):
    """LLM‑like free‑form candidate => sentence list via punctuation."""
    text = re.sub(r"\s*\[\d+]", "\n", text)
    return [s.strip() for s in re.split(r"[。！？.!?]\s*", text) if s.strip()]

def split_reference(text: str):
    """Reference formatted with 'In/From the perspective of' => chunks."""
    text = re.sub(r"\s*\[\d+]", "\n", text)
    chunks = [s.strip() for s in _PERSPECTIVE_SPLIT.split(text) if s.strip()]
    return chunks

#  mutual‑best matching identical to original
def mutual_best_pairs(sim: np.ndarray, thr: float):
    M = sim.copy(); pairs = []
    while True:
        max_val = np.nanmax(M)
        if np.isnan(max_val) or max_val < thr:
            break
        r, c = np.unravel_index(np.nanargmax(M), M.shape)
        if max_val == np.nanmax(M[r, :]) and max_val == np.nanmax(M[:, c]):
            pairs.append((r, c, max_val))
            M[r, :], M[:, c] = np.nan, np.nan
        else:
            M[r, c] = np.nan
    return pairs

# Main evaluation loop
def main():
    # -------- logging setup --------
    os.makedirs(LOGDIR, exist_ok=True)
    log_path = os.path.join(
        LOGDIR, f"5p_summary_core{datetime.now():%m%d_%H%M%S}.log")
    sys.stdout = sys.stderr = open(log_path, "w")

    print("=== Static parameters ===")
    for k, v in {
        "REFERENCE_CSV": REFERENCE_CSV,
        "CANDIDATE_CSV": CANDIDATE_CSV,
        "SBERT_PATH": SBERT_PATH,
        "THRESHOLD": THRESHOLD,
        "MODE": MODE,
        "MAX_WORDS": MAX_WORDS,
        "DEVICE": DEVICE,
    }.items():
        print(f"{k:>15s} : {v}")
    print()

    # load data
    reference_df  = pd.read_csv(REFERENCE_CSV)
    candidate_df  = pd.read_csv(CANDIDATE_CSV)

    assert {"prompt", "answer"}.issubset(reference_df.columns), \
        "reference CSV must contain prompt & answer columns"
    assert {"prompt", "answers"}.issubset(candidate_df.columns), \
        "candidate CSV must contain prompt & answers columns"

    # map prompt → candidate answer (take first if duplicates)
    cand_map = (candidate_df
                .drop_duplicates("prompt")
                .set_index("prompt")["answers"]
                .to_dict())

    print(f"Loaded {len(reference_df)} reference samples")
    print(f"Loaded {len(candidate_df)} candidate samples\n", flush=True)

    # SBERT model
    sbert = SentenceTransformer(SBERT_PATH, device=DEVICE)

    total = hit_all = 0
    partial_score_sum = 0.0

    # evaluation 
    for idx, row in reference_df.iterrows():
        ref_prompt  = row["prompt"]
        ref_raw     = row["answer"]
        cand_raw    = cand_map.get(ref_prompt)

        if cand_raw is None:
            print(f"[Row {idx}] prompt NOT found in candidate file – skipped.")
            continue

        ref_sents  = split_reference(ref_raw)
        cand_sents = split_candidate(cand_raw)

        if not ref_sents or not cand_sents:
            print(f"[Row {idx}] empty reference or candidate – skipped.")
            continue

        emb_ref = sbert.encode(ref_sents, convert_to_tensor=True,
                               normalize_embeddings=True)
        emb_can = sbert.encode(cand_sents, convert_to_tensor=True,
                               normalize_embeddings=True)

        pairs   = mutual_best_pairs(util.cos_sim(emb_ref, emb_can)
                                    .cpu().numpy(), THRESHOLD)
        matched = {r for r, _, _ in pairs}

        total += 1
        matched_count = len(matched)
        ref_count     = len(ref_sents)

        if matched_count == ref_count:
            hit_all += 1
        partial_score_sum += matched_count / ref_count

        # detailed per‑row log 
        print(f"\n=== Row {idx} ===")
        print(f"[PROMPT] {ref_prompt}")
        print(f"Ref ({len(ref_sents)}):")
        for i, s in enumerate(ref_sents):  print(f"  [R{i+1}] {s}")
        print(f"Cand({len(cand_sents)}):")
        for i, s in enumerate(cand_sents): print(f"  [C{i+1}] {s}")
        for r, c, v in pairs:
            print(f"  ✔ R{r+1} ↔ C{c+1}  sim={v:.3f}")
        print(f"Matched {matched_count}/{ref_count} → "
              f"{'✓ ALL' if matched_count == ref_count else '✗'}")
        print("-" * 60, flush=True)

    #  summary 
    acc_absolute = hit_all / total if total else 0.0
    acc_relative = partial_score_sum / total if total else 0.0

    print("\n==========================")
    print(f"Absolute Accuracy = {acc_absolute:.4f}  ({hit_all}/{total})")
    print(f"Relative Accuracy = {acc_relative:.4f}  (average matched ratio)")
    print("==========================")

if __name__ == "__main__":
    main()