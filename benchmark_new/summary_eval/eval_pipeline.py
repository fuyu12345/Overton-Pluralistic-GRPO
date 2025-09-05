#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
eval_fixed_candidate_against_all_refs.py
Evaluate ONE candidate CSV (prompt, answer_s) against reference test_{label}.csv
for labels in [5p,6p,7p,8p,9p,10p,gt10p]. Saves per-set logs + a summary CSV.
"""
import os, re, csv, pandas as pd, numpy as np, torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import math


CANDIDATE_COL = "answer_p"
# paths & labels
REFERENCE_DIR = "/home/zczlyf7/Overton-PPO/benchmark_new/dataset"
REFERENCE_TPL = "test_{label}.csv"


LLM_MODEL = "Qwen2.5-1.5b"
# MODE="explicit"
# LLM_MODEL= "modular_pluralism"
MODE = "0step"
# CANDIDATE_DIR ="/home/zczlyf7/Overton-PPO/op-grpo-results/qwen2.5-1.5b-im-rewarddown-unique-12k"
# CANDIDATE_DIR = f"/home/zczlyf7/Overton-PPO/workflow/{LLM_MODEL}/{MODE}"
CANDIDATE_DIR = "/home/zczlyf7/Overton-PPO/steps_results/qwen1.5b/0steps"
# CANDIDATE_TPL = "test_{label}_gen.csv"  
CANDIDATE_TPL = "qwen2.5-1.5b-im-{label}.csv"

# LABELS = [ "5p", "6p", "7p", "8p", "9p", "10p","gt10p"]
LABELS = ["5p", "10p"]
SBERT_PATH = "paraphrase-mpnet-base-v2"

THRESHOLD = 0.75

MAX_WORDS = 450
LOGDIR = "eval_logs_steps"
OUT_SUMMARY = f"st-{LLM_MODEL}-{MODE}-t{THRESHOLD}.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── helpers ─────────
_PERSPECTIVE_SPLIT = re.compile(r"(?=\b(?:In|From) the perspective of\b)", re.IGNORECASE)

def split_candidate(text: str):
    text = re.sub(r"\s*\[\d+]", "\n", text)
    return [s.strip() for s in re.split(r"[。！？.!?]\s*", text) if s.strip()]

def split_reference(text: str):
    text = re.sub(r"\s*\[\d+]", "\n", text)
    return [s.strip() for s in _PERSPECTIVE_SPLIT.split(text) if s.strip()]

def mutual_best_pairs(sim: np.ndarray, thr: float):
    M = sim.copy(); pairs = []
    while True:
        max_val = np.nanmax(M)
        if np.isnan(max_val) or max_val < thr: break
        r, c = np.unravel_index(np.nanargmax(M), M.shape)
        if max_val == np.nanmax(M[r, :]) and max_val == np.nanmax(M[:, c]):
            pairs.append((r, c, max_val)); M[r, :], M[:, c] = np.nan, np.nan
        else:
            M[r, c] = np.nan
    return pairs

def candidate_path_for(label: str):
    """Return candidate file path; accept missing .csv for robustness."""
    p_csv = os.path.join(CANDIDATE_DIR, CANDIDATE_TPL.format(label=label))
    if os.path.exists(p_csv):
        return p_csv
    # fallback if someone saved without .csv (e.g., merged_output_gt10p)
    p_noext = p_csv[:-4] if p_csv.endswith(".csv") else p_csv
    return p_noext if os.path.exists(p_noext) else p_csv  # default (may 404)

def evaluate_one_pair(sbert, reference_csv, candidate_csv, label):
    os.makedirs(LOGDIR, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M%S")
    log_path = os.path.join(LOGDIR, f"eval_{label}_{ts}.log")
    lf = open(log_path, "w")

    def log(*a, **k):
        print(*a, **k)
        print(*a, **k, file=lf, flush=True)

    log("=== Static parameters ===")
    for k, v in {
        "REFERENCE_CSV": reference_csv,
        "CANDIDATE_CSV": candidate_csv,
        "SBERT_PATH": SBERT_PATH,
        "THRESHOLD": THRESHOLD,
        "MODE": MODE,
        "MAX_WORDS": MAX_WORDS,
        "DEVICE": DEVICE,
        "LABEL": label
    }.items():
        log(f"{k:>15s} : {v}")
    log()

    # load data
    reference_df = pd.read_csv(reference_csv)
    candidate_df = pd.read_csv(candidate_csv)

    assert {"prompt", "answer"}.issubset(reference_df.columns), \
        f"[{label}] reference CSV must contain prompt & answer columns"
    assert {"prompt", CANDIDATE_COL}.issubset(candidate_df.columns), \
        f"[{label}] candidate CSV must contain prompt & {CANDIDATE_COL} columns"

    cand_map = (candidate_df
                .drop_duplicates("prompt")
                .set_index("prompt")[CANDIDATE_COL]
                .to_dict())

    log(f"Loaded {len(reference_df)} reference samples")
    log(f"Loaded {len(candidate_df)} candidate samples\n")

    total = hit_all = 0
    partial_score_sum = 0.0
    matching_ratios = []  # List to store matching ratios for std. dev. calculation

    for idx, row in reference_df.iterrows():
        ref_prompt = row["prompt"]
        ref_raw    = row["answer"]
        cand_raw   = cand_map.get(ref_prompt)

        if cand_raw is None:
            log(f"[Row {idx}] prompt NOT found in candidate file – skipped.")
            continue

        ref_sents  = split_reference(ref_raw)
        cand_sents = split_candidate(cand_raw)
        if not ref_sents or not cand_sents:
            log(f"[Row {idx}] empty reference or candidate – skipped.")
            continue

        emb_ref = sbert.encode(ref_sents, convert_to_tensor=True, normalize_embeddings=True)
        emb_can = sbert.encode(cand_sents, convert_to_tensor=True, normalize_embeddings=True)

        pairs = mutual_best_pairs(util.cos_sim(emb_ref, emb_can).cpu().numpy(), THRESHOLD)
        matched = {r for r, _, _ in pairs}

        total += 1
        matched_count, ref_count = len(matched), len(ref_sents)
        if matched_count == ref_count:
            hit_all += 1
        partial_score_sum += matched_count / ref_count
        matching_ratios.append(matched_count / ref_count)  # Store matching ratio for std. dev.

        log(f"\n=== Row {idx} ===")
        log(f"[PROMPT] {ref_prompt}")
        log(f"Ref ({len(ref_sents)}):");   [log(f"  [R{i+1}] {s}") for i, s in enumerate(ref_sents)]
        log(f"Cand({len(cand_sents)}):");  [log(f"  [C{i+1}] {s}") for i, s in enumerate(cand_sents)]
        for r, c, v in pairs: log(f"  ✔ R{r+1} ↔ C{c+1}  sim={v:.3f}")
        log(f"Matched {matched_count}/{ref_count} → {'✓ ALL' if matched_count==ref_count else '✗'}")
        log("-"*60)

    acc_absolute = hit_all / total if total else 0.0
    acc_relative = partial_score_sum / total if total else 0.0

    # Calculate standard deviation and standard error
    std_dev = np.std(matching_ratios)
    std_error = std_dev / math.sqrt(total) if total else 0.0

    log("\n==========================")
    log(f"Absolute Accuracy = {acc_absolute:.4f}  ({hit_all}/{total})")
    log(f"Relative Accuracy = {acc_relative:.4f}  (average matched ratio)")
    log(f"Standard Deviation of Matching Ratios = {std_dev:.4f}")
    log(f"Standard Error = {std_error:.4f}")
    log("==========================")

    lf.close()
    return {
        "label": label,
        "reference_csv": reference_csv,
        "candidate_csv": candidate_csv,
        "total": int(total),
        "hit_all": int(hit_all),
        "absolute_accuracy": float(acc_absolute),
        "relative_accuracy": float(acc_relative),
        "std_dev": float(std_dev),
        "std_error": float(std_error),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_path": log_path,
    }

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    sbert = SentenceTransformer(SBERT_PATH, device=DEVICE)

    results = []
    for label in LABELS:
        ref_csv  = os.path.join(REFERENCE_DIR, REFERENCE_TPL.format(label=label))
        cand_csv = candidate_path_for(label)
        if not os.path.exists(ref_csv):
            print(f"[WARN] Reference not found for {label}: {ref_csv} — skipping")
            continue
        if not os.path.exists(cand_csv):
            print(f"[WARN] Candidate not found for {label}: {cand_csv} — skipping")
            continue

        print(f"\n>>> Evaluating {label}")
        res = evaluate_one_pair(sbert, ref_csv, cand_csv, label)
        results.append(res)

    if results:
        out_path = os.path.join(LOGDIR, OUT_SUMMARY)
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "label", "reference_csv", "candidate_csv", "total", "hit_all",
                "absolute_accuracy", "relative_accuracy", "std_dev", "std_error",
                "timestamp", "log_path"
            ])
            writer.writeheader()
            for r in results: writer.writerow(r)
        print(f"\nSaved summary to: {out_path}")
    else:
        print("\nNo results produced — please check input paths.")

if __name__ == "__main__":
    main()
