import os
import sys
import re
import math
import random
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#  Logging
# LOGDIR = "/p/project1/westai0052/self_rewarding/verl/benchmark_new/logs"
# os.makedirs(LOGDIR, exist_ok=True)
# logfile = os.path.join(LOGDIR, f"GRPO_eval_{datetime.now():%Y%m%d_%H%M%S}.log")
# sys.stdout = open(logfile, "w")
# sys.stderr = sys.stdout

#  Config 
MODEL_DIR   = "/scratch/zczlyf7/HF_models/hf_grpo-qwen2.5-3b-Summary-fullsft-30step"
TEST_CSV    = "/home/zczlyf7/Overton-PPO/benchmark/dataset/test_5p.csv"
OUTPUT_CSV  = "grpo_eval_summaries_clean.csv"

BATCH_SIZE  = 10
MAX_NEW     = 1300
BASE_TEMP   = 0.7
BASE_TOPP   = 0.9
MAX_ATTEMPTS = 4   # total passes (initial + 3 retries)

#  Template
def wrap_prompt(original: str) -> str:
    return (
        f"Provide a structured multi-perspectives analysis on this topic: {original}\n\n"
        "### Output Format\n"
        "<core perspectives>\n"
        "In the perspective of <Perspective name>, <your explanation to this aspect>\n"
        "… (repeat for each perspective)\n"
        "</core perspectives>\n\n"
        "<summary>\n"
        "… (one coherent paragraph weaving together all perspectives in an engaging, conversational yet informative tone)\n"
        "</summary>\n\n"
        "### Task Instructions\n"
        "1. Inside `<core perspectives>…</core perspectives>`, you may have multiple sentences for different perspectives; list each perspective on its own line, using:\n"
        "   `In the perspective of {Perspective name}, {your explanation to this aspect}`\n"
        "2. Inside `<summary>…</summary>`, write a single, natural paragraph that summarizes all of the above perspectives. Ensure that each sentence in your summary corresponds clearly to each of the original core perspective sentence above.\n"
        "3. Only output the two tagged sections exactly as shown—do not add any extra commentary or text.\n"
    )

# Strict checker: core -> summary, adjacent (only whitespace/newlines between), both non-empty, nothing extra after </summary>
STRICT_FORMAT_RE = re.compile(
     r"(?is)<core\s+perspectives>\s*(?P<core>.+?)\s*</core\s+perspectives>\s*<summary>\s*(?P<sum>.+?)\s*</summary>"
)

def extract_blocks_strict(text: str):
    """
    Returns (core, summ) if valid else (None, None).
    """
    if not isinstance(text, str):
        return None, None
    m = STRICT_FORMAT_RE.match(text)
    if not m:
        return None, None
    core = m.group("core").strip()
    summ = m.group("sum").strip()
    if core and summ:
        return core, summ
    return None, None

# Load data 
df_test = pd.read_csv(TEST_CSV, usecols=["prompt"])
prompts = df_test["prompt"].astype(str).tolist()
N = len(prompts)
print(f"[INFO] Loaded {N} prompts from {TEST_CSV}")

# Load model 
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    trust_remote_code=True
)
# Make sure pad token is set
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    model.config.pad_token_id = tokenizer.eos_token_id

def make_model_input(raw: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": raw}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
          
        )
    return raw

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    batch_size=BATCH_SIZE,
)

# Generation with retries 
# Storage for outputs
answer_p = [None] * N
answer_s = [None] * N

remaining = list(range(N))  # indices not yet successfully extracted

def params_for_attempt(attempt_idx: int):
    """
    Slightly vary sampling params on retries to help escape local modes:
    - Increase temperature + small randomness
    - Jitter top_p
    """
    temp = BASE_TEMP + 0.1 * attempt_idx + random.uniform(-0.05, 0.05)
    top_p = min(0.98, max(0.85, BASE_TOPP - 0.02 * attempt_idx + random.uniform(-0.02, 0.02)))
    return max(0.2, temp), top_p

for attempt in range(MAX_ATTEMPTS):
    if not remaining:
        break

    temperature = BASE_TEMP  # always 0.7
    top_p = BASE_TOPP        # always 0.9

    print(f"\n[ATTEMPT {attempt+1}/{MAX_ATTEMPTS}] Remaining: {len(remaining)} | "
          f"temp={temperature:.2f}, top_p={top_p:.2f}")

    for start in range(0, len(remaining), BATCH_SIZE):
        batch_idx = remaining[start : start + BATCH_SIZE]
        batch_prompts = [prompts[i] for i in batch_idx]
        wrapped_inputs = [make_model_input(wrap_prompt(p)) for p in batch_prompts]

        batch_outputs = generator(
            wrapped_inputs,
            max_new_tokens=MAX_NEW,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False,
        )

        newly_success = []
        for pos_in_batch, (prompt_i, out_list) in enumerate(zip(batch_prompts, batch_outputs)):
            idx = batch_idx[pos_in_batch]
            text = out_list[0]["generated_text"] if out_list and len(out_list) > 0 else ""

            print(f"\n=== Prompt idx {idx} ===")
            print(f"Prompt: {prompt_i}")
            print(f"--- Model output (first 500 chars) ---\n{text}\n")

            core, summ = extract_blocks_strict(text)
            if core is not None and summ is not None:
                answer_p[idx] = core
                answer_s[idx] = summ
                newly_success.append(idx)

                print("--- Extracted <core perspectives> ")
                
                print("--- Extracted <summary> ")
               
            else:
                print("[WARN] Failed to extract valid blocks; will retry.")

        if newly_success:
            success_set = set(newly_success)
            remaining = [i for i in remaining if i not in success_set]

    print(f"[ATTEMPT {attempt+1}] Success so far: {N - len(remaining)} / {N}")

# Final report
if remaining:
    print(f"\n[FINAL] Could not extract {len(remaining)} samples after {MAX_ATTEMPTS} attempts. "
          f"Indices: {remaining}")
else:
    print("\n[FINAL] Successfully extracted all samples.")

# Save CSV 
rows = []
for i in range(N):
    rows.append({
        "prompt": prompts[i],
        "answer_p": "" if answer_p[i] is None else answer_p[i],
        "answer_s": "" if answer_s[i] is None else answer_s[i],
        "ok": int(answer_p[i] is not None and answer_s[i] is not None),
    })

df_out = pd.DataFrame(rows, columns=["prompt", "answer_p", "answer_s", "ok"])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df_out)} rows to {OUTPUT_CSV}")
print(f"OK rows: {df_out['ok'].sum()} / {len(df_out)}")
