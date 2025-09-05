"""
This script uses a large language model (LLM) to generate natural-language
summaries of value-based comments for each situation in the training dataset.

Workflow:
  1. Load the input CSV containing prompts and answers (situations + comments).
  2. Build structured prompts with an example and instructions for summarization.
  3. Run batched text generation with the specified LLM (temperature, top-p).
  4. Periodically checkpoint results to CSV (with resume support).
  5. Save final summaries in an output CSV for later use (e.g., dataset creation).

Logs are written to a timestamped log file under LOGDIR.
"""

import os
import sys
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

LOGDIR="/p/project1/westai0052/self_rewarding/verl/summary/logs"
os.makedirs(LOGDIR, exist_ok=True)
logfile = os.path.join(LOGDIR, f"summary{datetime.now():%Y%m%d_%H%M%S}.log")
sys.stdout = open(logfile, "w")
sys.stderr = sys.stdout

# --- Config ---
LLM_PATH = "/p/project1/westai0052/self_rewarding/open-r1-main/HF_models_datasets/models/Qwen3-14B"
INPUT_CSV = "/p/project1/westai0052/self_rewarding/verl/benchmark/dataset/final_train.csv"
OUTPUT_CSV = "llm_summaries_tem0.1_topp0.6_full.csv"

MAX_NEW_TOKENS = 550
TEMPERATURE = 0.1
TOP_P = 0.6

# Checkpointing
CHECKPOINT_EVERY = 100     # flush to CSV every N outputs
RESUME = True             # if OUTPUT_CSV exists, skip already-processed ids

# --- Load model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    device_map="auto",
    trust_remote_code=True
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    batch_size=150,  # Adjust batch size as needed
)

def build_prompt(situation: str, output_text: str) -> str:
    example_comments = (
        "1. Compassion, Showing empathy and concern for the well-being of others, in this case, providing a sleeping bag for someone sleeping rough.\n"
        "2. Social justice, Addressing the issue of homelessness and providing basic needs for those in need.\n"
        "3. Well-being, Ensuring that the person sleeping rough has access to a safe and comfortable place to sleep, which can improve their overall health and well-being.\n"
        "4. Responsibility, Taking responsibility for the well-being of others by providing assistance to those in need, such as providing a sleeping bag for someone sleeping rough.\n"
        "5. Autonomy, Respecting the individual's right to make choices about their own life, including providing them with a sleeping bag if they choose to use it.\n"
        "6. Right to shelter, The person sleeping rough has a right to access adequate shelter, which includes a sleeping bag.\n"
        "7. Right to dignity, Providing a sleeping bag for someone sleeping rough can help uphold their dignity and respect their basic human rights.\n"
        "8. Right to safety, Ensuring that the person sleeping rough has access to a safe place to sleep can protect their right to safety.\n"
        "9. Duty to assist, As members of society, we have a moral obligation to help those in need, such as providing a sleeping bag for someone sleeping rough.\n"
        "10. Duty to care for vulnerable individuals, It is our responsibility to provide support and assistance to those who are less fortunate or vulnerable, like someone sleeping rough. (e.g., providing a sleeping bag)."
    )
    example_summary = (
        "Buying a sleeping bag for someone sleeping rough can be viewed from multiple perspectives. "
        "From a compassionate and empathetic standpoint, it reflects genuine concern for another person’s well-being and an effort to ease their hardship. "
        "Through the lens of social justice, it acknowledges the structural issues surrounding homelessness and seeks to meet a basic human need. "
        "In terms of health and well-being, it provides a more comfortable and safer way to sleep, which may improve physical and mental conditions. "
        "This act also reflects a moral responsibility—a duty of care—to support vulnerable members of society. "
        "Furthermore, it touches on fundamental human rights, including the rights to shelter, safety, and dignity. "
        "Finally, it respects the individual’s autonomy by offering them a resource they are free to accept or refuse, supporting their ability to make choices about their own life."
    )
    return (
        "Please summarize the following value-based comments into a natural paragraph that reflects multiple perspectives on the situation. "
        "Ensure that each sentence in your summary corresponds clearly to each of the original comments.\n"
        "Make sure your summary is engaging and written in a conversational yet informative tone, making it accessible and thought-provoking for the reader.\n"
        "Use the example below to guide your style and format.\n\n"
        "### Example\n"
        "**Question:** Remaining silent when questioned by the authorities\n\n"
        "**Comments:**\n"
        f"{example_comments}\n\n"
        "**Summary:**\n"
        f"{example_summary}\n\n"
        "### Now your turn\n"
        f"**Question:** {situation}\n\n"
        "**Comments:**\n"
        f"{output_text}\n\n"
        "**Summary:**"
    )

def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def flush_buffer(buffer_rows, output_csv):
    """Append buffered rows to CSV and clear the buffer."""
    if not buffer_rows:
        return 0
    df_chunk = pd.DataFrame(buffer_rows, columns=["id", "situation", "prompt", "response"])
    file_exists = os.path.exists(output_csv)
    header = not file_exists or os.path.getsize(output_csv) == 0
    df_chunk.to_csv(output_csv, index=False, mode="a", header=header)
    buffer_rows.clear()
    return len(df_chunk)

def load_done_ids(output_csv):
    if RESUME and os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            done = pd.read_csv(output_csv, usecols=["id"])["id"].astype(str).tolist()
            return set(done)
        except Exception as e:
            print(f"Warning: could not read existing {output_csv} to resume ({e}). Starting fresh.")
            return set()
    return set()

def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["prompt", "answer"])
    # df = df[0:10000]
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Resume: skip ids already in OUTPUT_CSV
    done_ids = load_done_ids(OUTPUT_CSV)
    if done_ids:
        print(f"Resuming: found {len(done_ids)} completed ids in {OUTPUT_CSV}")

    rows_buffer = []
    total_written = 0
    prepared_prompts = []
    metadata = []

    # Prepare prompts (skipping done ids)
    for idx, row in df.iterrows():
        ex_id = row.get("id", idx)
        if str(ex_id) in done_ids:
            continue
        situation = row["prompt"]
        output_text = row["answer"]
        prompt = build_prompt(situation, output_text)
        prepared_prompts.append(prompt)
        metadata.append({
            "id": ex_id,
            "situation": situation,
            "prompt": prompt
        })

    print(f"Processing {len(prepared_prompts)} remaining rows (after resume filter)")

    # Batched inference
    batch_size = 150
    processed_since_flush = 0

    for batch_prompts, batch_meta in zip(batch_list(prepared_prompts, batch_size), batch_list(metadata, batch_size)):
        # Build chat-formatted inputs
        model_inputs = []
        for p in batch_prompts:
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": p},
            ]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                model_input = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                model_input = p
            model_inputs.append(model_input)

        # Generate outputs
        outputs = gen(
            model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        # Collect and buffer
        for meta, out_list in zip(batch_meta, outputs):
            generated = out_list[0]["generated_text"].strip()
            print(f"ID: {meta['id']}\nResponse: {generated}\n{'-'*40}")
            rows_buffer.append({
                "id": meta["id"],
                "situation": meta["situation"],
                "prompt": meta["prompt"],
                "response": generated,
            })
            processed_since_flush += 1

            # Checkpoint every N
            if processed_since_flush >= CHECKPOINT_EVERY:
                written = flush_buffer(rows_buffer, OUTPUT_CSV)
                total_written += written
                print(f"[Checkpoint] Appended {written} rows (total written: {total_written})")
                processed_since_flush = 0

    # Final flush
    written = flush_buffer(rows_buffer, OUTPUT_CSV)
    total_written += written
    print(f"Finalized. Appended {written} rows (total written: {total_written}).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure we at least see the error in the log
        print(f"Fatal error: {e}")
        raise
