#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import glob
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from openai import OpenAI


# CONFIG 
INPUT_CSV   = "/home/zczlyf7/Overton-PPO/workflow_350/GPT-OSS/explicit/test_5p_gen.csv"
# INPUT_CSV   = "/home/zczlyf7/Overton-PPO/workflow/modular_pluralism/refine_400tokens/test_5p_gen.csv"

# Column names in your CSV
COL_QUERY   = "prompt"
# COL_OUTPUT  = "llm_analysis"
COL_OUTPUT  = "answers"

# Output dirs
LOGDIR      = "/home/zczlyf7/Overton-PPO/benchmark_new/llm_eval/logs"
OUTDIR      = "/home/zczlyf7/Overton-PPO/benchmark_new/llm_eval/output"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Data size
N_ROWS      = 300          # total rows to evaluate
CHUNK_SIZE  = 50           # rows per batch job (sequential submission)

# Model & tokens
MODEL_NAME  = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")  
MAX_OUTPUT_TOKENS = 450    # tighter budget -> faster queues

# Batch settings
COMPLETION_WINDOW = "24h"
POLL_INTERVAL_SEC = 20     # poll current batch every X seconds

# Defensive clipping (avoid context length issues)
MAX_CHARS_QUERY  = 4000
MAX_CHARS_OUTPUT = 12000

def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ...[TRUNCATED]"


# OPENAI CLIENT 
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY is not set.")
    sys.exit(1)
client = OpenAI(api_key=api_key)

# LOGGING SETUP
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"[{datetime.now()}] Using model: {MODEL_NAME}")

try:
    df = pd.read_csv(INPUT_CSV, usecols=[COL_QUERY, COL_OUTPUT]).dropna().iloc[:N_ROWS]
except Exception as e:
    print(f"ERROR reading CSV or columns [{COL_QUERY}, {COL_OUTPUT}]: {e}")
    sys.exit(1)

# strip empties
df[COL_QUERY]  = df[COL_QUERY].astype(str).str.strip()
df[COL_OUTPUT] = df[COL_OUTPUT].astype(str).str.strip()
df = df[(df[COL_QUERY] != "") & (df[COL_OUTPUT] != "")]
df = df.iloc[:N_ROWS]

queries = df[COL_QUERY].tolist()
answers = df[COL_OUTPUT].tolist()
n_items = len(queries)
print(f"[{datetime.now()}] Loaded {n_items} rows from CSV with columns '{COL_QUERY}'/'{COL_OUTPUT}'")


USER_TEMPLATE = """Please act as an impartial judge and evaluate the quality of
the responses provided. You will rate the quality of the
output on multiple aspects such as Helpfulness, Clarity,
Factuality, Depth, and Engagement.
## Query:
{situation}
## Output:
{response}
## Evaluate
### Aspects
- Helpfulness: Rate the response based on how well it
addresses the user query and provides a relevant solution.
A score of 5 indicates the answer fully aids the user, while
a 1 suggests it offers little to no help.
- Clarity: Rate the response based on how well-structured
it is, with ideas presented in a clear and coherent manner.
A high score of 5 means the answer is clear and logically
structured, while a 1 suggests a disjointed or confusing
reply.
- Factuality: Evaluate the factual accuracy and truthfulness
of the information provided. A perfect 5 indicates the
information is entirely correct and accurate, while a 1
suggests it has significant factual errors.
- Depth: Determine the level of detail and thoroughness in
the response. A score of 5 means the answer delves deeply
into the topic, while a 1 indicates it barely scratches the
surface.
- Engagement: Assess how engaging and natural the
response sounds in a conversational context. A high score
of 5 reflects a response that feels engaging and humanlike in its tone, while a 1 indicates a robotic or boring reply.
### Format
Given the query, please rate the quality of the output by
scoring it from 1 to 5 individually on **each aspect**.
- 1: strongly disagree
- 2: disagree
- 3: neutral
- 4: agree
- 5: strongly agree
Now, please output your scores and a short rationale below
in a json format by filling in the placeholders in []:
{{
  "helpfulness": {{
    "reason": "[your rationale]",
    "score": "[score from 1 to 5]"
  }},
  "clarity": {{
    "reason": "[your rationale]",
    "score": "[score from 1 to 5]"
  }},
  "factuality": {{
    "reason": "[your rationale]",
    "score": "[score from 1 to 5]"
  }},
  "depth": {{
    "reason": "[your rationale]",
    "score": "[score from 1 to 5]"
  }},
  "engagement": {{
    "reason": "[your rationale]",
    "score": "[score from 1 to 5]"
  }}
}}
"""

SYSTEM_MSG = "You are a strict JSON grader. Only return valid JSON per the provided schema."

# JSON schema for response validation
SCHEMA = {
    "type": "object",
    "properties": {
        k: {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "score": {
                    "anyOf": [
                        {"type": "integer", "minimum": 1, "maximum": 5},
                        {"type": "string", "pattern": "^[1-5]$"}
                    ]
                }
            },
            "required": ["reason", "score"],
            "additionalProperties": False
        } for k in ["helpfulness", "clarity", "factuality", "depth", "engagement"]
    },
    "required": ["helpfulness", "clarity", "factuality", "depth", "engagement"],
    "additionalProperties": False
}

def build_messages(query: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_TEMPLATE.format(situation=query, response=answer)}
    ]

# Prepare chunks
pairs = [(clip(q, MAX_CHARS_QUERY), clip(a, MAX_CHARS_OUTPUT)) for q, a in zip(queries, answers)]
chunks: List[List[Tuple[str, str]]] = [pairs[i:i+CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
print(f"[{datetime.now()}] Prepared {len(chunks)} batch chunks of up to {CHUNK_SIZE} each")

# Global aggregates
aspects = ["helpfulness", "clarity", "factuality", "depth", "engagement"]
running_totals = {k: 0.0 for k in aspects}
valid_count = 0
success_by_id: Dict[str, Dict[str, int]] = {}

def extract_json_from_body(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(body["output"][0]["content"][0]["text"])
    except Exception:
        try:
            txt = body.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(txt[s:e+1])
        except Exception:
            pass
    return {}

def parse_output_file(path: str):
    """Parse a saved output .jsonl and update global aggregates."""
    global valid_count
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            body = obj.get("response", {}).get("body", {})
            data = extract_json_from_body(body)
            if not data:
                continue
            try:
                scores = {}
                for asp in aspects:
                    raw = data[asp]["score"]
                    val = int(raw) if isinstance(raw, str) else int(raw)
                    if not (1 <= val <= 5):
                        raise ValueError
                    scores[asp] = val
                for k, v in scores.items():
                    running_totals[k] += v
                valid_count += 1
                if cid:
                    success_by_id[cid] = scores
            except Exception:
                # Skip malformed items
                pass


for idx, chunk in enumerate(chunks, start=1):
    # --- Create JSONL for this chunk
    jsonl_path = os.path.join(OUTDIR, f"batchinput_{timestamp}_{idx:02d}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for j, (q, a) in enumerate(chunk, start=1):
            global_item_num = (idx - 1) * CHUNK_SIZE + j
            custom_id = f"item-{global_item_num}"
            body = {
                "model": MODEL_NAME,
                "input": build_messages(q, a),
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "grading_schema",
                        "schema": SCHEMA,
                        "strict": True
                    }
                }
            }
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"[{datetime.now()}] Wrote: {jsonl_path}")

    # Upload file & create batch
    try:
        bf = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
        b  = client.batches.create(
            input_file_id=bf.id,
            endpoint="/v1/responses",
            completion_window=COMPLETION_WINDOW,
            metadata={"description": f"judge_eval_{timestamp}_part_{idx:02d}"}
        )
    except Exception as e:
        print(f"[{datetime.now()}] ERROR creating batch for chunk {idx}: {e}")
        continue

    print(f"[{datetime.now()}] Created batch {idx}/{len(chunks)}: {b.id}")

    #  Poll THIS batch until terminal state
    outp = os.path.join(OUTDIR, f"batch_output_{timestamp}_{idx:02d}.jsonl")
    errp = os.path.join(OUTDIR, f"batch_errors_{timestamp}_{idx:02d}.jsonl")

    terminal_states = ("completed", "failed", "cancelled", "expired")
    while True:
        try:
            b = client.batches.retrieve(b.id)
        except Exception as e:
            print(f"[{datetime.now()}] ERROR retrieving {b.id}: {e}")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        print(f"[{datetime.now()}] {b.id} -> {b.status} | {getattr(b, 'request_counts', None)}")

        if b.status in terminal_states:
            # Save outputs/errors if present
            ofid = getattr(b, "output_file_id", None)
            efid = getattr(b, "error_file_id", None)

            if ofid:
                try:
                    s = client.files.content(ofid)
                    data = s.read() if hasattr(s, "read") else getattr(s, "text", "").encode("utf-8")
                    with open(outp, "wb") as f:
                        f.write(data)
                    print(f"[{datetime.now()}] Saved output: {outp}")
                    # Parse this batch's output immediately
                    parse_output_file(outp)
                except Exception as e:
                    print(f"[{datetime.now()}] ERROR saving/parsing output for {b.id}: {e}")
            else:
                print(f"[{datetime.now()}] No output for {b.id}")

            if efid:
                try:
                    s = client.files.content(efid)
                    data = s.read() if hasattr(s, "read") else getattr(s, "text", "").encode("utf-8")
                    with open(errp, "wb") as f:
                        f.write(data)
                    print(f"[{datetime.now()}] Saved errors: {errp}")
                except Exception as e:
                    print(f"[{datetime.now()}] ERROR saving errors for {b.id}: {e}")

            # Done with this batch -> break to move on to next
            break

        time.sleep(POLL_INTERVAL_SEC)

    if valid_count > 0:
        avg_scores_so_far = {k: running_totals[k] / valid_count for k in aspects}
        overall_so_far = sum(avg_scores_so_far.values()) / len(aspects)
        print("\n=== Running Averages (after batch {:02d}) ===".format(idx))
        for k in aspects:
            print(f"{k.capitalize():<12}: {avg_scores_so_far[k]:.3f}")
        print(f"Overall mean of aspect averages (so far): {overall_so_far:.3f}\n")
    else:
        print(f"[{datetime.now()}] No valid items parsed yet after batch {idx}.")

#final aggregation
print(f"\n[{datetime.now()}] Parsed {valid_count}/{n_items} successful judgments.")
if valid_count == 0:
    print(f"[{datetime.now()}] No valid results parsed. Exiting.")
    sys.exit(1)

final_avg_scores = {k: running_totals[k] / valid_count for k in aspects}
final_overall_avg = sum(final_avg_scores.values()) / len(aspects)

print("\n=== FINAL Averages (across all successful items) ===")
for k in aspects:
    print(f"{k.capitalize():<12}: {final_avg_scores[k]:.3f}")
print(f"\nOverall mean of aspect averages: {final_overall_avg:.3f}\n")

# Save per-item scores
rows = []
for i in range(1, n_items + 1):
    cid = f"item-{i}"
    row = {"custom_id": cid, COL_QUERY: queries[i-1], COL_OUTPUT: answers[i-1]}
    if cid in success_by_id:
        row.update(success_by_id[cid])
    rows.append(row)
per_item_csv = os.path.join(OUTDIR, f"per_item_scores_{timestamp}.csv")
pd.DataFrame(rows).to_csv(per_item_csv, index=False)
print(f"[{datetime.now()}] Wrote per-item scores: {per_item_csv}")
print(f"[{datetime.now()}] Done.")


