# construct_triplet_dataset.py
"""
Build an (anchor, positive, negative) triplet dataset from the original valueprism.csv answers.

Pipeline
1. **Sentence extraction** – pull out perspective/stand‑alone sentences from each answer.
2. **Embedding similarity filter** – compute pair‑wise cosine similarity with a SentenceTransformer
   model and keep only pairs whose similarity ≥ SIM_THRESHOLD.
3. **LLM semantic check** – starting from the most similar pair, query the LLM (majority voting
   over *repeat* generations) to decide whether two sentences truly express the same meaning.
   *   If the LLM says **Yes**, we take the first sentence as *anchor* and the second as
       *positive*.
   *   We then iterate through the remaining sentences (sorted by their similarity to the
       anchor, high→low) and ask the LLM again. The first one for which the LLM answers **No** is
       taken as the *negative*.
4. **Dataset write‑out** – store one row per anchor with columns: `prompt`, `anchor`, `positive`,
   `negative`, `row_id` (original dataframe index).

If an answer has < 2 sentences or never yields an anchor–positive match, it is skipped.
"""

# construct_triplet_dataset.py – *Stripped* version
"""
Build an (anchor, positive, negative) triplet dataset **without** the leading
"In the perspective of …" / "From the perspective of …" prefixes. The pipeline
is identical to the previous version, with the additional pre‑processing step:

*Every sentence is normalised by removing the heading clause (up to the first
comma).*
"""

import os
import re
import sys
from datetime import datetime
from itertools import combinations

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM


# CONFIG & LOGGING
LOGDIR = "/home/zczlyf7/Overton-PPO/logs"
os.makedirs(LOGDIR, exist_ok=True)
logfile = os.path.join(
    LOGDIR,
    f"triplet_dataset_filter_paraphrase-mpnet-base-v2_thr=0.75_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
sys.stdout = open(logfile, "w")
sys.stderr = sys.stdout

# nltk resources
nltk.download("punkt")
nltk.download("stopwords")

# paths
CSV_PATH = "/home/zczlyf7/Overton-PPO/eval_LLM_test/valueprism.csv"
OUT_CSV = "/home/zczlyf7/Overton-PPO/train_st/triplet_dataset.csv"

# hyper‑parameters
SIM_THRESHOLD = 0.75  # embedding threshold
EMBED_MODEL = "paraphrase-mpnet-base-v2"
REPEAT_VOTES = 1      # number of generations in majority vote

# devices & models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

LLM_PATH = "/scratch/zczlyf7/HF_models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
llm_model.eval()


# HELPERS
def strip_perspective(sentence: str) -> str:
    """Remove leading perspective clause, e.g.:
    "In the perspective of Justice, XYZ" → "XYZ"""    
    pattern = r"^(?:In|From) the perspective of [^,]+,\s*"
    return re.sub(pattern, "", sentence, flags=re.IGNORECASE)


def extract_sentences(text: str) -> list[str]:
    """Return list of sentences (original extraction rules), *then* strip prefixes."""

    pattern = (
        r"(In the perspective of .*?\.)(?=\s*In the perspective of|\s*From the perspective of|$)"
        r"|(From the perspective of .*?\.)(?=\s*In the perspective of|\s*From the perspective of|$)"
    )
    matches = [m.group(0).strip() for m in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE) if m]
    sentences = matches if matches else nltk.sent_tokenize(text)
    return [strip_perspective(s).strip() for s in sentences]


def extract_keywords(text: str) -> set[str]:
    stop_words = set(stopwords.words("english"))
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {tok for tok in tokens if tok.isalnum() and tok not in stop_words}


def mask_keywords(sentence: str, keywords: set[str], placeholder: str = "[KW]") -> str:
    tokens = re.findall(r"\b\w+\b", sentence)
    return " ".join([placeholder if tok.lower() in keywords else tok for tok in tokens])


def llm_majority_vote(pair: tuple[str, str], repeat: int = REPEAT_VOTES) -> str:
    """Return 'Yes' or 'No' via majority vote over *repeat* deterministic generations."""

    s1, s2 = pair
    prompt_single = (
        "Determine if the following sentence pair expresses the *same* meaning or perspective. "
        "Respond strictly with 'Yes' or 'No'.\n\n"
        f"Sentence A: {s1}\nSentence B: {s2}\nAnswer:"
    )
    prompts = [prompt_single] * repeat
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    votes = []
    for resp in decoded:
        ans = resp.split("Answer:")[-1].strip().split("\n")[0]
        votes.append("Yes" if "yes" in ans.lower() else "No")
    yes_majority = votes.count("Yes") >= (repeat // 2 + 1)
    return "Yes" if yes_majority else "No"

# MAIN
print("Loading CSV …", file=sys.stderr)
df = pd.read_csv(CSV_PATH)
# df = df[0:10]  # Uncomment for quick debug on a subset
print(f"Loaded {len(df)} rows", file=sys.stderr)

triplet_rows: list[dict] = []

for row_id, row in df.iterrows():
    prompt: str = row["prompt"]
    answer: str = row["answer"]

    sentences = extract_sentences(answer)
    if len(sentences) < 2:
        print(f"[{row_id}] <2 sentences – skipped.")
        continue

    # embeddings (mask prompt keywords to avoid leaking topical words)
    keywords = extract_keywords(prompt)
    masked = [mask_keywords(s, keywords) for s in sentences]
    embs = embed_model.encode(masked, convert_to_tensor=True, normalize_embeddings=True)

    # compute all upper‑tri similarities and filter by threshold
    candidate_pairs: list[tuple[int, int, float]] = []  # (i, j, sim)
    for i, j in combinations(range(len(sentences)), 2):
        sim = float(util.cos_sim(embs[i], embs[j]))
        if sim >= SIM_THRESHOLD:
            candidate_pairs.append((i, j, sim))

    if not candidate_pairs:
        print(f"[{row_id}] No pairs ≥ {SIM_THRESHOLD} – skipped.")
        continue

    # sort pairs by sim desc
    candidate_pairs.sort(key=lambda t: t[2], reverse=True)

    anchor_idx = positive_idx = None

    #  find first anchor‑positive match via LLM
    for i, j, sim in candidate_pairs:
        verdict = llm_majority_vote((sentences[i], sentences[j]))
        print(f"[{row_id}] Trying pair (i={i}, j={j}, sim={sim:.3f}) – LLM: {verdict}")
        if verdict == "Yes":
            anchor_idx, positive_idx = i, j
            break

    if anchor_idx is None:
        print(f"[{row_id}] No anchor‑positive confirmed – skipped.")
        continue

    # search for a negative
    anchor_sim_scores = [float(util.cos_sim(embs[anchor_idx], embs[k])) for k in range(len(sentences))]
    rest_indices = [k for k in range(len(sentences)) if k not in {anchor_idx, positive_idx}]
    rest_indices.sort(key=lambda k: anchor_sim_scores[k], reverse=True)

    negative_idx = None
    for k in rest_indices:
        verdict = llm_majority_vote((sentences[anchor_idx], sentences[k]))
        print(f"[{row_id}] Candidate negative k={k}, sim={anchor_sim_scores[k]:.3f} – LLM: {verdict}")
        if verdict == "No":
            negative_idx = k
            break

    if negative_idx is None:
        print(f"[{row_id}] No negative found – skipped (need complete triplet).")
        continue

    triplet_rows.append(
        {
            "row_id": row_id,
            "prompt": prompt,
            "anchor": sentences[anchor_idx],
            "positive": sentences[positive_idx],
            "negative": sentences[negative_idx],
        }
    )
    print(f"[{row_id}] Triplet saved.")

# SAVE
triplet_df = pd.DataFrame(triplet_rows)
triplet_df.to_csv(OUT_CSV, index=False)
print(f"\n Done. Triplet dataset written to {OUT_CSV} (n_rows={len(triplet_df)}).")
