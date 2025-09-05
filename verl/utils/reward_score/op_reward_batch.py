import re
from collections import Counter
from typing import List, Optional, Dict, Any

import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
for pkg, loc in (("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")):
    try:
        nltk.data.find(loc)
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))

# Regex patterns
_PERSPECTIVE_RE = re.compile(
    r"(?:^|\s)(?:From|In) the perspective of ([^,.\n]+)",
    re.IGNORECASE,
)

_CORE_TAG_RE = re.compile(
    r"<core\s+perspectives>\s*(.*?)\s*</core\s+perspectives>",
    re.IGNORECASE | re.DOTALL,
)

_SUMMARY_TAG_RE = re.compile(
    r"<summary>\s*(.*?)\s*</summary>",
    re.IGNORECASE | re.DOTALL,
)

def _extract_summary(text: str) -> str:
    """Extract the text inside <summary>...</summary>.  
    Fall back to full text if no tag is found."""
    m = _SUMMARY_TAG_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_core(text: str) -> str:
    """Extract the text inside <core perspectives>...</core perspectives>.  
    Fall back to full text if no tag is found."""
    m = _CORE_TAG_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _split_sentences(t: str) -> List[str]:
    """Split text into sentences along 'From/In the perspective of ...' boundaries."""
    pat = re.compile(r"(?:^|\n)(?=(?:From|In) the perspective of )", flags=re.IGNORECASE)
    return [p.strip() for p in pat.split(t + "\n") if p.strip()]


def _extract_perspectives(sentences: List[str]) -> List[str]:
    """Extract the 'XXX' inside '... the perspective of XXX' (in lowercase)."""
    out = []
    for s in sentences:
        m = _PERSPECTIVE_RE.search(s)
        if m:
            out.append(m.group(1).strip().lower())
    return out


#  Reward components 
def _format_reward(cand_sents: List[str]) -> int:
    """
    Assign 0/1/2 points based on how many sentences contain 'From/In the perspective of …':
        ≥70% → 2
        ≥30% → 1
        otherwise → 0
    """
    if not cand_sents:
        return 0
    ratio = sum(bool(_PERSPECTIVE_RE.search(s)) for s in cand_sents) / len(cand_sents)
    if ratio >= 0.7:
        return 2
    if ratio >= 0.3:
        return 1
    return 0


def _repeat_penalty(cand_sents: List[str], cand_persp: List[str]) -> float:
    """
    Apply a penalty for repeated sentences or repeated perspectives:
        -0.5 points for each duplicate (sentence or perspective).
    """
    sent_cnt  = Counter(s.lower() for s in cand_sents)
    persp_cnt = Counter(cand_persp)
    repeats   = sum(max(c - 1, 0) for c in sent_cnt.values()) + \
                sum(max(c - 1, 0) for c in persp_cnt.values())
    return 0.5 * repeats


def _kw_match_penalty(keywords: List[str], summary_text: str) -> float:
    """
    Penalty based on keyword coverage in the summary:
        Coverage ≥ 0.75 →  0
        0.5 ≤ Coverage < 0.75 → -0.5
        0.25 ≤ Coverage < 0.5  → -0.75
        Coverage < 0.25        → -1
    """
    if not keywords:
        return 0.0
    summ_low = summary_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in summ_low)
    rate = hits / len(keywords)
    if rate >= 0.75:
        return 0.0
    if rate >= 0.5:
        return -0.5
    if rate >= 0.25:
        return -0.75
    return -1.0


#Batched scoring
def _split_sentences(t: str):
    """Split text into sentences along 'From/In the perspective of …' boundaries (kept consistent with old code)."""
    pat = re.compile(r"(?:^|\n)(?=(?:From|In) the perspective of )",
                     flags=re.IGNORECASE)
    return [p.strip() for p in pat.split(t + "\n") if p.strip()]


def compute_score(
    data_sources : List[str],
    solution_strs: List[str],
    ground_truths: List[str],               # unused but kept for compatibility
    extra_infos  : List[Optional[Dict[str, Any]]],
) -> List[float]:
    """
    Compute final scores for a batch of candidate texts:
      - Format reward (0–2 points)
      - Minus repeat penalty
      - Plus keyword match penalty
      - Final score clipped at minimum 0
    """
    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    assert len(solution_strs) == len(extra_infos) == len(data_sources)

    scores = []
    debug  = []   # collect first 10 tuples for debugging

    for idx, cand_text in enumerate(solution_strs):
        core_text   = _extract_core(cand_text)
        summary_txt = _extract_summary(cand_text)

        cand_sents  = _split_sentences(core_text)
        persp_list  = _extract_perspectives(cand_sents)

        fmt_reward  = _format_reward(cand_sents)
        repeat_pen  = _repeat_penalty(cand_sents, persp_list)
        kw_penalty  = _kw_match_penalty(persp_list, summary_txt)

        total = max(fmt_reward - repeat_pen + kw_penalty, 0.0)
        scores.append(total)

        # Collect debug info for the first 10 samples
        if idx < 10:
            debug.append((fmt_reward, repeat_pen, kw_penalty, total))

    # Debug print
    if debug:
        print("\n=== Format-Reward Debug (first 10) ===")
        print(" idx | fmt_reward | repeat_pen | kw_penalty | total")
        for i, (f, r, k, t) in enumerate(debug):
            print(f"{i:4d} |    {f:6.2f}   |   {r:6.2f}  |   {k:6.2f}  | {t:6.2f}")
        print("=======================================\n")

    return scores
