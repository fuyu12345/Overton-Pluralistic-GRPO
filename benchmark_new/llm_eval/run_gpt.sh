#!/usr/bin/env bash
# ==============================
# Run LLM-as-Judge with Batch API
# ==============================

# --- API key (replace with yours) ---
export OPENAI_API_KEY=""

# --- Optional: choose model ---
# Choices: gpt-4o (stronger, more expensive) or gpt-4o-mini (cheaper, faster)
export JUDGE_MODEL="gpt-4.1"

# --- Run the Python script ---
python gpt.py
