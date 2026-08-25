"""
WAHM Model Registry
====================

All models evaluated in the WAHM benchmark. Organized into three groups:
  1. Arabic-centric: trained specifically for Arabic
  2. Multilingual: Arabic as one language among many
  3. Frontier: closed-source, largest-scale models

The comparison between groups is the scientific question:
  - Do Arabic-specialized models hallucinate less in dialect?
  - Does the Western base (adapted models) vs from-scratch (Jais) matter?
  - Does frontier scale compensate for less Arabic specialization?

All entries use the OpenAI chat-completions schema. For Azure deployments,
set the base_url to your Azure endpoint. For HuggingFace models served via
OpenRouter/Together, use their respective base URLs.

Access options:
  - Azure AI Foundry: set AZURE_ENDPOINT + AZURE_API_KEY
  - OpenRouter: set OPENROUTER_API_KEY, base_url = https://openrouter.ai/api/v1
  - Together AI: set TOGETHER_API_KEY, base_url = https://api.together.xyz/v1
  - OpenAI direct: set OPENAI_API_KEY
  - Anthropic: set ANTHROPIC_API_KEY (requires anthropic SDK, not openai)
"""

import os

MODELS = {

    # ================================================================
    # GROUP 1: ARABIC-CENTRIC (trained/adapted specifically for Arabic)
    # ================================================================

    "allam-7b": {
        "model_id": "HUMAIN/ALLaM-7B-Instruct",
        "display_name": "ALLaM 7B",
        "family": "arabic_centric",
        "origin": "SDAIA/HUMAIN, Saudi Arabia",
        "approach": "Adapted from Llama, 3T+ tokens, vocabulary expansion",
        "size": "7B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Saudi national model, directly relevant since SDAIA is our institutional partner",
    },

    "fanar-9b": {
        "model_id": "QCRI/Fanar-1-9B",
        "display_name": "Fanar 1 9B",
        "family": "arabic_centric",
        "origin": "QCRI, Qatar",
        "approach": "Continual pretrained from Gemma-2, 1T tokens",
        "size": "9B",
        "base_url_env": "FANAR_BASE_URL",
        "key_env": "FANAR_API_KEY",
        "notes": "Open base checkpoint; evaluate through raw text completion",
    },

    "falcon-arabic-7b": {
        "model_id": "tiiuae/Falcon-H1-Arabic-7B-Instruct",
        "display_name": "Falcon-H1 Arabic 7B",
        "family": "arabic_centric",
        "origin": "TII, UAE",
        "approach": "Adapted from Falcon-3, OALL leaderboard #1",
        "size": "7B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Current top of Open Arabic LLM Leaderboard at 71.47%",
    },

    "jais-8b": {
        "model_id": "core42/jais-2-8b-chat",
        "display_name": "Jais 2 8B",
        "family": "arabic_centric",
        "origin": "Core42/MBZUAI, UAE",
        "approach": "Trained from scratch, 126B Arabic tokens",
        "size": "8B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Only from-scratch model (no Western base), tests inherited-bias hypothesis",
    },

    # ================================================================
    # GROUP 2: MULTILINGUAL (Arabic as one language among many)
    # ================================================================

    "qwen3-8b": {
        "model_id": "qwen/qwen3-8b",
        "display_name": "Qwen3 8B",
        "family": "multilingual",
        "origin": "Alibaba, China",
        "approach": "Multilingual, leads HELM Arabic benchmarks in 8B class",
        "size": "8B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Strong Arabic despite no Arabic-specific training",
    },

    "llama-3.3-70b": {
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "display_name": "Llama 3.3 70B",
        "family": "multilingual",
        "origin": "Meta, USA",
        "approach": "Multilingual, 20+ languages, standard Western baseline",
        "size": "70B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "The standard multilingual baseline, no Arabic specialization",
    },

    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "display_name": "Gemma 2 9B",
        "family": "multilingual",
        "origin": "Google, USA",
        "approach": "Multilingual, base model for Fanar",
        "size": "9B",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Fanar's base — comparing Gemma vs Fanar directly tests what cultural alignment added",
    },

    "gemma-2-9b-base": {
        "model_id": "google/gemma-2-9b",
        "display_name": "Gemma 2 9B Base",
        "family": "multilingual",
        "origin": "Google, USA",
        "approach": "Primarily English-pretrained base checkpoint; no instruction tuning",
        "size": "9B",
        "base_url_env": "GEMMA2_BASE_URL",
        "key_env": "GEMMA2_BASE_API_KEY",
        "notes": "Exact base checkpoint used by Fanar; isolates post-training and cultural alignment effects",
    },

    # ================================================================
    # GROUP 3: FRONTIER (closed-source, largest scale)
    # ================================================================

    "gpt-4o": {
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "family": "frontier",
        "origin": "OpenAI, USA",
        "approach": "Flagship multimodal, largest scale",
        "size": "undisclosed",
        "base_url_env": "AZURE_ENDPOINT",
        "key_env": "AZURE_API_KEY",
        "notes": "Also used as translation model — note confound in paper",
    },

    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "family": "frontier",
        "origin": "OpenAI, USA",
        "approach": "Distilled frontier, cost-efficient",
        "size": "undisclosed",
        "base_url_env": "AZURE_ENDPOINT",
        "key_env": "AZURE_API_KEY",
        "notes": "Tests whether distillation preserves or degrades dialect robustness",
    },

    "claude-sonnet-4": {
        "model_id": "anthropic/claude-sonnet-4",
        "display_name": "Claude Sonnet 4",
        "family": "frontier",
        "origin": "Anthropic, USA",
        "approach": "Constitutional AI, safety-focused training",
        "size": "undisclosed",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Different safety training paradigm than GPT — interesting comparison",
    },

    "gemini-2.5-flash": {
        "model_id": "google/gemini-2.5-flash-preview",
        "display_name": "Gemini 2.5 Flash",
        "family": "frontier",
        "origin": "Google, USA",
        "approach": "Fast frontier, strong multilingual",
        "size": "undisclosed",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Same company as Gemma (Fanar's base) but frontier scale",
    },

    "deepseek-v3": {
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "display_name": "DeepSeek V3",
        "family": "frontier",
        "origin": "DeepSeek, China",
        "approach": "MoE, strong multilingual and reasoning",
        "size": "685B MoE",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "notes": "Chinese-origin frontier model, interesting for cross-cultural comparison",
    },
}

# ================================================================
# Convenience groupings for generate.py
# ================================================================

ARABIC_CENTRIC = [k for k, v in MODELS.items() if v["family"] == "arabic_centric"]
MULTILINGUAL = [k for k, v in MODELS.items() if v["family"] == "multilingual"]
FRONTIER = [k for k, v in MODELS.items() if v["family"] == "frontier"]

# Workshop cut: minimum viable comparison (3 models)
WORKSHOP_CUT = ["allam-7b", "gpt-4o", "falcon-arabic-7b"]

# Full paper: all models
FULL_CUT = list(MODELS.keys())


def print_registry():
    """Print a formatted summary of all models."""
    for group, label in [
        (ARABIC_CENTRIC, "ARABIC-CENTRIC"),
        (MULTILINGUAL, "MULTILINGUAL"),
        (FRONTIER, "FRONTIER"),
    ]:
        print(f"\n{'='*70}")
        print(f"  {label} ({len(group)} models)")
        print(f"{'='*70}")
        for key in group:
            m = MODELS[key]
            print(f"  {key:<22s} {m['display_name']:<25s} {m['size']:<8s} {m['origin']}")
            if m.get("notes"):
                print(f"  {'':22s} -> {m['notes']}")

    total = len(MODELS)
    print(f"\n{'='*70}")
    print(f"  TOTAL: {total} models")
    print(f"  Arabic-centric: {len(ARABIC_CENTRIC)}, Multilingual: {len(MULTILINGUAL)}, Frontier: {len(FRONTIER)}")
    print(f"  Workshop cut: {', '.join(WORKSHOP_CUT)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print_registry()
