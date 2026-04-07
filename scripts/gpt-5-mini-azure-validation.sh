#!/usr/bin/env bash

set -euo pipefail

MODEL="gpt-5.4"
PROVIDER="azure"
LIMIT="30"
OUTPUT_DIR="${OUTPUT_DIR:-./results/260403-search-gpt-54-value-test}"
ENDPOINT="https://societalllm.openai.azure.com/"
AZURE_API_VERSION="2024-12-01-preview"
CONFIG="search"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
A2A_REASONING_EFFORT="${A2A_REASONING_EFFORT:-${REASONING_EFFORT}}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"

JUDGE_MODEL="gpt-5_2025-08-07"
JUDGE_PROVIDER="azure"
JUDGE_ENDPOINT="https://trapi.research.microsoft.com/gcr/shared"
ENABLE_BOOTSTRAP="${ENABLE_BOOTSTRAP:-1}"
BOOTSTRAP_NUM_RESAMPLES="${BOOTSTRAP_NUM_RESAMPLES:-1000}"
BOOTSTRAP_SAMPLE_RATIO="${BOOTSTRAP_SAMPLE_RATIO:-0.8}"
BOOTSTRAP_CONFIDENCE_LEVEL="${BOOTSTRAP_CONFIDENCE_LEVEL:-0.95}"
BOOTSTRAP_RANDOM_SEED="${BOOTSTRAP_RANDOM_SEED:-42}"

# MAIN_AGENT_VALUE_PROMPT="You value success, competence, and personal accomplishment. You strive to excel, demonstrate ability, and gain recognition for your performance. Your worth is measured by mastery and results."
# SUB_AGENT_VALUE_PROMPT="You value success, competence, and personal accomplishment. You strive to excel, demonstrate ability, and gain recognition for your performance. Your worth is measured by mastery and results."

MAIN_AGENT_VALUE_PROMPT="You value success, competence, and personal accomplishment. You strive to excel, demonstrate ability, and gain recognition for your performance. Your worth is measured by mastery and results."

SUB_AGENT_VALUE_PROMPT="You reject achievement. You avoid competition and status-seeking, embracing simplicity and modesty. You are content without striving for distinction or outperforming others."

if [ -z "${ENDPOINT}" ]; then
    echo "ENDPOINT must be set." >&2
    exit 1
fi

if [ -z "${JUDGE_ENDPOINT}" ]; then
    echo "JUDGE_ENDPOINT must be set." >&2
    exit 1
fi

# Authentication uses Azure AD token flow, not API key.
# Supported options:
# 1. export AZURE_AD_TOKEN=...
# 2. az login
# 3. Managed Identity on the host
export AZURE_API_VERSION

BOOTSTRAP_ARGS=()
case "${ENABLE_BOOTSTRAP}" in
    1|true|TRUE|yes|YES)
        BOOTSTRAP_ARGS+=(
            --enable-bootstrap
            --bootstrap-num-resamples "${BOOTSTRAP_NUM_RESAMPLES}"
            --bootstrap-sample-ratio "${BOOTSTRAP_SAMPLE_RATIO}"
            --bootstrap-confidence-level "${BOOTSTRAP_CONFIDENCE_LEVEL}"
            --bootstrap-random-seed "${BOOTSTRAP_RANDOM_SEED}"
        )
        ;;
esac

REASONING_ARGS=()
if [ -n "${REASONING_EFFORT}" ]; then
    REASONING_ARGS+=(--reasoning_effort "${REASONING_EFFORT}")
fi
if [ -n "${A2A_REASONING_EFFORT}" ]; then
    REASONING_ARGS+=(--a2a_reasoning_effort "${A2A_REASONING_EFFORT}")
fi
if [ -n "${JUDGE_REASONING_EFFORT}" ]; then
    REASONING_ARGS+=(--judge_reasoning_effort "${JUDGE_REASONING_EFFORT}")
fi

are-benchmark run \
    --hf-dataset datasets/gaia2/ \
    --hf-split validation \
    --hf-config "${CONFIG}" \
    --agent default \
    --model "${MODEL}" \
    --provider "${PROVIDER}" \
    --endpoint "${ENDPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_provider "${JUDGE_PROVIDER}" \
    --judge_endpoint "${JUDGE_ENDPOINT}" \
    --a2a_app_prop 1 \
    --a2a_model "${MODEL}" \
    --a2a_model_provider "${PROVIDER}" \
    --a2a_endpoint "${ENDPOINT}" \
    -l "${LIMIT}" \
    --main_agent_value_prompt "${MAIN_AGENT_VALUE_PROMPT}" \
    --sub_agent_value_prompt "${SUB_AGENT_VALUE_PROMPT}" \
    "${REASONING_ARGS[@]}" \
    "${BOOTSTRAP_ARGS[@]}"
    
