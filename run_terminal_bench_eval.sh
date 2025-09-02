#!/bin/bash

export LITELLM_MODEL="${LITELLM_MODEL:-"anthropic/claude-sonnet-4-20250514"}"
export LITELLM_TEMPERATURE="${LITELLM_TEMPERATURE:-0.1}"


export LITE_LLM_API_KEY="${LITE_LLM_API_KEY}"
export LITE_LLM_API_BASE="${LITE_LLM_API_BASE}"

uv run tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path src.agents.orchestrator_agent:OrchestratorAgent \
    --n-concurrent-trials 5 \
    --n-attempts 5

    # --task-id hello-world