"""Configuration for the Week 2 Brain/agent loop."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "mock"

    # Model used by the selected provider.
    # For Ollama, this should match `ollama list`.
    model_name: str = "gpt-5"

    # Used by cloud providers such as OpenAI.
    # Ollama does not require an API key.
    api_key_env_var: str = "OPENAI_API_KEY"

    # Local Ollama server.
    ollama_base_url: str = "http://localhost:11434"

    # Prompt templates.
    brain_prompt_template: str = ""
    body_prompt_template: str = ""

    # Maximum number of planning/replanning attempts after the initial plan.
    # Phase-B F1 uses a single closed-loop replan; this remains a hard safety bound.
    max_replanning_attempts: int = 3

    # Per-request HTTP timeout for Ollama / cloud LLM calls (seconds).
    llm_request_timeout_seconds: float = 60.0

    # Episode-level wall-clock budget for all LLM calls combined (seconds).
    max_episode_llm_wall_seconds: float = 180.0
